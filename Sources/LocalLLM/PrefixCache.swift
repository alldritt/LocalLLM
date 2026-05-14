import Foundation
import CryptoKit
import MLX
import MLXLMCommon
import os

/// Shared logger for the prefix-cache subsystem. Silent by default; stream with:
///   `log stream --subsystem com.localllm --category prefix-cache`
let prefixCacheLog = Logger(subsystem: "com.localllm", category: "prefix-cache")

/// Writes a `[KVCache]` to disk in a simple safetensors layout that round-trips
/// correctly for KVCacheSimple. We can't use MLXLMCommon.savePromptCache because
/// loadPromptCache rejects files whose caches had empty metaState (which is the
/// default for KVCacheSimple) — the count-mismatch check fires.
///
/// Writes are atomic: we save to `<url>.tmp` and rename on success, so a crash
/// during save never leaves a zero-byte or half-written file at the real path.
func savePrefixCacheBundle(url: URL, cache: [KVCache], promptTokenCount: Int) throws {
    var arrays: [String: MLXArray] = [:]
    var metadata: [String: String] = [
        "version": "1",
        "cacheCount": String(cache.count),
        "promptTokenCount": String(promptTokenCount)
    ]
    for (i, c) in cache.enumerated() {
        let state = c.state
        metadata["stateCount.\(i)"] = String(state.count)
        for (j, arr) in state.enumerated() {
            arrays["c\(i).\(j)"] = arr
        }
    }
    // MLX's save(arrays:metadata:url:) only accepts ".safetensors" or ".npz" extensions,
    // so the tmp file has to keep the same extension. Use ".partial.safetensors".
    let tmp = url.deletingPathExtension().appendingPathExtension("partial.safetensors")
    try? FileManager.default.removeItem(at: tmp)
    try save(arrays: arrays, metadata: metadata, url: tmp)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
    try FileManager.default.moveItem(at: tmp, to: url)
}

/// Reads a file written by `savePrefixCacheBundle` back into fresh KVCacheSimple objects.
/// Returns nil if the file is unreadable or in an unexpected format.
func loadPrefixCacheBundle(url: URL) throws -> (cache: [KVCache], promptTokenCount: Int) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)
    guard let countStr = metadata["cacheCount"], let count = Int(countStr), count > 0 else {
        throw PrefixCacheError.malformed("missing or invalid cacheCount")
    }
    let promptTokenCount = Int(metadata["promptTokenCount"] ?? "") ?? 0
    var caches: [KVCache] = []
    for i in 0..<count {
        let stateCount = Int(metadata["stateCount.\(i)"] ?? "") ?? 2
        var stateArrays: [MLXArray] = []
        for j in 0..<stateCount {
            guard let arr = arrays["c\(i).\(j)"] else {
                throw PrefixCacheError.malformed("missing array c\(i).\(j)")
            }
            stateArrays.append(arr)
        }
        let cache = KVCacheSimple()
        cache.state = stateArrays
        caches.append(cache)
    }
    return (caches, promptTokenCount)
}

enum PrefixCacheError: LocalizedError {
    case malformed(String)
    var errorDescription: String? {
        if case .malformed(let s) = self { return "Prefix cache malformed: \(s)" }
        return nil
    }
}

/// Persists the KV cache for a chat's static prefix (system message + tool advertisements)
/// to disk so first-turn cold prompt eval is avoided after the first time the prefix is seen.
///
/// The cache is keyed by a fingerprint of (model ID + composed system prompt + sorted tool
/// definitions). Anything that changes those bytes invalidates the cache file.
public actor PrefixCacheManager {
    public static let shared = PrefixCacheManager()

    /// `~/Library/Application Support/MLXChatTest/prefix-cache/`
    private let directory: URL

    /// Keep at most this many cache files. Oldest by access time get evicted.
    private let maxEntries: Int = 8

    init() {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let bundle = (Bundle.main.bundleIdentifier ?? "LocalLLM")
        self.directory = base
            .appendingPathComponent(bundle, isDirectory: true)
            .appendingPathComponent("prefix-cache", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
    }

    /// Computes a fingerprint for the static prefix. Identical inputs produce identical fingerprints,
    /// allowing cross-session and cross-launch cache hits.
    public static func fingerprint(
        modelID: String,
        composedSystemPrompt: String,
        tools: [any LocalLLMTool]
    ) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(modelID.utf8))
        hasher.update(data: Data([0]))
        hasher.update(data: Data(composedSystemPrompt.utf8))
        hasher.update(data: Data([0]))
        for tool in tools.sorted(by: { $0.name < $1.name }) {
            hasher.update(data: Data(tool.name.utf8))
            hasher.update(data: Data(tool.description.utf8))
            for p in tool.parameters.sorted(by: { $0.name < $1.name }) {
                hasher.update(data: Data(p.name.utf8))
                hasher.update(data: Data(p.kind.rawValue.utf8))
                hasher.update(data: Data(p.description.utf8))
                hasher.update(data: [p.isRequired ? 1 : 0])
            }
            hasher.update(data: Data([0]))
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    func cacheURL(for fingerprint: String) -> URL {
        directory.appendingPathComponent("\(fingerprint).safetensors")
    }

    public func exists(fingerprint: String) -> Bool {
        let url = cacheURL(for: fingerprint)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        // Treat zero-byte files as absent — leftover from a crashed save would otherwise
        // poison the cache directory and silently disable warmup.
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return size > 0
    }

    /// Returns the on-disk URL for this fingerprint if a cached file exists. The caller
    /// performs the actual `loadPromptCache` inside whatever isolation domain owns the
    /// resulting `[KVCache]` (which is not Sendable, so it shouldn't cross actors).
    public func urlIfPresent(fingerprint: String) -> URL? {
        guard exists(fingerprint: fingerprint) else { return nil }
        let url = cacheURL(for: fingerprint)
        touch(url)
        return url
    }

    /// Reserves the destination URL for a save and runs LRU eviction. Caller writes
    /// the cache there with `savePromptCache(url:cache:metadata:)` — kept outside this
    /// actor because `[KVCache]` is not Sendable and can't cross actor isolation.
    public func reserveURL(for fingerprint: String) -> URL {
        evictIfNeeded()
        return cacheURL(for: fingerprint)
    }

    private func touch(_ url: URL) {
        try? FileManager.default.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: url.path
        )
    }

    private func evictIfNeeded() {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        let safetensors = entries.filter {
            $0.pathExtension == "safetensors"
                && !$0.deletingPathExtension().pathExtension.hasSuffix("partial")
        }
        guard safetensors.count > maxEntries else { return }
        let sorted = safetensors.sorted { lhs, rhs in
            let lDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let rDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return lDate < rDate
        }
        for url in sorted.prefix(safetensors.count - maxEntries) {
            try? FileManager.default.removeItem(at: url)
        }
    }
}
