#if os(macOS)
import Foundation
import HuggingFace
import MLXLMCommon
import Tokenizers

// mlx-swift-lm 3.x decoupled from Hugging Face: hosts supply a Downloader and
// a TokenizerLoader. These are hand-written (rather than the MLXHuggingFace
// macros) so we control two things the macros don't offer:
//
// 1. The flat `~/Library/Caches/models/<repo-id>/` layout — every model the
//    apps have already downloaded lives there; re-downloading 5–20 GB per
//    model on a package migration is not acceptable.
// 2. An `add_generation_prompt` override on applyChatTemplate — the prefix
//    cache prewarm must render the system+tools prefix WITHOUT the assistant
//    generation prompt or the baked KV is not a prefix of real turns.

/// Downloads models into (and prefers) the flat cache layout.
///
/// Files are fetched individually with `transport: .lfs` — the classic resolve
/// endpoint — because `downloadSnapshot` hardwires automatic transport, which
/// routes files >=16 MiB through Hugging Face's Xet protocol; on networks where
/// the Xet CAS endpoint stalls, every large-model download times out at 0%.
/// Per-file downloads with size checks also give us resume and idempotent
/// template top-ups for free.
public struct FlatCacheDownloader: Downloader {
    public init() {}

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let fm = FileManager.default
        let flat = Self.flatDirectory(forModelID: id)

        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw NSError(
                domain: "LocalLLM", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(id)"]
            )
        }

        // Offline fast path: a complete pre-existing download (config + weights
        // + a usable chat template) never touches the network.
        let hasWeights = fm.fileExists(atPath: flat.appendingPathComponent("config.json").path)
            && ((try? fm.contentsOfDirectory(atPath: flat.path))?
                .contains(where: { $0.hasSuffix(".safetensors") }) ?? false)
        if hasWeights && Self.hasChatTemplate(in: flat) {
            return flat
        }

        let client = HubClient()
        let rev = revision ?? "main"
        let entries: [Git.TreeEntry]
        do {
            entries = try await client.listFiles(in: repoID, revision: rev)
        } catch {
            // Listing failed (offline?) — with weights present, let the load
            // proceed; the template may live in tokenizer_config.json.
            if hasWeights { return flat }
            throw error
        }

        // Standalone chat templates ride along regardless of the caller's
        // patterns (Qwen 3.5 ships chat_template.jinja; mlx-lm's patterns
        // don't always include it).
        let wanted = entries.filter { entry in
            Self.matches(entry.path, patterns: patterns)
                || entry.path.hasPrefix("chat_template")
        }

        try fm.createDirectory(at: flat, withIntermediateDirectories: true)
        let totalBytes = Int64(wanted.compactMap(\.size).reduce(0, +))
        var doneBytes: Int64 = 0

        for entry in wanted {
            let dest = flat.appendingPathComponent(entry.path)

            // Already complete (size matches) — skip. This makes interrupted
            // downloads resumable and template top-ups idempotent.
            if let size = entry.size,
               let attrs = try? fm.attributesOfItem(atPath: dest.path),
               (attrs[.size] as? Int) == size {
                doneBytes += Int64(size)
                reportProgress(doneBytes, of: totalBytes, via: progressHandler)
                continue
            }

            try fm.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            try? fm.removeItem(at: dest)
            _ = try await client.downloadFile(
                at: entry.path,
                from: repoID,
                to: dest,
                revision: rev,
                transport: .lfs
            )
            doneBytes += Int64(entry.size ?? 0)
            reportProgress(doneBytes, of: totalBytes, via: progressHandler)
        }
        return flat
    }

    private func reportProgress(
        _ done: Int64, of total: Int64,
        via handler: @Sendable (Progress) -> Void
    ) {
        let progress = Progress(totalUnitCount: max(total, 1))
        progress.completedUnitCount = min(done, max(total, 1))
        handler(progress)
    }

    /// Minimal glob matching for hub file patterns (`*` and `?`); an empty
    /// pattern list matches everything.
    static func matches(_ path: String, patterns: [String]) -> Bool {
        guard !patterns.isEmpty else { return true }
        for pattern in patterns {
            var regex = "^"
            for ch in pattern {
                switch ch {
                case "*": regex += ".*"
                case "?": regex += "."
                default: regex += NSRegularExpression.escapedPattern(for: String(ch))
                }
            }
            regex += "$"
            if path.range(of: regex, options: .regularExpression) != nil {
                return true
            }
        }
        return false
    }

    /// A chat template is available either as a standalone file or embedded in
    /// tokenizer_config.json.
    private static func hasChatTemplate(in directory: URL) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appendingPathComponent("chat_template.jinja").path)
            || fm.fileExists(atPath: directory.appendingPathComponent("chat_template.json").path) {
            return true
        }
        let tokenizerConfig = directory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: tokenizerConfig),
              let text = String(data: data, encoding: .utf8) else { return false }
        return text.contains("\"chat_template\"")
    }

    public static func flatDirectory(forModelID id: String) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return caches
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
    }
}

/// Loads a swift-transformers tokenizer from the model directory and bridges
/// it to MLXLMCommon's Tokenizer protocol.
public struct TransformersTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream: upstream)
    }
}

struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    let upstream: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // Honor an add_generation_prompt override riding in additionalContext —
        // the channel the prefix-cache prewarm uses (see prewarmPrefix). The
        // full overload is a protocol requirement implemented by the concrete
        // tokenizers, so both knobs work through `any Tokenizer`.
        var context = additionalContext ?? [:]
        let addGenerationPrompt = (context.removeValue(forKey: "add_generation_prompt") as? Bool) ?? true
        return try upstream.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: tools,
            additionalContext: context.isEmpty ? nil : context
        )
    }
}
#endif
