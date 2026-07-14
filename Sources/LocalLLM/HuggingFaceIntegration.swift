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

/// Downloads snapshots into (and prefers) the flat cache layout.
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

        // Complete pre-existing download (config + at least one weights file).
        let hasWeights = fm.fileExists(atPath: flat.appendingPathComponent("config.json").path)
            && ((try? fm.contentsOfDirectory(atPath: flat.path))?
                .contains(where: { $0.hasSuffix(".safetensors") }) ?? false)
        if hasWeights {
            // Older downloads fetched only *.json/*.safetensors and missed
            // standalone chat templates (Qwen 3.5 ships chat_template.jinja).
            // Top up just those files; best-effort so offline use still works
            // for models whose template lives in tokenizer_config.json.
            if !Self.hasChatTemplate(in: flat),
               let snapshot = try? await HubClient().downloadSnapshot(
                   of: repoID,
                   revision: revision ?? "main",
                   matching: ["chat_template*", "*.jinja"],
                   progressHandler: { @MainActor _ in }
               ),
               let items = try? fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil) {
                for item in items {
                    let dest = flat.appendingPathComponent(item.lastPathComponent)
                    // Remove any stale entry (including dangling symlinks, which
                    // fileExists reports as absent but copyItem trips over).
                    try? fm.removeItem(at: dest)
                    // Hub snapshots are symlink farms into a blobs store; copy
                    // the resolved target or the relative link breaks here.
                    try? fm.copyItem(at: item.resolvingSymlinksInPath(), to: dest)
                }
            }
            return flat
        }
        let snapshot = try await HubClient().downloadSnapshot(
            of: repoID,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )

        // Mirror into the flat layout so cache checks, disk polling, and
        // nativeContextWindow all keep working against one location. Hub
        // snapshots are symlink farms into a blobs store — copy resolved
        // targets, never the (relative) links.
        try fm.createDirectory(at: flat, withIntermediateDirectories: true)
        for item in try fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil) {
            let dest = flat.appendingPathComponent(item.lastPathComponent)
            try? fm.removeItem(at: dest)
            try fm.copyItem(at: item.resolvingSymlinksInPath(), to: dest)
        }
        return flat
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
