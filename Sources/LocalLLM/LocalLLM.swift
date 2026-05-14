import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public struct ModelPreset: Sendable, Identifiable, Hashable {
    public let id: String              // HuggingFace repo ID
    public let displayName: String
    public let parameterLabel: String  // "32B", "14B", etc.
    public let expectedDownloadBytes: Int64

    public init(id: String, displayName: String, parameterLabel: String, expectedDownloadBytes: Int64) {
        self.id = id
        self.displayName = displayName
        self.parameterLabel = parameterLabel
        self.expectedDownloadBytes = expectedDownloadBytes
    }

    public static let qwen2_5_32B_4bit = ModelPreset(
        id: "mlx-community/Qwen2.5-32B-Instruct-4bit",
        displayName: "Qwen 2.5 32B Instruct (4-bit)",
        parameterLabel: "32B",
        expectedDownloadBytes: 18_400_000_000
    )

    public static let qwen2_5_14B_4bit = ModelPreset(
        id: "mlx-community/Qwen2.5-14B-Instruct-4bit",
        displayName: "Qwen 2.5 14B Instruct (4-bit)",
        parameterLabel: "14B",
        expectedDownloadBytes: 8_500_000_000
    )

    public static let qwen2_5_7B_4bit = ModelPreset(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        displayName: "Qwen 2.5 7B Instruct (4-bit)",
        parameterLabel: "7B",
        expectedDownloadBytes: 4_400_000_000
    )

    public static let qwen2_5_3B_4bit = ModelPreset(
        id: "mlx-community/Qwen2.5-3B-Instruct-4bit",
        displayName: "Qwen 2.5 3B Instruct (4-bit)",
        parameterLabel: "3B",
        expectedDownloadBytes: 2_000_000_000
    )

    public static let llama3_2_3B_4bit = ModelPreset(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        displayName: "Llama 3.2 3B Instruct (4-bit)",
        parameterLabel: "3B",
        expectedDownloadBytes: 2_000_000_000
    )

    public static let all: [ModelPreset] = [
        .qwen2_5_32B_4bit,
        .qwen2_5_14B_4bit,
        .qwen2_5_7B_4bit,
        .qwen2_5_3B_4bit,
        .llama3_2_3B_4bit
    ]
}

public struct LocalLLMGenerationStats: Sendable, Hashable {
    public let promptTokens: Int
    public let generationTokens: Int
    public let promptSeconds: Double
    public let generationSeconds: Double
    public let toolPasses: Int

    public var promptTokensPerSecond: Double {
        promptSeconds > 0 ? Double(promptTokens) / promptSeconds : 0
    }
    public var generationTokensPerSecond: Double {
        generationSeconds > 0 ? Double(generationTokens) / generationSeconds : 0
    }
    public var totalSeconds: Double { promptSeconds + generationSeconds }
}

public actor LocalLLM {
    public struct Configuration: Sendable {
        public var modelID: String
        public var maxTokens: Int
        public var temperature: Float
        public var topP: Float
        public var maxContextTokens: Int
        public var estimatedCharsPerToken: Double
        /// Lower temperature used during turns that have tools registered. Qwen and other
        /// tool-calling models emit more reliable JSON at lower sampling temperatures.
        public var toolTemperature: Float
        /// Rough total download size for the configured model. Used only to drive the progress bar
        /// because the HF Hub progress is file-count-weighted and crawls inside large shards.
        public var expectedDownloadBytes: Int64

        public init(
            modelID: String = "mlx-community/Qwen2.5-32B-Instruct-4bit",
            maxTokens: Int = 2048,
            temperature: Float = 0.7,
            topP: Float = 0.95,
            maxContextTokens: Int = 8192,
            estimatedCharsPerToken: Double = 3.5,
            expectedDownloadBytes: Int64 = 18_400_000_000,
            toolTemperature: Float = 0.2
        ) {
            self.modelID = modelID
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.maxContextTokens = maxContextTokens
            self.estimatedCharsPerToken = estimatedCharsPerToken
            self.expectedDownloadBytes = expectedDownloadBytes
            self.toolTemperature = toolTemperature
        }
    }

    public enum LoadState: Sendable, Equatable {
        case idle
        case downloading(fraction: Double, bytes: Int64, expected: Int64)
        case preparing
        case ready
        case failed(String)
    }

    public private(set) var configuration: Configuration
    public private(set) var state: LoadState = .idle

    private var container: ModelContainer?
    private var pollTask: Task<Void, Never>?

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
        MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)
    }

    public func load(
        progress: (@Sendable (LoadState) -> Void)? = nil
    ) async throws -> ModelContainer {
        if let container { return container }

        let expected = configuration.expectedDownloadBytes
        state = .downloading(fraction: 0, bytes: 0, expected: expected)
        progress?(state)

        let cacheDir = modelCacheDirectory()
        startDiskPolling(directory: cacheDir, expected: expected, progress: progress)

        do {
            let modelConfig = ModelConfiguration(id: configuration.modelID)
            let loaded = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfig
            ) { p in
                let fraction = p.fractionCompleted
                Task { [weak self] in
                    await self?.bumpFraction(fraction)
                }
            }

            pollTask?.cancel()
            pollTask = nil

            state = .preparing
            progress?(state)

            self.container = loaded
            self.state = .ready
            progress?(state)
            return loaded
        } catch {
            pollTask?.cancel()
            pollTask = nil
            self.state = .failed(String(describing: error))
            progress?(state)
            throw error
        }
    }

    public func updateConfiguration(_ update: (inout Configuration) -> Void) {
        update(&configuration)
    }

    /// Drops the currently-loaded model and loads the chosen preset. The progress
    /// callback receives the same LoadState transitions as the initial `load()`.
    public func switchModel(
        to preset: ModelPreset,
        progress: (@Sendable (LoadState) -> Void)? = nil
    ) async throws {
        guard preset.id != configuration.modelID || container == nil else { return }
        container = nil
        configuration.modelID = preset.id
        configuration.expectedDownloadBytes = preset.expectedDownloadBytes
        state = .idle
        _ = try await load(progress: progress)
    }

    public func currentPreset() -> ModelPreset? {
        ModelPreset.all.first { $0.id == configuration.modelID }
    }

    /// Builds the KV cache for the static prefix (system + tools) and persists it to disk.
    /// Cheap no-op if a cache file already exists for this (model, system, tools) combination.
    /// Safe to call repeatedly; intended to run in the background after `load()` and after
    /// `switchModel(...)`.
    public func prewarmPrefix(
        composedSystemPrompt: String,
        tools: [any LocalLLMTool]
    ) async {
        let fingerprint = PrefixCacheManager.fingerprint(
            modelID: configuration.modelID,
            composedSystemPrompt: composedSystemPrompt,
            tools: tools
        )
        if await PrefixCacheManager.shared.exists(fingerprint: fingerprint) {
            print("[prefix-cache] prewarm skipped (already present) fingerprint=\(fingerprint.prefix(12))…")
            return
        }
        guard let container else {
            print("[prefix-cache] prewarm aborted (container nil)")
            return
        }
        print("[prefix-cache] prewarm starting fingerprint=\(fingerprint.prefix(12))…")

        try? await container.perform { context in
            let toolSpecs = tools.isEmpty
                ? nil
                : tools.map { ToolSpecBuilder.toolSpec(for: $0) }
            let messages: [[String: Any]] = [["role": "system", "content": composedSystemPrompt]]
            let promptTokens: [Int]
            do {
                promptTokens = try context.tokenizer.applyChatTemplate(
                    messages: messages,
                    chatTemplate: nil,
                    addGenerationPrompt: false,
                    truncation: false,
                    maxLength: nil,
                    tools: toolSpecs
                )
            } catch {
                return
            }

            let lmInput = LMInput(tokens: MLXArray(promptTokens))
            let cache = context.model.newCache(parameters: nil)
            // Constructing the iterator runs the chunked prefill into `cache`; we never
            // call next() so no tokens are generated — cache holds exactly the prefix.
            _ = try TokenIterator(
                input: lmInput,
                model: context.model,
                cache: cache,
                parameters: GenerateParameters()
            )
            // Drain all pending GPU work into materialized arrays BEFORE save. Without
            // this `save()` internally calls eval() which collides with the in-flight
            // asyncEval from the iterator's prepare step, crashing with
            //   "A command encoder is already encoding to this command buffer".
            let stateArrays = cache.flatMap { $0.state }
            eval(stateArrays)
            Stream().synchronize()
            let destination = await PrefixCacheManager.shared.reserveURL(for: fingerprint)
            do {
                try savePrefixCacheBundle(
                    url: destination,
                    cache: cache,
                    promptTokenCount: promptTokens.count
                )
                print("[prefix-cache] prewarm saved tokens=\(promptTokens.count) -> \(destination.lastPathComponent)")
            } catch {
                print("[prefix-cache] prewarm save FAILED: \(error)")
            }
        }
    }

    func generateParameters(forTools: Bool = false) -> GenerateParameters {
        GenerateParameters(
            maxTokens: configuration.maxTokens,
            temperature: forTools ? configuration.toolTemperature : configuration.temperature,
            topP: configuration.topP
        )
    }

    func requireContainer() async throws -> ModelContainer {
        if let container { return container }
        return try await load()
    }

    /// `~/Library/Caches/huggingface/models--<org>--<repo>/` — matches `swift-transformers` layout.
    private func modelCacheDirectory() -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let slug = configuration.modelID.replacingOccurrences(of: "/", with: "--")
        return caches
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("models--\(slug)", isDirectory: true)
    }

    private func startDiskPolling(
        directory: URL,
        expected: Int64,
        progress: (@Sendable (LoadState) -> Void)?
    ) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let bytes = Self.directorySize(at: directory)
                await self?.bumpBytes(bytes, expected: expected, progress: progress)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    private func bumpBytes(
        _ bytes: Int64,
        expected: Int64,
        progress: (@Sendable (LoadState) -> Void)?
    ) {
        guard case .downloading(let oldFraction, _, _) = state else { return }
        let fractionFromBytes = expected > 0 ? min(Double(bytes) / Double(expected), 0.999) : 0
        let fraction = max(oldFraction, fractionFromBytes)
        state = .downloading(fraction: fraction, bytes: bytes, expected: expected)
        progress?(state)
    }

    private func bumpFraction(_ fraction: Double) {
        guard case .downloading(let oldFraction, let bytes, let expected) = state else { return }
        let merged = max(oldFraction, fraction)
        state = .downloading(fraction: merged, bytes: bytes, expected: expected)
    }

    private static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileSizeKey],
            options: [.skipsHiddenFiles],
            errorHandler: nil
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileSizeKey])
            if let s = values?.totalFileAllocatedSize {
                total += Int64(s)
            } else if let s = values?.fileSize {
                total += Int64(s)
            }
        }
        return total
    }
}
