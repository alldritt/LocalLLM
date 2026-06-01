#if os(macOS)
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Hub
import MLX
import StableDiffusion

/// A local text-to-image model that can be downloaded and run via MLX.
public struct ImageModelPreset: Sendable, Identifiable, Hashable {
    public let id: String              // HuggingFace repo ID
    public let displayName: String
    public let expectedDownloadBytes: Int64
    /// Raw value of `StableDiffusionConfiguration.Preset` this maps to.
    let presetKey: String

    public static let sdxlTurbo = ImageModelPreset(
        id: "stabilityai/sdxl-turbo",
        displayName: "SDXL Turbo",
        expectedDownloadBytes: 6_940_000_000,
        presetKey: "sdxl-turbo"
    )

    public static let stableDiffusion21 = ImageModelPreset(
        id: "stabilityai/stable-diffusion-2-1-base",
        displayName: "Stable Diffusion 2.1",
        expectedDownloadBytes: 5_200_000_000,
        presetKey: "base"
    )

    public static let all: [ImageModelPreset] = [.sdxlTurbo, .stableDiffusion21]

    /// Resolve the underlying mlx-swift-examples configuration.
    var sdConfiguration: StableDiffusionConfiguration {
        (StableDiffusionConfiguration.Preset(rawValue: presetKey) ?? .sdxlTurbo).configuration
    }
}

public enum LocalImageGenError: LocalizedError {
    case noOutput
    case encodingFailed
    case unsupportedModel

    public var errorDescription: String? {
        switch self {
        case .noOutput: return "Image generation produced no output."
        case .encodingFailed: return "Failed to encode the generated image as PNG."
        case .unsupportedModel: return "The selected model does not support text-to-image generation."
        }
    }
}

/// Stateful local Stable Diffusion image generator. Mirrors the `LocalLLM`
/// actor: one resident model at a time, lazy download-on-load with disk-polled
/// progress, and an explicit `unload()` so callers can coordinate memory with
/// the text model. Weights are stored alongside the text models under
/// `~/Library/Caches/models/<repo-id>` via a `HubApi` with an explicit
/// download base.
public actor LocalImageGenerator {
    public enum LoadState: Sendable, Equatable {
        case idle
        case downloading(fraction: Double, bytes: Int64, expected: Int64)
        case preparing
        case ready
        case failed(String)
    }

    public struct Configuration: Sendable {
        public var preset: ImageModelPreset
        /// 8-bit quantize the UNet + text encoders. Lower memory, slight quality cost.
        public var quantize: Bool
        /// Override the preset's default step count. `nil` uses the preset default
        /// (SDXL Turbo = 2, SD 2.1 = 50).
        public var steps: Int?
        /// Output edge length in pixels (square). Must be a multiple of 8; the
        /// latent grid is `edge / 8`.
        public var imageEdge: Int

        public init(
            preset: ImageModelPreset = .sdxlTurbo,
            quantize: Bool = false,
            steps: Int? = nil,
            imageEdge: Int = 512
        ) {
            self.preset = preset
            self.quantize = quantize
            self.steps = steps
            self.imageEdge = imageEdge
        }
    }

    public private(set) var configuration: Configuration
    public private(set) var state: LoadState = .idle

    private var generator: TextToImageGenerator?
    /// The preset the resident generator was built for, so a preset change forces a reload.
    private var loadedPresetID: String?
    private var pollTask: Task<Void, Never>?

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    public func updateConfiguration(_ update: (inout Configuration) -> Void) {
        update(&configuration)
    }

    /// Drops the in-memory diffusion model, freeing its weights.
    public func unload() {
        pollTask?.cancel()
        pollTask = nil
        generator = nil
        loadedPresetID = nil
        state = .idle
    }

    public func currentPreset() -> ImageModelPreset {
        configuration.preset
    }

    // MARK: - Storage location (shared with text models)

    /// `HubApi` download base. Matches the MLX text-model location so image and
    /// text weights live side by side under `~/Library/Caches/models/`.
    public nonisolated static var cacheBaseURL: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
    }

    /// On-disk directory for a model's weights: `<cacheBase>/models/<repo-id>`.
    public nonisolated static func modelDirectory(forModelID id: String) -> URL {
        cacheBaseURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
    }

    /// Heuristic for "model is already on disk" — ≥80% of expected bytes present.
    public nonisolated static func isDownloaded(modelID: String, expectedBytes: Int64) -> Bool {
        guard expectedBytes > 0 else { return false }
        let dir = modelDirectory(forModelID: modelID)
        guard FileManager.default.fileExists(atPath: dir.path) else { return false }
        return Double(directorySize(at: dir)) >= Double(expectedBytes) * 0.8
    }

    private nonisolated static func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        // NOTE: do NOT skip hidden files. HuggingFace streams in-progress
        // downloads into a hidden `.cache/huggingface/download/*.incomplete`
        // subdirectory and only moves them to the visible path on completion;
        // skipping hidden files would report 0 bytes for the entire download.
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileSizeKey],
            options: [],
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

    // MARK: - Load / prepare

    /// Download (if needed) and load the configured model into memory without
    /// generating. Drives the same state transitions as `generate()`. Used by
    /// the Settings screen to pre-fetch and surface progress.
    public func prepare() async throws {
        _ = try await loadGenerator()
    }

    private func loadGenerator() async throws -> TextToImageGenerator {
        if let generator, loadedPresetID == configuration.preset.id {
            return generator
        }
        generator = nil

        let preset = configuration.preset
        let sdConfig = preset.sdConfiguration
        let hub = HubApi(downloadBase: Self.cacheBaseURL)
        let expected = preset.expectedDownloadBytes

        state = .downloading(fraction: 0, bytes: 0, expected: expected)
        startDiskPolling(directory: Self.modelDirectory(forModelID: preset.id), expected: expected)

        do {
            // Fetch weights (no-op when already cached). Progress is reported via
            // disk polling rather than the (non-Sendable) hub progress handler.
            try await sdConfig.download(hub: hub)

            pollTask?.cancel()
            pollTask = nil
            state = .preparing

            let loadConfig = LoadConfiguration(float16: true, quantize: configuration.quantize)
            guard let created = try sdConfig.textToImageGenerator(
                hub: hub, configuration: loadConfig
            ) else {
                throw LocalImageGenError.unsupportedModel
            }
            created.ensureLoaded()

            self.generator = created
            self.loadedPresetID = preset.id
            self.state = .ready
            return created
        } catch {
            pollTask?.cancel()
            pollTask = nil
            self.state = .failed(String(describing: error))
            throw error
        }
    }

    private func startDiskPolling(directory: URL, expected: Int64) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                let bytes = Self.directorySize(at: directory)
                await self?.bumpBytes(bytes, expected: expected)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    private func bumpBytes(_ bytes: Int64, expected: Int64) {
        guard case .downloading(let oldFraction, _, _) = state else { return }
        let fractionFromBytes = expected > 0 ? min(Double(bytes) / Double(expected), 0.999) : 0
        let fraction = max(oldFraction, fractionFromBytes)
        state = .downloading(fraction: fraction, bytes: bytes, expected: expected)
    }

    // MARK: - Generation

    /// Generate a PNG image from `prompt`. Downloads/loads the model on first use.
    public func generate(prompt: String, negativePrompt: String = "", seed: UInt64? = nil) async throws -> Data {
        let generator = try await loadGenerator()

        var params = configuration.preset.sdConfiguration.defaultParameters()
        params.prompt = prompt
        params.negativePrompt = negativePrompt
        params.latentSize = [max(8, configuration.imageEdge / 8), max(8, configuration.imageEdge / 8)]
        if let steps = configuration.steps { params.steps = steps }
        if let seed { params.seed = seed }

        // Build + evaluate the denoising graph step by step. Runs on the actor's
        // executor — MLXArrays never escape, so no Sendable concerns.
        let latents = generator.generateLatents(parameters: params)
        var lastXt: MLXArray?
        for xt in latents {
            eval(xt)
            lastXt = xt
        }
        guard let lastXt else { throw LocalImageGenError.noOutput }

        // Decode the final latent into an RGB raster and encode as PNG.
        var decoded = generator.decode(xt: lastXt)
        decoded = (decoded * 255).asType(.uint8).squeezed()
        eval(decoded)
        return try Self.pngData(from: Image(decoded).asCGImage())
    }

    private nonisolated static func pngData(from image: CGImage) throws -> Data {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output as CFMutableData, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw LocalImageGenError.encodingFailed
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw LocalImageGenError.encodingFailed
        }
        return output as Data
    }
}

#endif
