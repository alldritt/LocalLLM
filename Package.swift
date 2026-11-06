// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LocalLLM",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "LocalLLM", targets: ["LocalLLM"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        // LLM libraries moved out of mlx-swift-examples into mlx-swift-lm in
        // 2026; qwen3_5 / qwen3_5_moe support lives there (2.30+).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
        // Image generation lives in the separate LocalImageGen package: its
        // StableDiffusion dependency pins mlx-swift 0.31.x, which would hold every
        // host of this package back from newer mlx-swift / mlx-swift-lm.
        // 3.x hosts supply their own HF downloader + tokenizer (see
        // HuggingFaceIntegration.swift).
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0")
    ],
    targets: [
        .target(
            name: "LocalLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXNN", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXRandom", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXLLM", package: "mlx-swift-lm", condition: .when(platforms: [.macOS])),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm", condition: .when(platforms: [.macOS])),
                .product(name: "HuggingFace", package: "swift-huggingface", condition: .when(platforms: [.macOS])),
                .product(name: "Tokenizers", package: "swift-transformers", condition: .when(platforms: [.macOS]))
            ]
        ),
        .testTarget(
            name: "LocalLLMTests",
            dependencies: ["LocalLLM"]
        )
    ]
)
