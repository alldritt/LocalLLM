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
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.2"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "2.21.2")
    ],
    targets: [
        .target(
            name: "LocalLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXNN", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXRandom", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXLLM", package: "mlx-swift-examples", condition: .when(platforms: [.macOS])),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples", condition: .when(platforms: [.macOS]))
            ]
        ),
        .testTarget(
            name: "LocalLLMTests",
            dependencies: ["LocalLLM"]
        )
    ]
)
