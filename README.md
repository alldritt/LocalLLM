# LocalLLM

A small Swift package that wraps [MLX Swift](https://github.com/ml-explore/mlx-swift) +
[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) behind a chat-shaped API
modelled on Apple's `LanguageModelSession`. macOS only.

Default model: **`mlx-community/Qwen2.5-32B-Instruct-4bit`** (~18 GB on disk, ~20–25 tok/s on
an M1 Max 64 GB; needs ~22 GB of resident memory once warm).

## Usage

```swift
import LocalLLM

let llm = LocalLLM()                          // defaults to Qwen 2.5 32B 4-bit

// First call downloads the model into ~/Library/Caches/huggingface/...
// Hand a progress closure if you want to render a load bar.
_ = try await llm.load { fraction in
    print("downloading: \(Int(fraction * 100))%")
}

let session = ChatSession(llm: llm, systemPrompt: "You are a concise assistant.")

for try await chunk in await session.streamResponse(to: "Summarize Apple Silicon in one sentence.") {
    print(chunk, terminator: "")
}
```

## Picking a different model

```swift
let llm = LocalLLM(configuration: .init(
    modelID: "mlx-community/Llama-3.3-70B-Instruct-4bit",
    maxTokens: 4096,
    temperature: 0.6
))
```

Any model in the `mlx-community` HuggingFace org that the MLXLLM factory supports will work.

## Consuming from an app

Add as a local path dependency in your app's `project.yml` (XcodeGen) or directly in Xcode:

```yaml
packages:
  LocalLLM:
    path: ../LocalLLM
targets:
  YourApp:
    dependencies:
      - package: LocalLLM
```

## Entitlements

For sandboxed macOS apps you need:

- **Outgoing Network Connections** — only required for the first-run model download from HuggingFace.
- **Hardened Runtime → JIT (`com.apple.security.cs.allow-jit`)** — Metal compute pipelines.

No "Increased Memory Limit" entitlement is needed on macOS; that's iOS-only.

## Model storage

Weights are cached under `~/Library/Caches/huggingface/models--mlx-community--Qwen2.5-32B-Instruct-4bit/`
by `swift-transformers`. Delete that directory to force a re-download.

## What this package is *not*

- No tool-calling abstraction yet (MLXLMCommon ships `ToolCallProcessor` — `ChatSession` ignores
  `.toolCall` events for now).
- No persistence. The host app owns chat history (see FoundationTest for a working example).
- No context-window management. Qwen 2.5 32B has a 32k window but RAM is the real constraint.
