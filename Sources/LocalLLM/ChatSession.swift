import Foundation
import MLXLMCommon

public actor ChatSession {
    public private(set) var transcript: [ChatMessage] = []
    /// Hard cap on tool-call passes per user turn — prevents runaway loops if the model
    /// keeps calling tools without ever producing a final answer.
    public var maxToolPasses: Int = 6

    private let llm: LocalLLM

    public init(llm: LocalLLM, systemPrompt: String? = nil) {
        self.llm = llm
        if let systemPrompt, !systemPrompt.isEmpty {
            transcript.append(ChatMessage(role: .system, content: systemPrompt))
        }
    }

    public init(llm: LocalLLM, transcript: [ChatMessage]) {
        self.llm = llm
        self.transcript = transcript
    }

    public func setSystemPrompt(_ prompt: String) {
        if let first = transcript.first, first.role == .system {
            transcript[0].content = prompt
        } else {
            transcript.insert(ChatMessage(role: .system, content: prompt), at: 0)
        }
    }

    public func append(_ message: ChatMessage) {
        transcript.append(message)
    }

    @discardableResult
    public func condense(keepingLastExchanges pairs: Int = 1) -> [ChatMessage] {
        var kept: [ChatMessage] = []
        if let first = transcript.first, first.role == .system {
            kept.append(first)
        }
        let nonSystem = transcript.filter { $0.role != .system }
        let tail = nonSystem.suffix(pairs * 2)
        kept.append(contentsOf: tail)
        transcript = kept
        return kept
    }

    public func estimatedTokens(charsPerToken: Double) -> Int {
        let chars = transcript.reduce(0) { $0 + $1.content.count }
        return Int((Double(chars) / charsPerToken).rounded())
    }

    /// Streams the model's response to `userText`. When `tools` is non-empty, intercepts
    /// tool calls, executes them, and continues generation until the model returns text
    /// without further tool calls (or `maxToolPasses` is reached).
    public func streamResponse(
        to userText: String,
        tools: [any LocalLLMTool] = []
    ) -> AsyncThrowingStream<LocalLLMEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    self.transcript.append(ChatMessage(role: .user, content: userText))
                    try await self.runToolLoop(tools: tools, continuation: continuation)
                    continuation.yield(.finished)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runToolLoop(
        tools: [any LocalLLMTool],
        continuation: AsyncThrowingStream<LocalLLMEvent, Error>.Continuation
    ) async throws {
        let toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })

        var promptTokens = 0
        var generationTokens = 0
        var promptSeconds: Double = 0
        var generationSeconds: Double = 0
        var passCount = 0

        defer {
            let stats = LocalLLMGenerationStats(
                promptTokens: promptTokens,
                generationTokens: generationTokens,
                promptSeconds: promptSeconds,
                generationSeconds: generationSeconds,
                toolPasses: passCount
            )
            continuation.yield(.stats(stats))
        }

        for pass in 0..<maxToolPasses {
            let (assistantText, calls, info) = try await singlePass(tools: tools, continuation: continuation)
            passCount += 1
            promptTokens += info.promptTokenCount
            generationTokens += info.generationTokenCount
            promptSeconds += info.promptTime
            generationSeconds += info.generateTime

            self.transcript.append(ChatMessage(role: .assistant, content: assistantText))

            if calls.isEmpty {
                if assistantText.isEmpty {
                    continuation.yield(.text(
                        "[The model produced no visible output. Try rephrasing the question.]"
                    ))
                }
                return
            }

            for parsed in calls {
                let call = LocalLLMToolCall(name: parsed.name, arguments: parsed.arguments)
                continuation.yield(.toolCall(call))

                if let tool = toolsByName[call.name] {
                    do {
                        let output = try await tool.execute(arguments: parsed.arguments)
                        self.transcript.append(ChatMessage(role: .tool, content: output))
                        continuation.yield(.toolResult(call: call, content: output, isError: false))
                    } catch {
                        let msg = "Error: \(error.localizedDescription)"
                        self.transcript.append(ChatMessage(role: .tool, content: msg))
                        continuation.yield(.toolResult(call: call, content: msg, isError: true))
                    }
                } else {
                    let msg = "Unknown tool: \(call.name)"
                    self.transcript.append(ChatMessage(role: .tool, content: msg))
                    continuation.yield(.toolResult(call: call, content: msg, isError: true))
                }
            }

            if pass == maxToolPasses - 1 {
                continuation.yield(.text("\n[Tool-call budget exhausted after \(maxToolPasses) passes.]"))
            }
        }
    }

    private func singlePass(
        tools: [any LocalLLMTool],
        continuation: AsyncThrowingStream<LocalLLMEvent, Error>.Continuation
    ) async throws -> (text: String, calls: [(name: String, arguments: [String: String])], info: GenerateCompletionInfo) {
        let snapshot = Self.applyingSystemAddenda(to: self.transcript, hasTools: !tools.isEmpty)
        let container = try await self.llm.requireContainer()
        let parameters = await self.llm.generateParameters(forTools: !tools.isEmpty)

        return try await container.perform { context in
            let chatMessages = snapshot.map(Self.makeChatMessage)
            let toolSpecs: [[String: Any]]? = tools.isEmpty
                ? nil
                : tools.map { ToolSpecBuilder.toolSpec(for: $0) }
            let input = try await context.processor.prepare(
                input: UserInput(chat: chatMessages, tools: toolSpecs)
            )

            var text = ""
            var rawAccumulated = ""
            var calls: [(name: String, arguments: [String: String])] = []
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            let parser = ToolStreamParser()

            let info = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: context
            ) { token in
                if Task.isCancelled { return .stop }
                detokenizer.append(token: token)
                if let chunk = detokenizer.next() {
                    rawAccumulated += chunk
                    for output in parser.ingest(chunk) {
                        switch output {
                        case .text(let s):
                            text += s
                            continuation.yield(.text(s))
                        case .toolCall(let name, let arguments):
                            calls.append((name, arguments))
                        }
                    }
                }
                return .more
            }

            for output in parser.finish() {
                if case .text(let s) = output {
                    text += s
                    continuation.yield(.text(s))
                }
            }

            _ = rawAccumulated
            return (text, calls, info)
        }
    }

    private static func applyingSystemAddenda(to transcript: [ChatMessage], hasTools: Bool) -> [ChatMessage] {
        var addenda: [String] = []
        addenda.append(currentDateLine())
        if hasTools {
            addenda.append(ToolPromptDefaults.toolTrustPreamble)
        }
        let prefix = addenda.joined(separator: "\n\n")

        var copy = transcript
        if let first = copy.first, first.role == .system {
            copy[0].content = prefix + "\n\n" + first.content
        } else {
            copy.insert(ChatMessage(role: .system, content: prefix), at: 0)
        }
        return copy
    }

    private static func currentDateLine() -> String {
        let f = DateFormatter()
        f.locale = .current
        f.timeZone = .current
        f.dateFormat = "EEEE, yyyy-MM-dd HH:mm zzz"
        return "Current date/time: \(f.string(from: Date()))."
    }

    private static func makeChatMessage(_ message: ChatMessage) -> Chat.Message {
        let role: Chat.Message.Role = switch message.role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        case .tool: .tool
        }
        return Chat.Message(role: role, content: message.content)
    }

}
