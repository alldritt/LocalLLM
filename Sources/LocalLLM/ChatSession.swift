#if os(macOS)
import Foundation
import MLX
import MLXLMCommon

/// Carries the KV cache + a cursor marking how many input tokens are already in it
/// across the multiple passes of one tool-using turn. We reuse the cache so that
/// pass N+1 only processes the newly-appended tool-result tokens instead of re-
/// evaluating the entire transcript from scratch.
private final class ToolLoopState: @unchecked Sendable {
    var cache: [KVCache]?
    /// Number of *prompt* tokens (from the chat template render) already absorbed.
    /// Generated tokens are also in the cache but are counted separately because
    /// the next pass's chat-template render will *replace* them with the same
    /// content rendered into the assistant turn, so they line up identically.
    var consumedInputTokens: Int = 0
}

/// Errors surfaced from `ChatSession.streamResponse`.
public enum ChatSessionError: Error, LocalizedError, Sendable {
    /// Rendered prompt exceeded the configured context window. Host should
    /// either condense the transcript, shorten the user message, or raise
    /// `Configuration.maxContextTokens`.
    case contextOverflow(promptTokens: Int, budget: Int, maxContextTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .contextOverflow(let prompt, let budget, let maxCtx):
            return "Prompt (\(prompt) tokens) exceeds available context budget (\(budget) of \(maxCtx) tokens). Condense the transcript or raise maxContextTokens."
        }
    }
}

public actor ChatSession {
    public private(set) var transcript: [ChatMessage] = []
    /// Hard cap on tool-call passes per user turn — prevents runaway loops if the model
    /// keeps calling tools without ever producing a final answer.
    public var maxToolPasses: Int = 6

    private let llm: LocalLLM

    /// KV cache + consumed-token cursor that survives across user turns within this
    /// session. Invalidated when the transcript prefix is no longer guaranteed to
    /// match (system prompt edit, condense, etc.).
    private let cacheState = ToolLoopState()

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
        invalidateCache()
        if let first = transcript.first, first.role == .system {
            transcript[0].content = prompt
        } else {
            transcript.insert(ChatMessage(role: .system, content: prompt), at: 0)
        }
    }

    /// Drops the persistent KV cache. Used when the transcript prefix changes shape
    /// (system prompt edit, condense, etc.) so the next pass starts from a clean slate.
    public func invalidateCache() {
        cacheState.cache = nil
        cacheState.consumedInputTokens = 0
    }

    public func append(_ message: ChatMessage) {
        transcript.append(message)
    }

    @discardableResult
    public func condense(keepingLastExchanges pairs: Int = 1) -> [ChatMessage] {
        invalidateCache()
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

        // KV cache reuse: passes within this turn AND subsequent turns of this
        // session all share `cacheState`. The cursor advances monotonically.
        let loopState = self.cacheState

        // Look up the on-disk warm-cache URL for this (model, system, tools) combo;
        // the actual load happens inside container.perform in singlePass so the
        // [KVCache] never crosses actor isolation. nil if no warm cache exists.
        let warmCacheURL: URL?
        if loopState.cache == nil {
            let composedSystem = Self.composedSystemContent(
                transcript: self.transcript,
                hasTools: !tools.isEmpty
            )
            if !composedSystem.isEmpty {
                let modelID = await self.llm.configuration.modelID
                let fingerprint = PrefixCacheManager.fingerprint(
                    modelID: modelID,
                    composedSystemPrompt: composedSystem,
                    tools: tools
                )
                warmCacheURL = await PrefixCacheManager.shared.urlIfPresent(fingerprint: fingerprint)
                prefixCacheLog.debug("lookup fingerprint=\(fingerprint.prefix(12), privacy: .public)… url=\(warmCacheURL?.lastPathComponent ?? "MISS", privacy: .public)")
            } else {
                warmCacheURL = nil
            }
        } else {
            warmCacheURL = nil
        }

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
            // Pass the warm-cache URL only on the first pass; once loopState has a
            // cache, subsequent passes just reuse it.
            let url = pass == 0 ? warmCacheURL : nil
            let (rawText, visibleText, calls, info) = try await singlePass(
                tools: tools,
                state: loopState,
                warmCacheURL: url,
                continuation: continuation
            )
            passCount += 1
            promptTokens += info.promptTokenCount
            generationTokens += info.generationTokenCount
            promptSeconds += info.promptTime
            generationSeconds += info.generateTime

            // Store the raw model output (including any <tool_call> tags) so the next
            // pass's chat-template re-render produces tokens that align with the cache.
            // The UI shows `visibleText` via the streamed .text events, not this field.
            self.transcript.append(ChatMessage(role: .assistant, content: rawText))

            if calls.isEmpty {
                if visibleText.isEmpty {
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
        state: ToolLoopState,
        warmCacheURL: URL?,
        continuation: AsyncThrowingStream<LocalLLMEvent, Error>.Continuation
    ) async throws -> (raw: String, text: String, calls: [(name: String, arguments: [String: String])], info: GenerateCompletionInfo) {
        let snapshot = Self.applyingSystemAddenda(to: self.transcript, hasTools: !tools.isEmpty)
        let container = try await self.llm.requireContainer()
        let parameters = await self.llm.generateParameters(forTools: !tools.isEmpty)

        // Capture context-window enforcement values before crossing into the
        // container's isolation domain. Reserve room for the response but never
        // give the output cap more than 25% of the total window.
        let configSnapshot = await self.llm.configuration
        let maxCtx = configSnapshot.maxContextTokens
        let maxOut = configSnapshot.maxTokens
        let promptBudget = max(maxCtx - maxOut, (maxCtx * 3) / 4)

        return try await container.perform { context in
            // Load the on-disk warm cache (if we haven't already) — fresh [KVCache]
            // instances per session so mutation during generation doesn't alias.
            if state.cache == nil, let url = warmCacheURL {
                do {
                    let (loadedCache, tc) = try loadPrefixCacheBundle(url: url)
                    state.cache = loadedCache
                    state.consumedInputTokens = tc
                    prefixCacheLog.debug("loaded url=\(url.lastPathComponent, privacy: .public) promptTokenCount=\(tc) cacheCount=\(loadedCache.count) firstOffset=\(loadedCache.first?.offset ?? -1)")
                } catch {
                    prefixCacheLog.error("load FAILED: \(error.localizedDescription, privacy: .public)")
                }
            }
            let chatMessages = snapshot.map(Self.makeChatMessage)
            let toolSpecs: [[String: Any]]? = tools.isEmpty
                ? nil
                : tools.map { ToolSpecBuilder.toolSpec(for: $0) }
            let fullInput = try await context.processor.prepare(
                input: UserInput(chat: chatMessages, tools: toolSpecs)
            )

            // Figure out what the iterator should actually process this pass.
            // If the cache already holds a prefix matching the front of this
            // pass's tokens, skip those — pass eats only the new tail.
            let fullTokens: MLXArray = fullInput.text.tokens
            let totalCount = fullTokens.dim(-1)

            // Context-window enforcement. Throw before allocating a fresh cache
            // or starting an iterator — if the host doesn't catch and condense,
            // running anyway would either OOM or silently truncate quality.
            if totalCount > promptBudget {
                throw ChatSessionError.contextOverflow(
                    promptTokens: totalCount,
                    budget: promptBudget,
                    maxContextTokens: maxCtx
                )
            }

            let consumed = state.consumedInputTokens
            let canReuse = state.cache != nil && consumed > 0 && consumed < totalCount
            prefixCacheLog.debug("singlePass consumed=\(consumed) totalCount=\(totalCount) canReuse=\(canReuse, privacy: .public)")

            let iterInput: LMInput
            if canReuse {
                let tail = fullTokens[consumed ..< totalCount]
                iterInput = LMInput(text: LMInput.Text(tokens: tail))
            } else {
                state.cache = context.model.newCache(parameters: parameters)
                iterInput = fullInput
            }
            let cache = state.cache!

            let promptStart = Date.timeIntervalSinceReferenceDate

            let additionalEOSTokenIds = Set(
                context.configuration.extraEOSTokens.compactMap {
                    context.tokenizer.convertTokenToId($0)
                }
            )

            var text = ""
            var rawAccumulated = ""
            var calls: [(name: String, arguments: [String: String])] = []
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            let parser = ToolStreamParser()

            var promptTime: TimeInterval = 0
            var generationStart = promptStart
            var generationTokenCount = 0

            // Guard all MLX work in a scoped error handler. Prompt processing
            // (TokenIterator construction) and per-token generation run in the
            // C++ MLX layer, where an invalid configuration — e.g. a max-tokens /
            // context-window combination that produces a bad shape — raises an
            // error. With no handler installed, `ErrorHandler.dispatch` calls
            // `fatalError` and the whole process dies; `withError` converts that
            // same error into a recoverable Swift `throws` (`MLXError.caught`).
            try withError { mlxError in
                let iterator = try TokenIterator(
                    input: iterInput,
                    model: context.model,
                    cache: cache,
                    parameters: parameters
                )

                for token in iterator {
                    if Task.isCancelled { break }
                    if promptTime == 0 {
                        let now = Date.timeIntervalSinceReferenceDate
                        promptTime = now - promptStart
                        generationStart = now
                    }
                    if token == context.tokenizer.unknownTokenId
                        || token == context.tokenizer.eosTokenId
                        || additionalEOSTokenIds.contains(token) {
                        break
                    }
                    detokenizer.append(token: token)
                    generationTokenCount += 1
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
                }

                for output in parser.finish() {
                    if case .text(let s) = output {
                        text += s
                        continuation.yield(.text(s))
                    }
                }

                // MLX runs token generation through asyncEval; synchronize before
                // returning so the cache state is fully written before the next
                // pass reads it. This also flushes any deferred eval error.
                Stream().synchronize()
                try mlxError.check()
            }

            // Advance the cursor: after this pass, the cache holds the entire prompt
            // we just rendered + the tokens we generated. The next pass will re-render
            // a *longer* transcript whose prefix matches up through `totalCount + N`.
            state.consumedInputTokens = totalCount + generationTokenCount

            let generationTime = Date.timeIntervalSinceReferenceDate - generationStart
            let info = GenerateCompletionInfo(
                promptTokenCount: canReuse ? (totalCount - consumed) : totalCount,
                generationTokenCount: generationTokenCount,
                promptTime: promptTime,
                generationTime: generationTime
            )
            return (rawAccumulated, text, calls, info)
        }
    }

    /// Returns the composed system message content (preamble + chat's system prompt)
    /// that `applyingSystemAddenda` would produce, for fingerprinting purposes.
    private static func composedSystemContent(transcript: [ChatMessage], hasTools: Bool) -> String {
        let base = transcript.first(where: { $0.role == .system })?.content ?? ""
        guard hasTools else { return base }
        return ToolPromptDefaults.toolTrustPreamble + "\n\n" + base
    }

    private static func applyingSystemAddenda(to transcript: [ChatMessage], hasTools: Bool) -> [ChatMessage] {
        // No dynamic content injected here — keeps the system prompt byte-stable across
        // app launches so we can persist its KV-cache to disk. The current_time tool
        // remains available for any time-sensitive question.
        guard hasTools else { return transcript }
        let preamble = ToolPromptDefaults.toolTrustPreamble
        var copy = transcript
        if let first = copy.first, first.role == .system {
            copy[0].content = preamble + "\n\n" + first.content
        } else {
            copy.insert(ChatMessage(role: .system, content: preamble), at: 0)
        }
        return copy
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
#endif
