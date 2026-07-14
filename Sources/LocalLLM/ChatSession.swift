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
    ///
    /// Passing `agentOptions` switches the turn to agent semantics — pseudo-tool
    /// interception, an explicit-finish contract, and typed exhaustion. See
    /// `AgentLoopOptions`. When nil, behavior is exactly the pre-agent loop.
    public func streamResponse(
        to userText: String,
        tools: [any LocalLLMTool] = [],
        agentOptions: AgentLoopOptions? = nil,
        consent: (any AgentConsentDelegate)? = nil
    ) -> AsyncThrowingStream<LocalLLMEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    self.transcript.append(ChatMessage(role: .user, content: userText))
                    try await self.runToolLoop(
                        tools: tools,
                        agentOptions: agentOptions,
                        consent: consent,
                        continuation: continuation
                    )
                    continuation.yield(.finished)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// True after the user grants conversation-scope consent; covers action
    /// tools for the rest of this session.
    private var conversationConsentGranted = false

    private func runToolLoop(
        tools: [any LocalLLMTool],
        agentOptions: AgentLoopOptions?,
        consent: (any AgentConsentDelegate)?,
        continuation: AsyncThrowingStream<LocalLLMEvent, Error>.Continuation
    ) async throws {
        let toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })
        let pseudoByName = Dictionary(
            uniqueKeysWithValues: (agentOptions?.pseudoTools ?? []).map { ($0.name, $0) }
        )
        // The tool list as the model sees it: real tools plus pseudo-tool specs.
        // Used for the spec, the system addenda, and the prefix-cache fingerprint —
        // so agent-mode and plain-mode turns never share a warm cache.
        let specTools: [any LocalLLMTool] =
            tools + (agentOptions?.pseudoTools ?? []).map { $0.specAdapter }
        let passCap = agentOptions?.maxPasses ?? maxToolPasses
        var nudged = false
        let actionToolNames = Set(tools.filter { $0.kind == .action }.map(\.name))
        // Task-scope grant: covers action calls for the rest of this turn's loop.
        var taskConsentGranted = false
        // Accumulated visible text across this turn's passes — the fallback result
        // when a finish pseudo-tool arrives with an empty `result` argument.
        var turnVisibleText = ""

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
                hasTools: !specTools.isEmpty
            )
            if !composedSystem.isEmpty {
                let modelID = await self.llm.configuration.modelID
                let fingerprint = PrefixCacheManager.fingerprint(
                    modelID: modelID,
                    composedSystemPrompt: composedSystem,
                    tools: specTools
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

        for pass in 0..<passCap {
            // Pass the warm-cache URL only on the first pass; once loopState has a
            // cache, subsequent passes just reuse it.
            let url = pass == 0 ? warmCacheURL : nil
            let passResult: (raw: String, text: String,
                             calls: [(name: String, arguments: [String: String])],
                             malformed: [String], info: GenerateCompletionInfo)
            do {
                passResult = try await singlePass(
                    tools: specTools,
                    state: loopState,
                    warmCacheURL: url,
                    continuation: continuation
                )
            } catch let error as ChatSessionError {
                // Agent turns end honestly on context overflow instead of dying
                // as a raw thrown error mid-task; plain chat keeps throwing (its
                // host shows the condense hint).
                guard agentOptions != nil, case .contextOverflow = error else { throw error }
                continuation.yield(.budgetExhausted(passes: passCount, reason: .contextOverflow))
                return
            }
            let (rawText, visibleText, calls, malformed, info) = passResult

            // Bare-text pseudo-call rescue: small models drift out of the
            // <tool_call> format after long text answers and write e.g.
            // `finish {"status": "success"}` as plain text. When a pass has no
            // parsed calls, recognize a trailing bare invocation of a pseudo-tool
            // and route it through the normal interception path. The invocation
            // text is model syntax, not answer content — exclude it from the
            // result-substitution source.
            var effectiveCalls = calls
            var answerText = visibleText
            var rescuedBareCall = false
            if effectiveCalls.isEmpty, malformed.isEmpty, !pseudoByName.isEmpty,
               let bare = Self.trailingBarePseudoCall(
                   in: visibleText, names: Array(pseudoByName.keys)
               ) {
                effectiveCalls.append((bare.name, bare.arguments))
                answerText = bare.strippedText
                rescuedBareCall = true
            }
            // Envelope-less pseudo-call rescue: a common small-model shape is the
            // pseudo-tool name as text followed by a tool_call block holding only
            // the bare arguments object (`finish` text + <tool_call>{{"status":…}}
            // </tool_call>). parseToolCall rejects that block (no name field), so
            // rescue over text + block content before treating it as malformed.
            var rescuedMalformed = false
            if effectiveCalls.isEmpty, !malformed.isEmpty, !pseudoByName.isEmpty {
                let combined = answerText + "\n" + malformed.joined(separator: "\n")
                if let bare = Self.trailingBarePseudoCall(
                    in: combined, names: Array(pseudoByName.keys)
                ) {
                    effectiveCalls.append((bare.name, bare.arguments))
                    answerText = bare.strippedText
                    rescuedBareCall = true
                    rescuedMalformed = true
                }
            }

            if !answerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                turnVisibleText += (turnVisibleText.isEmpty ? "" : "\n") + answerText
            }
            passCount += 1
            promptTokens += info.promptTokenCount
            generationTokens += info.generationTokenCount
            promptSeconds += info.promptTime
            generationSeconds += info.generateTime

            // Store the raw model output (including any <tool_call> tags) so the next
            // pass's chat-template re-render produces tokens that align with the cache.
            // The UI shows `visibleText` via the streamed .text events, not this field.
            self.transcript.append(ChatMessage(role: .assistant, content: rawText))

            // A malformed tool call is neither silence nor a call — feed the parse
            // failure back to the model as a tool error so it can repair, instead
            // of dropping it invisibly (the pass budget still bounds retries).
            if effectiveCalls.isEmpty, !malformed.isEmpty, !rescuedMalformed {
                let msg = "Error: your tool call could not be parsed. Emit exactly "
                    + "one JSON object of the form {\"name\": \"<tool_name>\", "
                    + "\"arguments\": { ... }} inside the tool-call tags, on a single "
                    + "line, with newlines inside string values escaped as \\n."
                self.transcript.append(ChatMessage(role: .tool, content: msg))
                continuation.yield(.toolResult(
                    call: LocalLLMToolCall(name: "invalid_tool_call", arguments: [:]),
                    content: msg,
                    isError: true
                ))
                continue
            }

            if effectiveCalls.isEmpty {
                if let options = agentOptions {
                    // Preferred: don't ask for the sign-off — prefill it. The model
                    // generates only the status choice, so format drift is impossible.
                    if let signoff = options.forcedSignoff,
                       let pseudo = pseudoByName[signoff.toolName] {
                        let micro: (choice: String, rawText: String, counts: (prompt: Int, generation: Int))
                        do {
                            micro = try await forcedSignoffChoice(
                                signoff: signoff, tools: specTools, state: loopState
                            )
                        } catch let error as ChatSessionError {
                            guard case .contextOverflow = error else { throw error }
                            continuation.yield(.budgetExhausted(passes: passCount, reason: .contextOverflow))
                            return
                        }
                        promptTokens += micro.counts.prompt
                        generationTokens += micro.counts.generation
                        // Store the exact prefill+completion so the next pass's
                        // template re-render aligns with what the KV cache holds.
                        self.transcript.append(ChatMessage(role: .assistant, content: micro.rawText))
                        let args = [signoff.parameterName: micro.choice]
                        let call = LocalLLMToolCall(name: signoff.toolName, arguments: args)
                        continuation.yield(.toolCall(call))
                        switch await pseudo.handler(args) {
                        case .continueLoop(let toolResult):
                            self.transcript.append(ChatMessage(role: .tool, content: toolResult))
                            continuation.yield(.toolResult(call: call, content: toolResult, isError: false))
                            continue
                        case .finish(let result, let status):
                            self.transcript.append(ChatMessage(
                                role: .tool,
                                content: "Finished (\(status.rawValue))."
                            ))
                            let trimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
                            let effective = trimmed.isEmpty ? turnVisibleText : result
                            continuation.yield(.agentFinished(
                                result: effective, status: status, route: .forcedSignoff
                            ))
                            return
                        }
                    }
                    // Fallback (no forced sign-off configured): one text nudge,
                    // then honest exhaustion.
                    if !nudged {
                        nudged = true
                        self.transcript.append(ChatMessage(
                            role: .user,
                            content: ToolPromptDefaults.agentFinishNudge
                        ))
                        continue
                    }
                    continuation.yield(.budgetExhausted(passes: passCount, reason: .noFinishCall))
                    return
                }
                if visibleText.isEmpty {
                    continuation.yield(.text(
                        "[The model produced no visible output. Try rephrasing the question.]"
                    ))
                }
                return
            }

            var callObjs = effectiveCalls.map {
                LocalLLMToolCall(name: $0.name, arguments: $0.arguments)
            }

            // Loop-level consent: one batched request per pass for its
            // action-kind calls, unless a task- or conversation-scope grant
            // already covers them. Applies in agent AND plain mode.
            var deniedCallIDs: Set<UUID> = []
            if let consent {
                let actionCalls = callObjs.filter { actionToolNames.contains($0.name) }
                if !actionCalls.isEmpty, !taskConsentGranted, !conversationConsentGranted {
                    let decision = await consent.requestConsent(for: actionCalls)
                    deniedCallIDs = Set(actionCalls.map(\.id))
                        .subtracting(decision.approvedCallIDs)
                    for (idx, call) in callObjs.enumerated() {
                        if let override = decision.argumentOverrides[call.id] {
                            callObjs[idx] = LocalLLMToolCall(
                                id: call.id, name: call.name, arguments: override
                            )
                        }
                    }
                    switch decision.scope {
                    case .thisAction: break
                    case .thisTask: taskConsentGranted = true
                    case .thisConversation: self.conversationConsentGranted = true
                    }
                }
            }

            for call in callObjs {
                continuation.yield(.toolCall(call))

                if deniedCallIDs.contains(call.id) {
                    let msg = "The user DENIED this action. It was NOT performed. "
                        + "Do not claim it was done — adapt your approach or give up honestly."
                    self.transcript.append(ChatMessage(role: .tool, content: msg))
                    continuation.yield(.toolResult(call: call, content: msg, isError: true))
                    continue
                }

                if let pseudo = pseudoByName[call.name] {
                    switch await pseudo.handler(call.arguments) {
                    case .continueLoop(let toolResult):
                        self.transcript.append(ChatMessage(role: .tool, content: toolResult))
                        continuation.yield(.toolResult(call: call, content: toolResult, isError: false))
                    case .finish(let result, let status):
                        // Keep the transcript tool-call/tool-result pattern regular so
                        // a follow-up turn in this session renders coherently.
                        self.transcript.append(ChatMessage(
                            role: .tool,
                            content: "Finished (\(status.rawValue))."
                        ))
                        // Empty result → the turn's streamed text IS the answer.
                        let trimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
                        let effective = trimmed.isEmpty ? turnVisibleText : result
                        continuation.yield(.agentFinished(
                            result: effective,
                            status: status,
                            route: rescuedBareCall ? .bareText : .toolCall
                        ))
                        return
                    }
                } else if let tool = toolsByName[call.name] {
                    do {
                        let output = try await tool.execute(arguments: call.arguments)
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

            if pass == passCap - 1, agentOptions == nil {
                continuation.yield(.text("\n[Tool-call budget exhausted after \(passCap) passes.]"))
            }
        }

        // Agent mode: fell off the pass loop without an explicit finish. A single
        // post-loop check also covers continue-on-final-pass edges (nudge or
        // sign-off bounce on the last iteration).
        if agentOptions != nil {
            continuation.yield(.budgetExhausted(passes: passCount, reason: .passBudget))
        }
    }

    private func singlePass(
        tools: [any LocalLLMTool],
        state: ToolLoopState,
        warmCacheURL: URL?,
        continuation: AsyncThrowingStream<LocalLLMEvent, Error>.Continuation
    ) async throws -> (raw: String, text: String, calls: [(name: String, arguments: [String: String])], malformed: [String], info: GenerateCompletionInfo) {
        let snapshot = Self.applyingSystemAddenda(to: self.transcript, hasTools: !tools.isEmpty)
        let container = try await self.llm.requireContainer()
        let parameters = await self.llm.generateParameters(forTools: !tools.isEmpty)

        // Capture context-window enforcement values before crossing into the
        // container's isolation domain. Reserve room for the response but never
        // give the output cap more than 25% of the total window.
        let configSnapshot = await self.llm.configuration
        let maxCtx = configSnapshot.maxContextTokens
        let maxOut = configSnapshot.maxTokens
        let enableThinking = configSnapshot.enableThinking
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
                input: UserInput(
                    chat: chatMessages,
                    tools: toolSpecs,
                    // Qwen3 reads `enable_thinking` from the template kwargs to emit or
                    // suppress its <think> block; other templates ignore the extra key.
                    additionalContext: ["enable_thinking": enableThinking]
                )
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
            var malformed: [String] = []
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
                            case .malformedToolCall(let raw):
                                malformed.append(raw)
                            }
                        }
                    }
                }

                for output in parser.finish() {
                    switch output {
                    case .text(let s):
                        text += s
                        continuation.yield(.text(s))
                    case .toolCall(let name, let arguments):
                        calls.append((name, arguments))
                    case .malformedToolCall(let raw):
                        malformed.append(raw)
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
            return (rawAccumulated, text, calls, malformed, info)
        }
    }

    /// The forced sign-off micro-pass: re-renders the transcript, appends the
    /// prefilled tool-call tokens (`<tool_call>{"name": …, "arguments": {"status": "`),
    /// and lets the model generate only the choice value — a handful of tokens on
    /// a warm cache. Returns the resolved choice, the exact raw text the model is
    /// considered to have produced (prefill + completion, stored verbatim in the
    /// transcript so the next re-render aligns with the KV cache), and token
    /// counts for stats.
    private func forcedSignoffChoice(
        signoff: ForcedSignoff,
        tools: [any LocalLLMTool],
        state: ToolLoopState
    ) async throws -> (choice: String, rawText: String, counts: (prompt: Int, generation: Int)) {
        let snapshot = Self.applyingSystemAddenda(to: self.transcript, hasTools: !tools.isEmpty)
        let container = try await self.llm.requireContainer()
        let baseParameters = await self.llm.generateParameters(forTools: true)
        let configSnapshot = await self.llm.configuration
        let maxCtx = configSnapshot.maxContextTokens
        let enableThinking = configSnapshot.enableThinking
        let promptBudget = max(maxCtx - configSnapshot.maxTokens, (maxCtx * 3) / 4)

        // Qwen3 thinking templates open every assistant turn with a think block;
        // prefill the empty-think convention so the tool-call prefix stays
        // in-distribution. Rendering must use the SAME enable_thinking as the
        // main passes or the prompt prefix (and KV cache) would diverge.
        let thinkPrefix = enableThinking ? "<think>\n\n</think>\n\n" : ""
        let prefill = thinkPrefix
            + "<tool_call>\n{\"name\": \"\(signoff.toolName)\", "
            + "\"arguments\": {\"\(signoff.parameterName)\": \""

        let parameters: GenerateParameters = {
            var p = baseParameters
            p.maxTokens = 16
            p.temperature = 0
            return p
        }()

        return try await container.perform { context in
            let chatMessages = snapshot.map(Self.makeChatMessage)
            let toolSpecs: [[String: Any]]? = tools.isEmpty
                ? nil
                : tools.map { ToolSpecBuilder.toolSpec(for: $0) }
            let fullInput = try await context.processor.prepare(
                input: UserInput(
                    chat: chatMessages,
                    tools: toolSpecs,
                    additionalContext: ["enable_thinking": enableThinking]
                )
            )
            let renderedTokens: MLXArray = fullInput.text.tokens
            let renderedCount = renderedTokens.dim(-1)
            let prefillIDs = context.tokenizer.encode(text: prefill, addSpecialTokens: false)
            if renderedCount + prefillIDs.count > promptBudget {
                throw ChatSessionError.contextOverflow(
                    promptTokens: renderedCount + prefillIDs.count,
                    budget: promptBudget,
                    maxContextTokens: maxCtx
                )
            }
            let prefillArray = MLXArray(prefillIDs.map { Int32($0) }).asType(renderedTokens.dtype)

            let consumed = state.consumedInputTokens
            let canReuse = state.cache != nil && consumed > 0 && consumed < renderedCount
            let iterTokens: MLXArray
            if canReuse {
                iterTokens = concatenated(
                    [renderedTokens[consumed ..< renderedCount], prefillArray], axis: -1
                )
            } else {
                state.cache = context.model.newCache(parameters: parameters)
                iterTokens = concatenated([renderedTokens, prefillArray], axis: -1)
            }
            let cache = state.cache!

            let additionalEOSTokenIds = Set(
                context.configuration.extraEOSTokens.compactMap {
                    context.tokenizer.convertTokenToId($0)
                }
            )
            var completion = ""
            var generated = 0
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

            try withError { mlxError in
                let iterator = try TokenIterator(
                    input: LMInput(text: LMInput.Text(tokens: iterTokens)),
                    model: context.model,
                    cache: cache,
                    parameters: parameters
                )
                for token in iterator {
                    if Task.isCancelled { break }
                    if token == context.tokenizer.unknownTokenId
                        || token == context.tokenizer.eosTokenId
                        || additionalEOSTokenIds.contains(token) {
                        break
                    }
                    detokenizer.append(token: token)
                    generated += 1
                    if let chunk = detokenizer.next() { completion += chunk }
                    // The choice ends at the value's closing quote.
                    if completion.contains("\"") { break }
                }
                Stream().synchronize()
                try mlxError.check()
            }

            // The cache now holds: full render + prefill + generated completion.
            state.consumedInputTokens = renderedCount + prefillIDs.count + generated

            let value = String(completion.prefix(while: { $0 != "\"" }))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let choice = signoff.choices.first(where: {
                value == $0 || value.hasPrefix($0) || (!value.isEmpty && $0.hasPrefix(value))
            }) ?? signoff.choices[0]

            let promptCount = (canReuse ? renderedCount - consumed : renderedCount) + prefillIDs.count
            return (choice, prefill + completion, (prompt: promptCount, generation: generated))
        }
    }

    /// Finds a pseudo-tool invocation written as plain text — `finish {"status":
    /// "success"}` — in a pass's visible output. Returns the parsed call plus the
    /// text with the invocation (and everything after it, e.g. repeated attempts)
    /// removed. Requires a brace-delimited argument object directly after the
    /// name, so prose mentions of the word never match.
    private static func trailingBarePseudoCall(
        in text: String,
        names: [String]
    ) -> (name: String, arguments: [String: String], strippedText: String)? {
        for name in names {
            var searchRange = text.startIndex..<text.endIndex
            while let nameRange = text.range(of: name, range: searchRange) {
                searchRange = nameRange.upperBound..<text.endIndex
                // Word boundary before the name — don't match "unfinished".
                if nameRange.lowerBound > text.startIndex {
                    let prev = text[text.index(before: nameRange.lowerBound)]
                    if prev.isLetter || prev.isNumber { continue }
                }
                var i = nameRange.upperBound
                while i < text.endIndex, text[i] == " " || text[i] == ":" || text[i] == "\n" {
                    i = text.index(after: i)
                }
                guard i < text.endIndex, text[i] == "{",
                      let close = matchingBrace(in: text, from: i),
                      let args = ToolStreamParser.parseArguments(String(text[i...close]))
                else { continue }
                let stripped = String(text[..<nameRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return (name, args, stripped)
            }
        }
        return nil
    }

    /// Index of the brace closing the object opened at `open`, tracking string
    /// literals so braces inside values don't unbalance the scan.
    private static func matchingBrace(in text: String, from open: String.Index) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var i = open
        while i < text.endIndex {
            let ch = text[i]
            if inString {
                if escaped { escaped = false }
                else if ch == "\\" { escaped = true }
                else if ch == "\"" { inString = false }
            } else {
                switch ch {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 { return i }
                default: break
                }
            }
            i = text.index(after: i)
        }
        return nil
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
