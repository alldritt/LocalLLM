#if os(macOS)
import Foundation

/// How an agent-mode turn ended when a pseudo-tool completed the loop.
public enum FinishStatus: String, Sendable {
    case success
    case gaveUp = "gave_up"
}

/// Why an agent-mode turn ended without a pseudo-tool finishing it.
public enum ExhaustionReason: String, Sendable {
    /// The configured `maxPasses` ceiling was hit.
    case passBudget = "pass_budget"
    /// The model produced two tool-call-free passes (one nudge spent) without
    /// ever calling `finish`.
    case noFinishCall = "no_finish_call"
    /// The rendered prompt outgrew the context window mid-turn. Agent turns end
    /// honestly with this instead of dying as a raw thrown error.
    case contextOverflow = "context_overflow"
}

/// What a pseudo-tool handler wants the loop to do next.
public enum PseudoToolOutcome: Sendable {
    /// Feed `toolResult` back to the model as an ordinary tool result and continue.
    case continueLoop(toolResult: String)
    /// End the loop now. Surfaces as `.agentFinished(result:status:)`. When
    /// `result` is empty, the loop substitutes the turn's streamed visible text —
    /// models write reliable answers as text and unreliable ones as long JSON
    /// string arguments, so text is the canonical carrier of the answer.
    case finish(result: String, status: FinishStatus)
}

/// A tool that appears in the spec the model sees but is intercepted and handled
/// in-process — its handler runs instead of any real `execute()`, and it decides
/// per call whether the loop continues or finishes.
public struct PseudoTool: Sendable {
    public let name: String
    public let description: String
    public let parameters: [LocalLLMToolParameter]
    public let handler: @Sendable ([String: String]) async -> PseudoToolOutcome

    public init(
        name: String,
        description: String,
        parameters: [LocalLLMToolParameter],
        handler: @escaping @Sendable ([String: String]) async -> PseudoToolOutcome
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
    }

    /// Adapter so pseudo-tools ride the existing spec-building and prefix-cache
    /// fingerprint paths. `execute` is unreachable — calls are intercepted first.
    /// Public so hosts can include the pseudo-tool spec when prewarming a prefix
    /// cache for agent-mode turns (the fingerprint must match `runToolLoop`'s).
    public var specAdapter: any LocalLLMTool { SpecAdapter(pseudo: self) }

    private struct SpecAdapter: LocalLLMTool {
        let pseudo: PseudoTool
        var name: String { pseudo.name }
        var description: String { pseudo.description }
        var parameters: [LocalLLMToolParameter] { pseudo.parameters }
        func execute(arguments: [String: String]) async throws -> String {
            assertionFailure("Pseudo-tool \(name) must be intercepted, never executed")
            return ""
        }
    }
}

/// Configuration for the forced sign-off micro-pass. When an agent-mode pass
/// produces no tool calls, instead of nudging with prompt text the loop
/// *prefills* the assistant response with a tool call for `toolName`, up to the
/// opening quote of `parameterName`'s value — the model generates only the
/// choice. Structure by construction: format drift cannot occur because the
/// model never chooses the format.
public struct ForcedSignoff: Sendable {
    /// Name of the pseudo-tool to force. Must be present in `pseudoTools`.
    public let toolName: String
    /// The forced-choice parameter (e.g. "status").
    public let parameterName: String
    /// Allowed completions. The first entry is the fallback when the model's
    /// completion matches none of them.
    public let choices: [String]

    public init(toolName: String, parameterName: String, choices: [String]) {
        precondition(!choices.isEmpty, "ForcedSignoff requires at least one choice")
        self.toolName = toolName
        self.parameterName = parameterName
        self.choices = choices
    }
}

/// Opt-in agent semantics for `ChatSession.streamResponse`. When nil, the session
/// behaves exactly as before: completion is inferred from the model producing no
/// tool calls, `maxToolPasses` caps the loop, and exhaustion is a prose notice.
///
/// In agent mode:
/// - pseudo-tools are injected into the tool spec and intercepted in-process;
/// - a pass with no tool calls triggers the forced sign-off micro-pass when
///   configured (preferred), else one text nudge then honest exhaustion;
/// - exhaustion yields `.budgetExhausted(passes:reason:)` instead of transcript text.
public struct AgentLoopOptions: Sendable {
    /// Iteration ceiling for this turn. Replaces `maxToolPasses` in agent mode.
    public var maxPasses: Int
    /// Pseudo-tools injected into the tool spec and intercepted in-process.
    public var pseudoTools: [PseudoTool]
    /// When set, silence triggers a prefilled sign-off instead of a text nudge.
    public var forcedSignoff: ForcedSignoff?

    public init(
        maxPasses: Int = 15,
        pseudoTools: [PseudoTool] = [],
        forcedSignoff: ForcedSignoff? = nil
    ) {
        self.maxPasses = maxPasses
        self.pseudoTools = pseudoTools
        self.forcedSignoff = forcedSignoff
    }
}

/// How long an approval lasts.
public enum ConsentScope: String, Sendable {
    /// Covers only the calls in the approved batch.
    case thisAction = "action"
    /// Covers the rest of the current turn's loop (the default in hosts).
    case thisTask = "task"
    /// Covers the rest of this ChatSession.
    case thisConversation = "conversation"
}

/// The user's answer to a batched consent request.
public struct ConsentDecision: Sendable {
    /// Calls the user approved; the rest return "declined" tool errors.
    public let approvedCallIDs: Set<UUID>
    public let scope: ConsentScope
    /// Per-call argument replacements — e.g. a user-edited script. Applied
    /// before execution and reflected in the yielded `.toolCall` event.
    public let argumentOverrides: [UUID: [String: String]]

    public init(
        approvedCallIDs: Set<UUID>,
        scope: ConsentScope,
        argumentOverrides: [UUID: [String: String]] = [:]
    ) {
        self.approvedCallIDs = approvedCallIDs
        self.scope = scope
        self.argumentOverrides = argumentOverrides
    }
}

/// Loop-level consent: called once per pass with that pass's action-kind calls
/// (unless a task- or conversation-scope grant already covers them). Suspends
/// until the user decides. Replaces per-call confirmation inside tools — tools
/// are pure executors; policy lives at the loop boundary.
public protocol AgentConsentDelegate: Sendable {
    func requestConsent(for calls: [LocalLLMToolCall]) async -> ConsentDecision
}

extension ToolPromptDefaults {
    /// Injected once, as a user-role message, when an agent-mode pass produces
    /// neither tool calls nor a finishing pseudo-tool call.
    public static let agentFinishNudge = """
        You have not called `finish`. Call the `finish` tool now — as a proper tool \
        call, exactly like the other tools, NOT by writing "finish" as text. \
        Use status "success" if the task is complete, or "gave_up" if you cannot \
        complete it.
        """
}
#endif
