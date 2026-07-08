#if os(macOS)
import Foundation

/// How an agent-mode turn ended when a pseudo-tool completed the loop.
public enum FinishStatus: String, Sendable {
    case success
    case gaveUp = "gave_up"
}

/// What a pseudo-tool handler wants the loop to do next.
public enum PseudoToolOutcome: Sendable {
    /// Feed `toolResult` back to the model as an ordinary tool result and continue.
    case continueLoop(toolResult: String)
    /// End the loop now. Surfaces as `.agentFinished(result:status:)`.
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

/// Opt-in agent semantics for `ChatSession.streamResponse`. When nil, the session
/// behaves exactly as before: completion is inferred from the model producing no
/// tool calls, `maxToolPasses` caps the loop, and exhaustion is a prose notice.
///
/// In agent mode:
/// - pseudo-tools are injected into the tool spec and intercepted in-process;
/// - a pass with no tool calls earns one nudge to call `finish`; a second silent
///   pass ends the turn as `.budgetExhausted`;
/// - exhaustion yields `.budgetExhausted(passes:)` instead of transcript text.
public struct AgentLoopOptions: Sendable {
    /// Iteration ceiling for this turn. Replaces `maxToolPasses` in agent mode.
    public var maxPasses: Int
    /// Pseudo-tools injected into the tool spec and intercepted in-process.
    public var pseudoTools: [PseudoTool]

    public init(maxPasses: Int = 15, pseudoTools: [PseudoTool] = []) {
        self.maxPasses = maxPasses
        self.pseudoTools = pseudoTools
    }
}

extension ToolPromptDefaults {
    /// Injected once, as a user-role message, when an agent-mode pass produces
    /// neither tool calls nor a finishing pseudo-tool call.
    public static let agentFinishNudge = """
        You have not called `finish`. If the task is complete, call `finish` with the result now. \
        If you cannot complete it, call `finish` with status "gave_up" and explain what you tried.
        """
}
#endif
