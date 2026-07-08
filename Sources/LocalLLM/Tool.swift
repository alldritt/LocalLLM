#if os(macOS)
import Foundation

public protocol LocalLLMTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: [LocalLLMToolParameter] { get }
    func execute(arguments: [String: String]) async throws -> String
}

public struct LocalLLMToolParameter: Sendable, Hashable {
    public enum Kind: String, Sendable, Codable {
        case string
        case integer
        case number
        case boolean
    }

    public let name: String
    public let kind: Kind
    public let description: String
    public let isRequired: Bool

    public init(name: String, kind: Kind, description: String, isRequired: Bool = true) {
        self.name = name
        self.kind = kind
        self.description = description
        self.isRequired = isRequired
    }
}

public struct LocalLLMToolCall: Sendable, Hashable, Identifiable {
    public let id: UUID
    public let name: String
    public let arguments: [String: String]

    public init(id: UUID = UUID(), name: String, arguments: [String: String]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public enum LocalLLMEvent: Sendable {
    case text(String)
    case toolCall(LocalLLMToolCall)
    case toolResult(call: LocalLLMToolCall, content: String, isError: Bool)
    case stats(LocalLLMGenerationStats)
    case finished
    /// Agent mode only: a pseudo-tool handler explicitly ended the turn.
    case agentFinished(result: String, status: FinishStatus)
    /// Agent mode only: the pass budget ran out (or the model went silent twice)
    /// before any pseudo-tool ended the turn. Not a success — hosts should render
    /// it as an honest failure carrying whatever the transcript learned.
    case budgetExhausted(passes: Int)
}

public enum ToolPromptDefaults {
    /// Prepended to the system message whenever tools are passed to streamResponse.
    /// Anchors the model so it treats tool output as authoritative ground truth rather
    /// than hedging ("simulated", "this is fictional", etc.).
    public static let toolTrustPreamble = """
        Tool results are real data from this Mac. Use them directly in your answer.
        """
}

enum ToolSpecBuilder {
    static func toolSpec(for tool: any LocalLLMTool) -> [String: Any] {
        var properties: [String: Any] = [:]
        var required: [String] = []
        for p in tool.parameters {
            properties[p.name] = [
                "type": p.kind.rawValue,
                "description": p.description
            ]
            if p.isRequired { required.append(p.name) }
        }
        return [
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required
                ]
            ]
        ]
    }
}

#endif
