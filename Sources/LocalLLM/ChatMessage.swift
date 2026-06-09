#if os(macOS)
import Foundation

public struct ChatMessage: Sendable, Hashable, Codable, Identifiable {
    public enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    public let id: UUID
    public var role: Role
    public var content: String

    public init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

/// Disambiguating alias for hosts whose own module also declares a `ChatMessage`.
/// The module name `LocalLLM` is shadowed by the `LocalLLM` actor, so
/// `LocalLLM.ChatMessage` can't be used to qualify; this alias gives callers an
/// unambiguous name.
public typealias LocalLLMChatMessage = ChatMessage

#endif
