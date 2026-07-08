#if os(macOS)
import Foundation

/// Parses a token-by-token text stream from a tool-calling model, separating visible text
/// from `<tool_call>{...}</tool_call>` blocks. Unlike MLXLMCommon's ToolCallProcessor, the
/// JSON parser handles multi-line content — Qwen 2.5 often emits pretty-printed JSON inside
/// the tool_call tags.
final class ToolStreamParser {
    enum Output {
        case text(String)
        case toolCall(name: String, arguments: [String: String])
        /// A <tool_call> block whose JSON couldn't be parsed even after repair,
        /// or one left unterminated at end of generation. Never dropped silently —
        /// the loop feeds an error back to the model so it can retry.
        case malformedToolCall(raw: String)
    }

    private static let openTag = "<tool_call>"
    private static let closeTag = "</tool_call>"

    private var buffer: String = ""

    func ingest(_ chunk: String) -> [Output] {
        buffer += chunk
        return drain()
    }

    func finish() -> [Output] {
        var out: [Output] = []
        if !buffer.isEmpty {
            // An unterminated <tool_call> at end of generation (e.g. the output
            // token cap hit mid-arguments) is a malformed call, not visible text.
            if buffer.contains(Self.openTag) {
                out.append(.malformedToolCall(raw: buffer))
            } else {
                out.append(.text(buffer))
            }
            buffer = ""
        }
        return out
    }

    private func drain() -> [Output] {
        var out: [Output] = []
        while !buffer.isEmpty {
            if let openRange = buffer.range(of: Self.openTag) {
                let prefix = String(buffer[..<openRange.lowerBound])
                if !prefix.isEmpty {
                    out.append(.text(prefix))
                }
                let afterOpen = openRange.upperBound
                guard let closeRange = buffer.range(of: Self.closeTag, range: afterOpen..<buffer.endIndex) else {
                    buffer = String(buffer[openRange.lowerBound...])
                    return out
                }
                let json = String(buffer[afterOpen..<closeRange.lowerBound])
                if let parsed = Self.parseToolCall(json) {
                    out.append(.toolCall(name: parsed.name, arguments: parsed.arguments))
                } else {
                    out.append(.malformedToolCall(raw: json))
                }
                buffer = String(buffer[closeRange.upperBound...])
                continue
            }
            if let pendingPrefix = pendingOpenTagPrefix(buffer) {
                let emit = String(buffer.dropLast(pendingPrefix.count))
                if !emit.isEmpty { out.append(.text(emit)) }
                buffer = pendingPrefix
                return out
            }
            out.append(.text(buffer))
            buffer = ""
            return out
        }
        return out
    }

    /// If the buffer ends in a strict prefix of `<tool_call>`, returns that prefix so we
    /// can hold it back until the next chunk arrives — avoids splitting the start tag.
    private func pendingOpenTagPrefix(_ s: String) -> String? {
        let tag = Self.openTag
        let maxLen = min(tag.count - 1, s.count)
        if maxLen <= 0 { return nil }
        for len in stride(from: maxLen, through: 1, by: -1) {
            let suffix = String(s.suffix(len))
            if tag.hasPrefix(suffix) {
                return suffix
            }
        }
        return nil
    }

    /// Parses a bare arguments object — used by ChatSession's bare-text pseudo-call
    /// rescue, where the model writes `finish {"status": "success"}` as plain text
    /// instead of a <tool_call> block. Same leniency as tool-call parsing.
    static func parseArguments(_ json: String) -> [String: String]? {
        guard let obj = parseObject(json.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return nil
        }
        var args: [String: String] = [:]
        for (k, v) in obj { args[k] = stringify(v) }
        return args
    }

    private static func parseToolCall(_ json: String) -> (name: String, arguments: [String: String])? {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let obj = parseObject(trimmed) else { return nil }
        guard let name = obj["name"] as? String else { return nil }
        var args: [String: String] = [:]
        if let argsDict = obj["arguments"] as? [String: Any] {
            for (k, v) in argsDict {
                args[k] = stringify(v)
            }
        }
        return (name, args)
    }

    /// Tolerates the doubled-brace pattern (`{{...}}`) emitted by some quantized Qwen
    /// chat templates whose Jinja escaping leaks into the output. Tries the literal
    /// string first, then peels matched outer brace pairs up to a few levels.
    private static func parseObject(_ candidate: String) -> [String: Any]? {
        var attempt = candidate
        for _ in 0..<3 {
            if let obj = jsonObject(attempt) { return obj }
            // Repair pass: models emit multiline string values (long `result`
            // arguments especially) without \n escaping. Escape raw control
            // characters inside string literals only, then retry.
            if let obj = jsonObject(escapingControlCharsInStrings(attempt)) { return obj }
            guard attempt.hasPrefix("{{"), attempt.hasSuffix("}}") else { return nil }
            attempt = String(attempt.dropFirst().dropLast())
        }
        return nil
    }

    private static func jsonObject(_ s: String) -> [String: Any]? {
        guard let data = s.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Escapes raw newlines/tabs that appear INSIDE JSON string literals, walking
    /// the text with a minimal in-string/escape state machine. Control characters
    /// between tokens (pretty-printed JSON) are untouched — that form parses fine
    /// on the first attempt anyway.
    private static func escapingControlCharsInStrings(_ s: String) -> String {
        var out = String()
        out.reserveCapacity(s.count)
        var inString = false
        var escaped = false
        for ch in s {
            if inString {
                if escaped {
                    out.append(ch)
                    escaped = false
                    continue
                }
                switch ch {
                case "\\":
                    out.append(ch)
                    escaped = true
                case "\"":
                    out.append(ch)
                    inString = false
                case "\n":
                    out.append("\\n")
                case "\r":
                    out.append("\\r")
                case "\t":
                    out.append("\\t")
                default:
                    out.append(ch)
                }
            } else {
                out.append(ch)
                if ch == "\"" { inString = true }
            }
        }
        return out
    }

    private static func stringify(_ value: Any) -> String {
        switch value {
        case let s as String: return s
        case let i as Int: return String(i)
        case let d as Double: return String(d)
        case let b as Bool: return String(b)
        case is NSNull: return ""
        default:
            if let data = try? JSONSerialization.data(withJSONObject: value),
               let s = String(data: data, encoding: .utf8) {
                return s
            }
            return String(describing: value)
        }
    }
}

#endif
