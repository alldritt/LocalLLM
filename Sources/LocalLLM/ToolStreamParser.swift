import Foundation

/// Parses a token-by-token text stream from a tool-calling model, separating visible text
/// from `<tool_call>{...}</tool_call>` blocks. Unlike MLXLMCommon's ToolCallProcessor, the
/// JSON parser handles multi-line content — Qwen 2.5 often emits pretty-printed JSON inside
/// the tool_call tags.
final class ToolStreamParser {
    enum Output {
        case text(String)
        case toolCall(name: String, arguments: [String: String])
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
            out.append(.text(buffer))
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
            if let data = attempt.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                return obj
            }
            guard attempt.hasPrefix("{{"), attempt.hasSuffix("}}") else { return nil }
            attempt = String(attempt.dropFirst().dropLast())
        }
        return nil
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
