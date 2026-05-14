import Testing
@testable import LocalLLM

@Test func configurationDefaults() {
    let c = LocalLLM.Configuration()
    #expect(c.modelID == "mlx-community/Qwen2.5-32B-Instruct-4bit")
    #expect(c.maxTokens == 2048)
}

@Test func chatSessionSeedsSystemPrompt() async {
    let llm = LocalLLM()
    let session = ChatSession(llm: llm, systemPrompt: "You are helpful.")
    let transcript = await session.transcript
    #expect(transcript.count == 1)
    #expect(transcript.first?.role == .system)
}
