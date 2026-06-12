import os
import anthropic
from llm.base import BaseLLMProvider, LLMResponse

CLAUDE_MODEL = "claude-sonnet-4-20250514"


class ClaudeProvider(BaseLLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        context = self._build_context(chunks)
        message = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=self._system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        answer = message.content[0].text.strip()
        return LLMResponse(
            answer=answer,
            provider="claude",
            model=CLAUDE_MODEL,
            confident=not self._is_low_confidence(answer),
        )
