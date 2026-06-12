import os
from openai import OpenAI
from llm.base import BaseLLMProvider, LLMResponse

OPENAI_MODEL = "gpt-4o-mini"


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        context = self._build_context(chunks)
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=512,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        answer = response.choices[0].message.content.strip()
        return LLMResponse(
            answer=answer,
            provider="openai",
            model=OPENAI_MODEL,
            confident=not self._is_low_confidence(answer),
        )
