import os
import httpx
from llm.base import BaseLLMProvider, LLMResponse

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class HFProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")

    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        print(f"[HFProvider] URL: {GROQ_API_URL}")
        print(f"[HFProvider] GROQ_API_KEY first 20 chars: {self.api_key[:20]!r}")

        context = self._build_context(chunks)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            response = httpx.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.3},
                timeout=60,
            )
            print(f"[HFProvider] Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"[HFProvider] Response text: {response.text}")
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return LLMResponse(
                answer=answer,
                provider="llama",
                model="llama-3.1-8b-instant (Groq)",
                confident=not self._is_low_confidence(answer),
            )
        except Exception as exc:
            print(f"[HFProvider] Exception: {exc!r}")
            raise
