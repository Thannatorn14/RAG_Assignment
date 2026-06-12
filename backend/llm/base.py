from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    answer: str
    provider: str
    model: str
    confident: bool  # False = suggest escalation


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[dict]) -> LLMResponse:
        pass

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source {i} | Page {c['page']}]\n{c['text']}")
        return "\n\n".join(parts)

    def _system_prompt(self) -> str:
        return (
            "You are a helpful document assistant. Answer the user's question "
            "using ONLY the provided source excerpts. Be concise and accurate. "
            "If the answer cannot be found in the sources, say exactly: "
            "'I could not find a clear answer in the document.'"
        )

    def _is_low_confidence(self, answer: str) -> bool:
        low_conf_phrases = [
            "i could not find",
            "not mentioned",
            "not found in",
            "no information",
            "cannot determine",
            "does not appear",
        ]
        lowered = answer.lower()
        return any(p in lowered for p in low_conf_phrases) or len(answer.split()) < 15
