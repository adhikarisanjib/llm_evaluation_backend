from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    tokens_per_second: float | None = None
    raw_response: dict[str, Any] | None = None


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate(
        self, *, model_config, prompt: str, settings: dict
    ) -> LLMResponse:
        raise NotImplementedError
