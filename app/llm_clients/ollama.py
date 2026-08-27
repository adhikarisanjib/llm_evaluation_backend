import time

import httpx

from app.llm_clients.base import BaseLLMAdapter, LLMResponse


class OllamaAdapter(BaseLLMAdapter):
    async def generate(
        self, *, model_config, prompt: str, settings: dict
    ) -> LLMResponse:
        payload = {
            "model": model_config.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.get("temperature", 0.2),
                "top_p": settings.get("top_p", 0.95),
                "num_predict": settings.get("max_tokens", 1024),
            },
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{model_config.base_url.rstrip('/')}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - started) * 1000
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        tokens_per_second = None
        if eval_count is not None and eval_duration:
            tokens_per_second = eval_count / (eval_duration / 1_000_000_000)

        return LLMResponse(
            content=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=eval_count,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            raw_response=data,
        )
