import os
import time

import httpx

from app.llm_clients.base import BaseLLMAdapter, LLMResponse
from app.settings import settings as app_settings


class OpenAIAdapter(BaseLLMAdapter):
    async def generate(
        self, *, model_config, prompt: str, settings: dict
    ) -> LLMResponse:
        env_name = model_config.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(env_name)
        if not api_key and env_name == "OPENAI_API_KEY":
            api_key = app_settings.openai_api_key
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {env_name}")

        payload = {
            "model": model_config.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.get("temperature", 0.2),
            "top_p": settings.get("top_p", 0.95),
            "max_tokens": settings.get("max_tokens", 1024),
        }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{model_config.base_url.rstrip('/')}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - started) * 1000
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            raw_response=data,
        )
