import os
from time import perf_counter

from openai import AsyncOpenAI

from app.llm_clients.base import BaseLLMAdapter, LLMResponse


class OpenAIAdapter(BaseLLMAdapter):

    async def generate(
        self,
        model_config,
        prompt: str,
        settings: dict,
    ) -> LLMResponse:

        api_key_env = model_config.api_key_env or "OPENAI_API_KEY"

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"OpenAI API key not found in environment variable " f"'{api_key_env}'."
            )

        client_kwargs = {
            "api_key": api_key,
        }

        if model_config.base_url and "api.openai.com" not in model_config.base_url:
            client_kwargs["base_url"] = model_config.base_url.rstrip("/")

        client = AsyncOpenAI(**client_kwargs)

        temperature = settings.get("temperature", 0.2)
        max_tokens = settings.get("max_tokens", 1024)

        started = perf_counter()

        try:
            response = await client.responses.create(
                model=model_config.name,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        finally:
            await client.close()

        latency_ms = (perf_counter() - started) * 1000

        content = response.output_text or ""

        input_tokens = None
        output_tokens = None

        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        #
        # OpenAI doesn't expose Ollama-style eval_duration.
        #
        # For the cloud model, throughput is therefore measured
        # using the complete observed request latency.
        #
        tokens_per_second = None

        if output_tokens is not None and latency_ms > 0:
            tokens_per_second = output_tokens / (latency_ms / 1000)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            raw_response=response.model_dump(),
        )
