from time import perf_counter

from ollama import AsyncClient

from app.llm_clients.base import BaseLLMAdapter, LLMResponse


class OllamaAdapter(BaseLLMAdapter):

    async def generate(
        self,
        model_config,
        prompt: str,
        settings: dict,
    ) -> LLMResponse:

        host = (model_config.base_url or "http://localhost:11434").rstrip("/")

        client = AsyncClient(host=host)

        options = {}

        if settings.get("temperature") is not None:
            options["temperature"] = settings["temperature"]

        if settings.get("top_p") is not None:
            options["top_p"] = settings["top_p"]

        if settings.get("max_tokens") is not None:
            options["num_predict"] = settings["max_tokens"]

        started = perf_counter()

        response = await client.generate(
            model=model_config.name,
            prompt=prompt,
            stream=False,
            options=options,
        )

        latency_ms = (perf_counter() - started) * 1000

        content = response.response or ""

        input_tokens = response.prompt_eval_count
        output_tokens = response.eval_count

        tokens_per_second = None

        if (
            response.eval_count
            and response.eval_duration
            and response.eval_duration > 0
        ):
            tokens_per_second = response.eval_count / (
                response.eval_duration / 1_000_000_000
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            raw_response=response.model_dump(),
        )
