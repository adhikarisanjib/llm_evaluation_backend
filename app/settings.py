import os


class Settings:
    secret_key: str = os.getenv("SECRET_KEY", "some-random-secret-key")
    debug: bool = bool(int(os.getenv("DEBUG", "1")))

    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

    sandbox_backend: str = os.getenv("SANDBOX_BACKEND", "docker")
    sandbox_image: str = os.getenv("SANDBOX_IMAGE", "llm-eval-python-sandbox")
    sandbox_timeout_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "10"))


settings = Settings()
