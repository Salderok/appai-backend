"""Application settings loaded from environment via pydantic-settings.

All configuration must flow through `settings`; never call `os.getenv` elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root .env is also loaded as a fallback (handy in monorepo dev).
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app ---
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # --- security ---
    device_key: str = Field(default="dev-device-key-change-me", min_length=8)
    jwt_secret: str = Field(default="dev-jwt-secret-change-me", min_length=8)
    jwt_alg: str = "HS256"

    # --- database ---
    database_url: str = "sqlite+aiosqlite:///./appai.db"

    # --- llm providers ---
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    ollama_base_url: str | None = None
    ollama_default_model: str = "phi3:mini"

    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    # If the primary provider fails (network, quota), automatically retry the
    # request through Ollama at OLLAMA_BASE_URL using OLLAMA_DEFAULT_MODEL.
    enable_offline_fallback: bool = False

    # --- embeddings ---
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"

    # --- voice ---
    whisper_provider: Literal["openai", "local"] = "openai"
    tts_provider: Literal["openai", "edge", "none"] = "openai"
    tts_voice: str = "alloy"

    # --- files ---
    upload_dir: Path = BACKEND_DIR / "uploads"
    max_upload_mb: int = 25

    # --- cors ---
    cors_origins: list[str] = ["http://localhost:8000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
