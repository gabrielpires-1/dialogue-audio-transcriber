from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper_model: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "pt"  # Whisper takes ISO 639-1 codes (pt), not locale tags (pt-br).
    output_dir: Path = Path("recordings")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
