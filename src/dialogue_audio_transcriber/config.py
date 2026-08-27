import os
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _dotenv_value(*keys: str) -> str:
    candidates = (Path(".env"), Path(__file__).resolve().parent / ".env")
    for path in candidates:
        value = _read_keys_from_env_file(path, keys)
        if value:
            return value
    return ""


def _read_keys_from_env_file(path: Path, keys: tuple[str, ...]) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    wanted = set(keys)
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in wanted:
            return value.strip().strip("'").strip('"')
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    whisper_model: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "pt"  # Whisper takes ISO 639-1 codes (pt), not locale tags (pt-br).
    beam_size: int = 8
    diarization_device: str = "cpu"
    output_dir: Path = Path("recordings")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    hf_token: str = ""
    pyannote_api_key: str = ""

    @model_validator(mode="after")
    def fallback_secrets(self) -> Self:
        if not self.hf_token.strip():
            token = os.environ.get("HF_TOKEN", "").strip() or _dotenv_value(
                "TRANSCRIBER_HF_TOKEN", "HF_TOKEN"
            )
            if token:
                self.hf_token = token
        if not self.pyannote_api_key.strip():
            token = os.environ.get("PYANNOTEAI_API_KEY", "").strip() or _dotenv_value(
                "TRANSCRIBER_PYANNOTE_API_KEY", "PYANNOTEAI_API_KEY"
            )
            if token:
                self.pyannote_api_key = token
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
