"""
Configuration – loads all settings from environment / .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    audio_folder: Path = Field(..., description="Google Drive folder with .m4a memos")
    vault_path: Path = Field(..., description="Obsidian vault root")
    inbox_folder: str = Field("00 Inbox", description="Sub-folder for new notes")
    processed_db: Path = Field(
        Path(__file__).parent.parent.parent / "data" / "w2o.db",
        description="SQLite DB for processed files, tags and links",
    )

    # ── Whisper ────────────────────────────────────────────────────────────
    whisper_model: str = Field(
        "mlx-community/whisper-large-v3-mlx",
        description="HuggingFace repo ID or local path for mlx-whisper",
    )

    # ── Groq ───────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model identifier",
    )

    # ── Groq rate limits (configurable safety caps) ────────────────────────
    groq_rpm_limit: int = Field(28, description="Max requests per minute sent to Groq")
    groq_tpm_limit: int = Field(11_000, description="Max tokens per minute sent to Groq")
    groq_rpd_limit: int = Field(950, description="Max requests per day sent to Groq")

    # ── Logging / CLI ──────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Python logging level")
    show_rate_usage: bool = Field(False, description="Print Groq rate usage after each run")

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def inbox_path(self) -> Path:
        return self.vault_path / self.inbox_folder

    @field_validator("audio_folder", "vault_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


# Singleton – import this everywhere
settings = Settings()  # type: ignore[call-arg]
