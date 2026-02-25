"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide minimum env vars so Settings can be instantiated in tests."""
    monkeypatch.setenv("AUDIO_FOLDER", "/tmp/voice_memos")
    monkeypatch.setenv("VAULT_PATH", "/tmp/obsidian_vault")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
