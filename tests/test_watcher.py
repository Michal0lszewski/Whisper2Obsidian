"""Tests for watcher_node."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper2obsidian.nodes.watcher import has_new_memo, watcher_node


@pytest.fixture()
def audio_dir(tmp_path: Path) -> Path:
    d = tmp_path / "voice_memos"
    d.mkdir()
    return d


def _make_audio(dir: Path, name: str, mtime_offset: float = 0) -> Path:
    f = dir / name
    f.write_bytes(b"\x00" * 100)
    t = time.time() + mtime_offset
    import os
    os.utime(f, (t, t))
    return f


def test_watcher_picks_oldest_unprocessed(audio_dir: Path, tmp_path: Path) -> None:
    _make_audio(audio_dir, "old.m4a", mtime_offset=-200)
    _make_audio(audio_dir, "new.m4a", mtime_offset=0)

    db_path = tmp_path / "data" / "w2o.db"

    with (
        patch("whisper2obsidian.nodes.watcher.settings") as mock_settings,
        patch("whisper2obsidian.nodes.watcher.VaultIndex") as MockIndex,
    ):
        mock_settings.audio_folder = audio_dir
        mock_settings.processed_db = db_path
        instance = MockIndex.return_value
        instance.processed_stems.return_value = []

        state = watcher_node({})

    assert "old.m4a" in state["audio_path"]


def test_watcher_skips_processed(audio_dir: Path, tmp_path: Path) -> None:
    _make_audio(audio_dir, "processed.m4a", mtime_offset=-100)
    _make_audio(audio_dir, "fresh.m4a", mtime_offset=0)

    db_path = tmp_path / "data" / "w2o.db"

    with (
        patch("whisper2obsidian.nodes.watcher.settings") as mock_settings,
        patch("whisper2obsidian.nodes.watcher.VaultIndex") as MockIndex,
    ):
        mock_settings.audio_folder = audio_dir
        mock_settings.processed_db = db_path
        instance = MockIndex.return_value
        instance.processed_stems.return_value = ["processed"]

        state = watcher_node({})

    assert "fresh.m4a" in state["audio_path"]


def test_watcher_returns_empty_when_all_processed(audio_dir: Path, tmp_path: Path) -> None:
    _make_audio(audio_dir, "done.m4a")
    db_path = tmp_path / "data" / "w2o.db"

    with (
        patch("whisper2obsidian.nodes.watcher.settings") as mock_settings,
        patch("whisper2obsidian.nodes.watcher.VaultIndex") as MockIndex,
    ):
        mock_settings.audio_folder = audio_dir
        mock_settings.processed_db = db_path
        instance = MockIndex.return_value
        instance.processed_stems.return_value = ["done"]

        state = watcher_node({})

    assert state["audio_path"] == ""


def test_has_new_memo_routing() -> None:
    assert has_new_memo({"audio_path": "/some/path.m4a"}) == "transcription"
    assert has_new_memo({"audio_path": ""}) == "end"
    assert has_new_memo({}) == "end"
