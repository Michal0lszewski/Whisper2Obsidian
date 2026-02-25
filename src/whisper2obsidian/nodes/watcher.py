"""
watcher_node – Scans the audio folder for new .m4a files.

Returns the oldest unprocessed file so the graph processes memos
in chronological order. If no new file is found, the graph ends.
"""

from __future__ import annotations

import logging
from pathlib import Path

from whisper2obsidian.config import settings
from whisper2obsidian.services.metadata_parser import parse_metadata
from whisper2obsidian.services.vault_index import VaultIndex
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)


def watcher_node(state: W2OState) -> W2OState:
    """
    Scan AUDIO_FOLDER for unprocessed .m4a files (sorted oldest first by mtime).
    Populate `audio_path` and `metadata` in state.
    Set `audio_path` to "" when nothing new is found (triggers END branch).
    """
    index = VaultIndex(settings.processed_db)
    processed = set(index.processed_stems())

    audio_folder = settings.audio_folder
    if not audio_folder.exists():
        logger.error("Audio folder does not exist: %s", audio_folder)
        return {**state, "audio_path": "", "errors": ["Audio folder not found"]}

    # Collect all .m4a files not yet processed, sorted by mtime ascending
    candidates: list[Path] = sorted(
        (
            f
            for f in audio_folder.iterdir()
            if f.suffix.lower() == ".m4a" and f.stem not in processed
        ),
        key=lambda f: f.stat().st_mtime,
    )

    if not candidates:
        logger.info("No new voice memos found in %s", audio_folder)
        return {**state, "audio_path": "", "already_processed": list(processed)}

    chosen = candidates[0]
    logger.info("New memo found: %s (mtime %s)", chosen.name, chosen.stat().st_mtime)

    metadata = parse_metadata(chosen)

    return {
        **state,
        "audio_path": str(chosen),
        "metadata": metadata,
        "already_processed": list(processed),
    }


def has_new_memo(state: W2OState) -> str:
    """Conditional edge: route to transcription or END."""
    return "transcription" if state.get("audio_path") else "end"
