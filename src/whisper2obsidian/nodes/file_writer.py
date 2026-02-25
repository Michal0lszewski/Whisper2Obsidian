"""
file_writer_node – Writes the rendered Markdown to the Obsidian vault
and updates the SQLite vault index.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)


def file_writer_node(state: W2OState) -> W2OState:
    """
    Write state['note_markdown'] to the inbox folder in the Obsidian vault.
    Update the vault index (mark audio as processed, add new note's tags/links).
    """
    note_markdown = state.get("note_markdown", "")
    note_filename = state.get("note_filename", "untitled")
    audio_path = state.get("audio_path", "")
    analysis = state.get("analysis", {})
    metadata = state.get("metadata", {})

    if not note_markdown:
        return {**state, "errors": ["file_writer_node: note_markdown is empty"]}

    # Ensure inbox exists
    inbox = settings.inbox_path
    inbox.mkdir(parents=True, exist_ok=True)

    # Avoid collisions – append date if file already exists
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    candidate = inbox / f"{date_prefix}-{note_filename}.md"
    if candidate.exists():
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        candidate = inbox / f"{date_prefix}-{note_filename}-{ts}.md"

    candidate.write_text(note_markdown, encoding="utf-8")
    logger.info("Note written: %s", candidate)

    # Update vault index
    index = VaultIndex(settings.processed_db)
    stem = candidate.stem

    # Register the new note
    index.upsert_note(stem, analysis.get("title", stem), str(candidate))
    index.upsert_tags(stem, analysis.get("tags", []))
    index.upsert_links(stem, analysis.get("suggested_links", []))

    # Mark the audio file as processed (by its stem)
    if audio_path:
        index.mark_processed(Path(audio_path).stem)

    return {
        **state,
        "note_path": str(candidate),
    }
