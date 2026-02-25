"""
watcher_node – Scans the audio folder for new .m4a files.

Returns the oldest unprocessed file so the graph processes memos
in chronological order. If no new file is found, the graph ends.

"Already processed" is decided by two independent checks (either is enough):
  1. SQLite DB – stem registered by file_writer after a successful run.
  2. Inbox filesystem – a .md note whose filename contains the audio stem
     already exists in the vault inbox (covers manual moves / DB resets).

If a <stem>.txt transcript sidecar exists next to the audio, the flag
`transcript_cached: True` is added to state so transcription_node skips
Whisper and loads the transcript from disk directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from whisper2obsidian.config import settings
from whisper2obsidian.nodes.transcription import transcript_txt_path
from whisper2obsidian.services.metadata_parser import parse_metadata
from whisper2obsidian.services.vault_index import VaultIndex
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)


def _note_exists_in_inbox(stem: str) -> bool:
    """Return True if any .md in the vault inbox contains `stem` in its filename."""
    inbox = settings.inbox_path
    if not inbox.exists():
        return False
    # The file_writer names notes like  YYYY-MM-DD-<slug>.md  so we check
    # whether the audio stem appears anywhere inside an existing .md filename.
    return any(stem in md.stem for md in inbox.glob("*.md"))


def watcher_node(state: W2OState) -> W2OState:
    """
    Scan AUDIO_FOLDER for unprocessed .m4a files (sorted oldest first by mtime).
    Populate `audio_path` and `metadata` in state.
    Set `audio_path` to "" when nothing new is found (triggers END branch).
    """
    index = VaultIndex(settings.processed_db)
    processed_db_stems = set(index.processed_stems())

    audio_folder = settings.audio_folder
    if not audio_folder.exists():
        logger.error("Audio folder does not exist: %s", audio_folder)
        return {**state, "audio_path": "", "errors": ["Audio folder not found"]}

    # Collect all .m4a files not yet processed, sorted by mtime ascending
    candidates: list[Path] = sorted(
        (
            f
            for f in audio_folder.iterdir()
            if f.suffix.lower() == ".m4a"
            and f.stem not in processed_db_stems           # fast DB check
            and not _note_exists_in_inbox(f.stem)          # filesystem safety-net
        ),
        key=lambda f: f.stat().st_mtime,
    )

    if not candidates:
        logger.info("No new voice memos found in %s", audio_folder)
        return {**state, "audio_path": "", "already_processed": list(processed_db_stems)}

    chosen = candidates[0]
    logger.info("New memo found: %s (mtime %s)", chosen.name, chosen.stat().st_mtime)

    metadata = parse_metadata(chosen)

    # Check whether a transcript sidecar already exists so we can skip Whisper
    txt_exists = transcript_txt_path(chosen).exists()
    if txt_exists:
        logger.info(
            "Transcript sidecar found (%s.txt) – Whisper will be skipped",
            chosen.stem,
        )

    return {
        **state,
        "audio_path": str(chosen),
        "metadata": metadata,
        "already_processed": list(processed_db_stems),
        "transcript_cached": txt_exists,     # signals transcription_node
    }


def has_new_memo(state: W2OState) -> str:
    """Conditional edge: route to transcription or END."""
    return "transcription" if state.get("audio_path") else "end"

