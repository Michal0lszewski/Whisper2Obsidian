"""
LangGraph state schema – shared across all graph nodes.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class W2OState(TypedDict, total=False):
    """Shared state passed between every node in the Whisper2Obsidian graph."""

    # ── Watcher ────────────────────────────────────────────────────────────
    audio_path: str                  # Absolute path to the .m4a file
    metadata: dict[str, Any]         # Parsed sidecar metadata from Voice Record Pro
    already_processed: list[str]     # File stems already in the DB

    # ── Transcription ──────────────────────────────────────────────────────
    transcript: str                  # Full transcribed text
    language: str                    # Detected language code (e.g. "en")
    transcript_token_count: int      # Estimated token count of transcript

    # ── Vault index ────────────────────────────────────────────────────────
    existing_tags: list[str]         # All tags found across the vault
    existing_links: dict[str, str]   # {file_stem: display_title} from vault

    # ── Analysis ───────────────────────────────────────────────────────────
    analysis: dict[str, Any]         # Structured JSON from Groq
    groq_tokens_used: int            # Actual tokens consumed for this run

    # ── Note generation ────────────────────────────────────────────────────
    note_markdown: str               # Final rendered Markdown content
    note_filename: str               # Target filename (without .md extension)
    note_path: str                   # Absolute path where the note was written

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: Annotated[list[str], add_messages]  # Non-fatal errors accumulated

    # ── Internal messages (LangGraph standard) ─────────────────────────────
    messages: Annotated[list, add_messages]
