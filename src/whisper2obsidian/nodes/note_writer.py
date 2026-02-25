"""
note_writer_node – Renders the final Obsidian Markdown note from the
Jinja2 template selected by the memo's category.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from whisper2obsidian.config import settings
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)

# Templates directory sits next to this package
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def note_writer_node(state: W2OState) -> W2OState:
    """
    Render a Markdown note from the analysis + metadata.
    Writes to state['note_markdown'] and state['note_filename'].
    """
    analysis: dict[str, Any] = state.get("analysis", {})
    metadata: dict[str, Any] = state.get("metadata", {})
    transcript: str = state.get("transcript", "")

    template_key = (
        analysis.get("category_override")
        or metadata.get("template_key", "default")
    )

    # Load the Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["slugify"] = _slugify
    env.filters["wikilink"] = _wikilink

    template_file = f"{template_key}.md.j2"
    if not (_TEMPLATES_DIR / template_file).exists():
        logger.warning("Template '%s' not found, falling back to default", template_file)
        template_file = "default.md.j2"

    template = env.get_template(template_file)

    context = {
        "title": analysis.get("title", metadata.get("title", "Untitled")),
        "summary": analysis.get("summary", ""),
        "key_points": analysis.get("key_points", []),
        "action_items": analysis.get("action_items", []),
        "tags": analysis.get("tags", []),
        "suggested_links": analysis.get("suggested_links", []),
        "mermaid_diagram": analysis.get("mermaid_diagram"),
        "dataview_fields": analysis.get("dataview_fields", {}),
        "metadata": metadata,
        "transcript": transcript,
        "language": state.get("language", ""),
        "groq_tokens_used": state.get("groq_tokens_used", 0),
    }

    note_markdown = template.render(**context)
    note_filename = _slugify(context["title"])

    logger.info("Note rendered: %s (%d chars)", note_filename, len(note_markdown))

    return {
        **state,
        "note_markdown": note_markdown,
        "note_filename": note_filename,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _slugify(value: str) -> str:
    """Convert a string to a filesystem-friendly slug (Obsidian-safe)."""
    value = re.sub(r"[^\w\s-]", "", str(value))
    return re.sub(r"[-\s]+", "-", value).strip("-").lower()


def _wikilink(stem: str, display: str | None = None) -> str:
    if display:
        return f"[[{stem}|{display}]]"
    return f"[[{stem}]]"
