"""
analysis_node – Uses Groq (Llama 3.3-70b) to analyse the transcript and
produce structured metadata for the Obsidian note.

Rate-limiting: every Groq call is guarded by GroqRateLimiter.await_capacity()
to avoid 429 errors on the free tier.

Long transcripts (> CHUNK_TOKEN_LIMIT) are split into chunks, each chunk
is summarised separately, then a synthesis call combines them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from typing import Any

import tiktoken
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from whisper2obsidian.config import settings
from whisper2obsidian.services.groq_rate_limiter import GroqRateLimiter
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)

# Max tokens we'll send to Groq in a single chunk (leaves room for prompt + reply)
CHUNK_TOKEN_LIMIT = 6_000

_enc = tiktoken.get_encoding("cl100k_base")

# Module-level singleton rate limiter (shared across all calls in one run)
_rate_limiter = GroqRateLimiter()

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert knowledge manager helping convert voice memo transcripts into
structured Obsidian notes. Analyse the transcript and return ONLY valid JSON
(no markdown, no explanation) with this exact schema:

{
  "title": "concise note title",
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2"],
  "action_items": ["action 1"],
  "tags": ["tag1", "tag2"],
  "suggested_links": ["existing-note-stem-1"],
  "category_override": null,
  "mermaid_diagram": null,
  "dataview_fields": {}
}

Rules:
- tags: lowercase, hyphen-separated. Prefer tags from the provided existing_tags list,
  only introduce new tags if genuinely needed.
- suggested_links: choose ONLY from the provided existing_links stems.
- mermaid_diagram: provide a Mermaid flowchart string ONLY for process/workflow memos, else null.
- category_override: override the category if the transcript clearly belongs to a different
  category than the metadata claims, else null.
- dataview_fields: any key::value pairs useful for Dataview queries (e.g. "project", "status").
""").strip()

_CHUNK_SYSTEM_PROMPT = textwrap.dedent("""
You are summarising a chunk of a longer voice memo transcript.
Return ONLY a plain text summary of the key points in this chunk (no JSON).
Be concise but preserve all important facts, names, and action items.
""").strip()

_SYNTHESIS_PROMPT = textwrap.dedent("""
You are combining chunk summaries of a voice memo into a final structured analysis.
Use the same JSON schema as before:
{title, summary, key_points, action_items, tags, suggested_links,
 category_override, mermaid_diagram, dataview_fields}
""").strip()


# ── Main node ────────────────────────────────────────────────────────────────

def analysis_node(state: W2OState) -> W2OState:
    """Synchronous wrapper – runs the async analysis in an event loop."""
    try:
        return asyncio.run(_analysis_async(state))
    except RuntimeError:
        # Already inside an event loop (e.g. Jupyter)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_analysis_async(state))


async def _analysis_async(state: W2OState) -> W2OState:
    transcript = state.get("transcript", "")
    if not transcript:
        return {**state, "errors": ["analysis_node: transcript is empty"]}

    existing_tags: list[str] = state.get("existing_tags", [])
    existing_links: dict[str, str] = state.get("existing_links", {})
    metadata: dict[str, Any] = state.get("metadata", {})
    token_count: int = state.get("transcript_token_count", len(_enc.encode(transcript)))

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )

    total_tokens_used = 0

    if token_count <= CHUNK_TOKEN_LIMIT:
        # ── Single-pass analysis ─────────────────────────────────────────
        analysis, tokens = await _analyse_single(
            llm, transcript, existing_tags, existing_links, metadata
        )
        total_tokens_used = tokens
    else:
        # ── Chunked analysis ─────────────────────────────────────────────
        logger.info(
            "Transcript too long (%d tokens) – splitting into chunks", token_count
        )
        analysis, tokens = await _analyse_chunked(
            llm, transcript, existing_tags, existing_links, metadata
        )
        total_tokens_used = tokens

    logger.info("Analysis complete, Groq tokens used: %d", total_tokens_used)

    if settings.show_rate_usage:
        _log_rate_usage()

    return {
        **state,
        "analysis": analysis,
        "groq_tokens_used": total_tokens_used,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _analyse_single(
    llm: ChatGroq,
    transcript: str,
    existing_tags: list[str],
    existing_links: dict[str, str],
    metadata: dict,
) -> tuple[dict, int]:
    user_content = _build_user_message(transcript, existing_tags, existing_links, metadata)
    estimated = len(_enc.encode(user_content)) + 1200  # prompt + generous reply budget

    await _rate_limiter.await_capacity(estimated)

    resp = await llm.ainvoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    )
    raw = resp.content.strip()
    actual_tokens = resp.usage_metadata.get("total_tokens", estimated) if resp.usage_metadata else estimated
    _rate_limiter.record_usage(actual_tokens)

    return _safe_json(raw), actual_tokens


async def _analyse_chunked(
    llm: ChatGroq,
    transcript: str,
    existing_tags: list[str],
    existing_links: dict[str, str],
    metadata: dict,
) -> tuple[dict, int]:
    chunks = _split_transcript(transcript, CHUNK_TOKEN_LIMIT)
    summaries: list[str] = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        logger.info("Summarising chunk %d/%d", i, len(chunks))
        estimated = len(_enc.encode(chunk)) + 600
        await _rate_limiter.await_capacity(estimated)

        resp = await llm.ainvoke(
            [SystemMessage(content=_CHUNK_SYSTEM_PROMPT), HumanMessage(content=chunk)]
        )
        actual = resp.usage_metadata.get("total_tokens", estimated) if resp.usage_metadata else estimated
        _rate_limiter.record_usage(actual)
        total_tokens += actual
        summaries.append(resp.content.strip())

    # Synthesis pass
    combined = "\n\n---\n\n".join(summaries)
    synth_user = _build_user_message(combined, existing_tags, existing_links, metadata)
    estimated = len(_enc.encode(synth_user)) + 1200
    await _rate_limiter.await_capacity(estimated)

    resp = await llm.ainvoke(
        [SystemMessage(content=_SYNTHESIS_PROMPT), HumanMessage(content=synth_user)]
    )
    actual = resp.usage_metadata.get("total_tokens", estimated) if resp.usage_metadata else estimated
    _rate_limiter.record_usage(actual)
    total_tokens += actual

    return _safe_json(resp.content.strip()), total_tokens


def _build_user_message(
    transcript: str,
    existing_tags: list[str],
    existing_links: dict[str, str],
    metadata: dict,
) -> str:
    tags_str = ", ".join(existing_tags[:100]) if existing_tags else "none"
    links_str = (
        "\n".join(f"  - {stem}: {title}" for stem, title in list(existing_links.items())[:50])
        if existing_links
        else "none"
    )
    return (
        f"METADATA:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"EXISTING VAULT TAGS (prefer these):\n{tags_str}\n\n"
        f"EXISTING NOTES (use stems for suggested_links):\n{links_str}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def _split_transcript(text: str, max_tokens: int) -> list[str]:
    """Split transcript into chunks of at most max_tokens tokens."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        wt = len(_enc.encode(word))
        if current_tokens + wt > max_tokens and current:
            chunks.append(" ".join(current))
            current = [word]
            current_tokens = wt
        else:
            current.append(word)
            current_tokens += wt

    if current:
        chunks.append(" ".join(current))
    return chunks


def _safe_json(text: str) -> dict:
    """Parse JSON, strip markdown fences if present."""
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        return {
            "title": "Untitled Memo",
            "summary": text[:500],
            "key_points": [],
            "action_items": [],
            "tags": [],
            "suggested_links": [],
            "category_override": None,
            "mermaid_diagram": None,
            "dataview_fields": {},
        }


def _log_rate_usage() -> None:
    report = _rate_limiter.usage_report()
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Groq Rate Usage", show_header=True)
    table.add_column("Metric")
    table.add_column("Used")
    table.add_column("Limit")
    table.add_row("RPM", str(report["rpm_used"]), str(report["rpm_limit"]))
    table.add_row("TPM", str(report["tpm_used"]), str(report["tpm_limit"]))
    table.add_row("RPD", str(report["rpd_used"]), str(report["rpd_limit"]))
    console.print(table)
