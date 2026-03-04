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
import time
from typing import Any

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from whisper2obsidian.config import settings
from whisper2obsidian.services.llm_rate_limiter import get_rate_limiter
from whisper2obsidian.state import W2OState
from whisper2obsidian.tools.vault_tools import get_known_tags, search_similar_notes

logger = logging.getLogger(__name__)

# Max tokens we'll send to Groq in a single chunk (leaves room for prompt + reply)
CHUNK_TOKEN_LIMIT = 6_000

_enc = tiktoken.get_encoding("cl100k_base")

_enc = tiktoken.get_encoding("cl100k_base")

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert knowledge manager helping convert voice memo transcripts into
structured Obsidian notes. Analyse the transcript and return ONLY valid JSON
(no markdown, no explanation) with this exact schema:

{
  "thought_process": "Brief step-by-step reasoning on what tools to use and why",
  "title": "concise note title",
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2"],
  "action_items": ["action 1"],
  "tags": [],
  "suggested_links": [],
  "category_override": null,
  "mermaid_diagram": null,
  "dataview_fields": {}
}

You have access to tools that can search for similar existing notes and retrieve known vault tags. 
You MUST NOT guess tags or links without verifying them. You MUST call `get_known_tags` and `search_similar_notes` before completing the JSON.

Rules:
- PERSPECTIVE: Write ALL text in the FIRST-PERSON ("I need to", "My idea is") as if YOU
  dictated this memo. NEVER use third-person ("The user wants", "The speaker").
- tags: MUST BE AN EMPTY ARRAY `[]` unless the transcript explicitly discusses the exact subject.
  Do NOT invent abstract connections.
- suggested_links: MUST BE AN EMPTY ARRAY `[]` unless directly, undeniably related.
- EXPLICIT SPOKEN INSTRUCTIONS: I will often dictate metadata commands at the end of the memo
  (e.g. "Tag this with CSF", "Link to NotebookLM"). You MUST obey these instructions! If I dictate
  a tag or link, add it to the arrays, and exclude the command from the final text summary.
- mermaid_diagram: provide a Mermaid flowchart string ONLY for process/workflow memos, else null.
- category_override: ONLY use one of these exact values if the transcript clearly belongs
  to a different category than the metadata claims, else null:
  "books", "course", "generic", "ideas", "meeting", "podcast", "research", "shopping", "todo"
- dataview_fields: any key::value pairs useful for Dataview queries (e.g. "project", "status").
""").strip()

_CHUNK_SYSTEM_PROMPT = textwrap.dedent("""
You are summarising a chunk of a longer voice memo transcript.
Return ONLY a plain text summary of the key points in this chunk (no JSON).
Be concise, preserve all important facts/names, and strictly use FIRST-PERSON ("I", "my")
perspective.
""").strip()

_SYNTHESIS_PROMPT = textwrap.dedent("""
You are combining chunk summaries of a voice memo into a final structured analysis.
Use the same JSON schema as before:
{thought_process, title, summary, key_points, action_items, tags, suggested_links,
 category_override, mermaid_diagram, dataview_fields}

You have access to tools that can search for similar existing notes and retrieve known vault tags. 
You MUST NOT guess tags or links without verifying them. You MUST call `get_known_tags` and `search_similar_notes` before completing the JSON.

Strictly maintain FIRST-PERSON ("I", "my") perspective.
tags and suggested_links MUST BE AN EMPTY ARRAY `[]` unless they are undeniably, explicitly 
the core subject of the transcript. Do NOT invent connections.
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

    if settings.cerebras_api_key:
        provider = "cerebras"
        llm = ChatOpenAI(
            api_key=settings.cerebras_api_key,
            base_url="https://api.cerebras.ai/v1",
            model=settings.cerebras_model,
            temperature=0.3,
        )
    else:
        provider = "groq"
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.3,
        )

    rate_limiter = get_rate_limiter(provider)

    total_tokens_used = 0
    start_time = time.perf_counter()

    if token_count <= CHUNK_TOKEN_LIMIT:
        # ── Single-pass analysis ─────────────────────────────────────────
        analysis, tokens = await _analyse_single(
            llm, rate_limiter, transcript, existing_tags, existing_links, metadata
        )
        total_tokens_used = tokens
    else:
        # ── Chunked analysis ─────────────────────────────────────────────
        logger.info("Transcript too long (%d tokens) – splitting into chunks", token_count)
        analysis, tokens = await _analyse_chunked(
            llm, rate_limiter, transcript, existing_tags, existing_links, metadata
        )
        total_tokens_used = tokens

    elapsed = time.perf_counter() - start_time
    logger.info("Analysis complete in %.2fs. Tokens used: %d", elapsed, total_tokens_used)

    if settings.show_rate_usage:
        report = rate_limiter.usage_report()
        _log_rate_usage(report, provider)

    return {
        **state,
        "analysis": analysis,
        "groq_tokens_used": total_tokens_used,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _analyse_single(
    llm: ChatGroq | ChatOpenAI,
    rate_limiter,
    transcript: str,
    existing_tags: list[str],
    existing_links: dict[str, str],
    metadata: dict,
) -> tuple[dict, int]:
    # Construct base prompt
    user_content = _build_user_message(transcript, existing_tags, existing_links, metadata)

    # Track tokens manually for local safety via limiters
    estimated = len(_enc.encode(user_content)) + 1200
    await rate_limiter.await_capacity(estimated)

    tools = [search_similar_notes, get_known_tags]
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)]

    actual_tokens = 0

    # Run a simple tool loop
    for _ in range(3):  # Max 3 tool iterations
        try:
            resp = await llm_with_tools.ainvoke(messages)
            if resp.usage_metadata:
                actual_tokens += resp.usage_metadata.get("total_tokens", 0)

            messages.append(resp)

            if not resp.tool_calls:
                # No more tools, final answer
                break

            for tool_call in resp.tool_calls:
                logger.info(
                    "[bold magenta]🛠️  Agent actively called tool: %s[/bold magenta]",
                    tool_call["name"],
                    extra={"markup": True},
                )
                if tool_call["name"] == "search_similar_notes":
                    tool_res = search_similar_notes.invoke(tool_call["args"])
                elif tool_call["name"] == "get_known_tags":
                    tool_res = get_known_tags.invoke(tool_call["args"])
                else:
                    tool_res = f"Unknown tool {tool_call['name']}"

                messages.append(ToolMessage(content=str(tool_res), tool_call_id=tool_call["id"]))
        except Exception as e:
            logger.error("LLM evaluation / tool call failed: %s", e)
            break

    # If the LLM stopped using tools but didn't output JSON cleanly, or if it hit iteration limit
    final_output = messages[-1].content
    raw = final_output.strip()

    rate_limiter.record_usage(actual_tokens)
    return _safe_json(raw), actual_tokens


async def _analyse_chunked(
    llm: ChatGroq | ChatOpenAI,
    rate_limiter,
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
        await rate_limiter.await_capacity(estimated)

        resp = await llm.ainvoke(
            [SystemMessage(content=_CHUNK_SYSTEM_PROMPT), HumanMessage(content=chunk)]
        )
        actual = (
            resp.usage_metadata.get("total_tokens", estimated) if resp.usage_metadata else estimated
        )
        rate_limiter.record_usage(actual)
        total_tokens += actual
        summaries.append(resp.content.strip())

    # Synthesis pass
    combined = "\n\n---\n\n".join(summaries)
    synth_user = _build_user_message(combined, existing_tags, existing_links, metadata)
    estimated = len(_enc.encode(synth_user)) + 1200
    await rate_limiter.await_capacity(estimated)

    tools = [search_similar_notes, get_known_tags]
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=_SYNTHESIS_PROMPT), HumanMessage(content=synth_user)]

    actual_tokens = 0
    # Run a simple tool loop
    for _ in range(3):  # Max 3 tool iterations
        try:
            resp = await llm_with_tools.ainvoke(messages)
            if resp.usage_metadata:
                actual_tokens += resp.usage_metadata.get("total_tokens", 0)

            messages.append(resp)

            if not getattr(resp, "tool_calls", None):
                break

            for tool_call in resp.tool_calls:
                logger.info(
                    "[bold magenta]🛠️  Agent actively called tool: %s[/bold magenta]",
                    tool_call["name"],
                    extra={"markup": True},
                )
                if tool_call["name"] == "search_similar_notes":
                    tool_res = search_similar_notes.invoke(tool_call["args"])
                elif tool_call["name"] == "get_known_tags":
                    tool_res = get_known_tags.invoke(tool_call["args"])
                else:
                    tool_res = f"Unknown tool {tool_call['name']}"

                messages.append(ToolMessage(content=str(tool_res), tool_call_id=tool_call["id"]))
        except Exception as e:
            logger.error("LLM synthesis evaluation / tool call failed: %s", e)
            break

    rate_limiter.record_usage(actual_tokens)
    total_tokens += actual_tokens

    final_output = messages[-1].content
    return _safe_json(final_output.strip()), total_tokens


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


def _log_rate_usage(report: dict, provider: str) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{provider.capitalize()} Rate Usage", show_header=True)
    table.add_column("Metric")
    table.add_column("Used")
    table.add_column("Limit")
    table.add_row("RPM", str(report["rpm_used"]), str(report["rpm_limit"]))
    table.add_row("TPM", str(report["tpm_used"]), str(report["tpm_limit"]))
    table.add_row("RPD", str(report["rpd_used"]), str(report["rpd_limit"]))
    console.print(table)
