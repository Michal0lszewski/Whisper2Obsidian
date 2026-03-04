"""
summarizer.py – Generates short semantic summaries of Markdown notes for vector embeddings.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from whisper2obsidian.config import settings
from whisper2obsidian.services.llm_rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


def _get_llm():
    """Return configured LLM based on API keys available."""
    # Prioritize Cerebras if key exists, otherwise use Groq
    if settings.cerebras_api_key:
        return ChatOpenAI(
            api_key=settings.cerebras_api_key,
            base_url="https://api.cerebras.ai/v1",
            model=settings.cerebras_model,
        )
    elif settings.groq_api_key:
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
    else:
        return None


def summarize_note(content: str, max_chars: int = 500) -> str:
    """
    Use LLM to generate a strict, short summary of the note content.
    If API fails or is not configured, fall back to simple text truncation.
    """
    if not content.strip():
        return ""

    llm = _get_llm()
    if not llm:
        logger.warning("No LLM configured for summarization, falling back to truncation")
        return _fallback_summary(content, max_chars)

    prompt = f"""
Summarize the following Obsidian note in a highly dense, semantic paragraph of NO MORE 
than {max_chars} characters.
Focus entirely on the exact concepts, entities, and core meaning of the text. Do not include 
conversational filler like "This note is about".

NOTE CONTENT:
{content}
"""
    try:
        # Approximate token cost for LLMRateLimiter
        estimated_tokens = len(prompt) // 4 + 100

        provider = "cerebras" if settings.cerebras_api_key else "groq"
        rate_limiter = get_rate_limiter(provider)

        import asyncio

        # We need to run await_capacity which is async inside a sync function.
        # This summarizer is technically sync because it uses invoke() and is called
        # from vector index mapping. If we're inside an asyncio loop we can handle this.
        # but since summarize_note is a normal def we must try/except.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(rate_limiter.await_capacity(estimated_tokens))
        except RuntimeError:
            asyncio.run(rate_limiter.await_capacity(estimated_tokens))

        messages = [
            SystemMessage(
                content="You are an expert summarizer. Generate extreme high-density "
                "semantic summaries."
            ),
            HumanMessage(content=prompt),
        ]

        # Some models fail if we don't control max_tokens
        response = llm.invoke(messages)

        summary = str(response.content).strip()
        # Ensure it respects length just in case the LLM ignored instructions
        if len(summary) > max_chars + 100:
            summary = summary[:max_chars] + "..."
        return summary

    except Exception as e:
        logger.error("LLM Summarization failed, using fallback: %s", e)
        return _fallback_summary(content, max_chars)


def _fallback_summary(content: str, max_chars: int) -> str:
    """Simple truncation fallback if LLM is unavailable."""
    # Strip some massive markdown elements if needed, but for now just truncate
    clean = content.replace("```", " ").strip()
    if len(clean) > max_chars:
        return clean[:max_chars] + "..."
    return clean
