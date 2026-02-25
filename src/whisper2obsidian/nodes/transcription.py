"""
transcription_node – Transcribes a .m4a audio file using mlx-whisper
on Apple Silicon (Neural Engine + Metal GPU).

mlx-whisper handles AAC/M4A natively via ffmpeg under the hood.

Transcript caching
------------------
After a successful transcription the full text is written as a plain-text
sidecar: ``<audio_stem>.transcript.txt`` (same folder as the audio file).

On subsequent runs, if that file already exists, Whisper is skipped and the
cached transcript is loaded directly – saving time and compute.
To force a re-transcription simply delete (or rename) the .transcript.txt file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tiktoken

from whisper2obsidian.config import settings
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)

# tiktoken encoder for rough token estimation (cl100k_base is close enough)
_enc = tiktoken.get_encoding("cl100k_base")


def _transcript_cache_path(audio_path: str) -> Path:
    """Return the expected .transcript.txt path for a given audio file."""
    audio = Path(audio_path)
    return audio.parent / (audio.stem + ".transcript.txt")


def transcription_node(state: W2OState) -> W2OState:
    """
    Run mlx-whisper on `state['audio_path']` and populate:
      - transcript
      - language
      - transcript_token_count

    If a cached .transcript.txt file exists next to the audio, Whisper is
    skipped and the cached text is used instead.
    """
    audio_path = state.get("audio_path", "")
    if not audio_path:
        return {**state, "errors": ["transcription_node: audio_path is empty"]}

    cache_path = _transcript_cache_path(audio_path)

    # ── 1. Try loading from cache ─────────────────────────────────────────────
    if cache_path.exists():
        try:
            transcript = cache_path.read_text(encoding="utf-8").strip()
            if transcript:
                token_count = len(_enc.encode(transcript))
                logger.info(
                    "Loaded transcript from cache: %s (%d chars, ~%d tokens)",
                    cache_path.name,
                    len(transcript),
                    token_count,
                )
                return {
                    **state,
                    "transcript": transcript,
                    "language": "cached",   # language unknown from cache
                    "transcript_token_count": token_count,
                }
            logger.warning("Cache file %s is empty – falling through to Whisper", cache_path.name)
        except OSError as exc:
            logger.warning("Could not read transcript cache (%s): %s – re-transcribing", cache_path, exc)

    # ── 2. Transcribe with Whisper ────────────────────────────────────────────
    logger.info("Transcribing %s with model %s", audio_path, settings.whisper_model)

    try:
        import mlx_whisper  # type: ignore[import]

        result: dict = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=settings.whisper_model,
            verbose=False,
        )
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        return {**state, "errors": [f"Transcription error: {exc}"]}

    transcript: str = result.get("text", "").strip()
    language: str = result.get("language", "unknown")
    token_count: int = len(_enc.encode(transcript))

    logger.info(
        "Transcription complete: %d chars, ~%d tokens, language=%s",
        len(transcript),
        token_count,
        language,
    )

    # ── 3. Write transcript to cache ──────────────────────────────────────────
    try:
        cache_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript cached → %s", cache_path.name)
    except OSError as exc:
        logger.warning("Could not write transcript cache (%s): %s", cache_path, exc)

    return {
        **state,
        "transcript": transcript,
        "language": language,
        "transcript_token_count": token_count,
    }
