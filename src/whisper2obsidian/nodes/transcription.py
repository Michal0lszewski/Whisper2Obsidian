"""
transcription_node – Transcribes a .m4a audio file using mlx-whisper
on Apple Silicon (Neural Engine + Metal GPU).

mlx-whisper handles AAC/M4A natively via ffmpeg under the hood.
"""

from __future__ import annotations

import logging

import tiktoken

from whisper2obsidian.config import settings
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)

# tiktoken encoder for rough token estimation (cl100k_base is close enough)
_enc = tiktoken.get_encoding("cl100k_base")


def transcription_node(state: W2OState) -> W2OState:
    """
    Run mlx-whisper on `state['audio_path']` and populate:
      - transcript
      - language
      - transcript_token_count
    """
    audio_path = state.get("audio_path", "")
    if not audio_path:
        return {**state, "errors": ["transcription_node: audio_path is empty"]}

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

    return {
        **state,
        "transcript": transcript,
        "language": language,
        "transcript_token_count": token_count,
    }
