"""
transcription_node – Transcribes a .m4a audio file using mlx-whisper
on Apple Silicon (Neural Engine + Metal GPU).

mlx-whisper handles AAC/M4A natively via ffmpeg under the hood.

Transcript caching
------------------
After a successful transcription two sidecar files are written next to the
audio file, inheriting its stem:

  <stem>.txt   – plain-text transcript (UTF-8)
  <stem>.json  – detected language, token count, timestamp

On subsequent runs, if <stem>.txt already exists, Whisper is skipped and the
cached transcript is loaded.  Language is restored from <stem>.json.
To force a re-transcription simply delete the .txt (and optionally .json) file.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import tiktoken

from whisper2obsidian.config import settings
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)

# tiktoken encoder for rough token estimation (cl100k_base is close enough)
_enc = tiktoken.get_encoding("cl100k_base")


def transcript_txt_path(audio_path: str | Path) -> Path:
    """Return the <stem>.txt transcript sidecar path for a given audio file."""
    audio = Path(audio_path)
    return audio.with_suffix(".txt")


def transcript_json_path(audio_path: str | Path) -> Path:
    """Return the <stem>.json metadata sidecar path for a given audio file."""
    audio = Path(audio_path)
    return audio.with_suffix(".json")


def transcription_node(state: W2OState) -> W2OState:
    """
    Run mlx-whisper on `state['audio_path']` and populate:
      - transcript
      - language
      - transcript_token_count

    Cache behaviour
    ---------------
    * If <stem>.txt exists alongside the audio, Whisper is skipped and the
      cached text is loaded.  Language is read from <stem>.json (if present).
    * After a fresh transcription both <stem>.txt and <stem>.json are written.
    """
    audio_path = state.get("audio_path", "")
    if not audio_path:
        return {**state, "errors": ["transcription_node: audio_path is empty"]}

    txt_path  = transcript_txt_path(audio_path)
    json_path = transcript_json_path(audio_path)

    # ── 1. Cache hit: load transcript from <stem>.txt ─────────────────────────
    if txt_path.exists():
        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if transcript:
                token_count = len(_enc.encode(transcript))

                # Restore language from companion .json if available
                language = "unknown"
                if json_path.exists():
                    try:
                        meta = json.loads(json_path.read_text(encoding="utf-8"))
                        language = meta.get("language", "unknown")
                    except (OSError, json.JSONDecodeError):
                        pass

                logger.info(
                    "Transcript loaded from cache: %s (%d chars, ~%d tokens, lang=%s)",
                    txt_path.name, len(transcript), token_count, language,
                )
                return {
                    **state,
                    "transcript": transcript,
                    "language": language,
                    "transcript_token_count": token_count,
                }
            logger.warning("Cache file %s is empty – re-transcribing", txt_path.name)
        except OSError as exc:
            logger.warning("Cannot read transcript cache (%s): %s – re-transcribing", txt_path, exc)

    # ── 2. Fresh transcription with Whisper ───────────────────────────────────
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
    language: str   = result.get("language", "unknown")
    token_count: int = len(_enc.encode(transcript))

    logger.info(
        "Transcription complete: %d chars, ~%d tokens, language=%s",
        len(transcript), token_count, language,
    )

    # ── 3. Write <stem>.txt and <stem>.json ───────────────────────────────────
    try:
        txt_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript written → %s", txt_path.name)
    except OSError as exc:
        logger.warning("Cannot write transcript cache (%s): %s", txt_path, exc)

    try:
        sidecar_metadata = state.get("metadata", {})
        meta = {
            "language": language,
            "token_count": token_count,
            "detected_at": datetime.now(UTC).isoformat(),
            "vrp_recording_date": sidecar_metadata.get("date", ""),
            "audio_file": Path(audio_path).name,
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Transcript metadata written → %s", json_path.name)
    except OSError as exc:
        logger.warning("Cannot write transcript metadata (%s): %s", json_path, exc)

    return {
        **state,
        "transcript": transcript,
        "language": language,
        "transcript_token_count": token_count,
    }
