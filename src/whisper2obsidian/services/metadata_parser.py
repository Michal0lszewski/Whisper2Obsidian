"""
metadata_parser.py – Parse Voice Record Pro companion sidecar files.

Voice Record Pro saves audio as .m4a and optionally writes a companion
file with recording metadata. Supported formats:
  - JSON sidecar     (.json same stem as .m4a)
  - XML sidecar      (.xml same stem as .m4a)
  - Meta-txt sidecar (.m4a.meta.txt  or  .meta.txt same stem as .m4a)
  - Fallback:        empty metadata dict when no sidecar exists

Example JSON structure (Voice Record Pro):
{
  "title": "Meeting with team",
  "category": "Meeting",
  "date": "2024-01-15T10:30:00",
  "duration": 125.4,
  "location": "",
  "notes": ""
}

Example .meta.txt structure (Voice Record Pro ≥ 4.x):
  File Name           : 20260225-094601.m4a
  Title               : 25 February 2026 09:46:01
  Creation Date       : Wednesday, 25 February 2026 at 09:46:01 ...
  Duration            : 00:00:28
  Category            : Ideas
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known categories from Voice Record Pro (mapped to template keys).
# Matching is case-insensitive (values are lowercased before lookup in _normalise).
CATEGORY_MAP: dict[str, str] = {
    # ── Books / Reading ────────────────────────────────────────────────
    "books": "books",
    "book": "books",
    "reading": "books",
    # ── Course / Learning ──────────────────────────────────────────────
    "course": "course",
    "courses": "course",
    "lecture": "course",
    "lectures": "course",
    "class": "course",
    # ── Generic / Default ──────────────────────────────────────────────
    "generic": "default",
    "general": "default",
    "note": "default",
    "notes": "default",
    "memo": "default",
    "memos": "default",
    "": "default",
    # ── Ideas ──────────────────────────────────────────────────────────
    "ideas": "idea",
    "idea": "idea",
    "brainstorm": "idea",
    "inspiration": "idea",
    # ── Meeting ────────────────────────────────────────────────────────
    "meeting": "meeting",
    "meetings": "meeting",
    # ── Podcast ────────────────────────────────────────────────────────
    "podcast": "podcast",
    "podcasts": "podcast",
    # ── Research ───────────────────────────────────────────────────────
    "research": "research",
    # ── Shopping ───────────────────────────────────────────────────────
    "shopping": "shopping",
    "shop": "shopping",
    "grocery": "shopping",
    "groceries": "shopping",
    # ── Todo / Tasks ───────────────────────────────────────────────────
    "todo": "todo",
    "todos": "todo",
    "task": "todo",
    "tasks": "todo",
    "reminder": "todo",
    "reminders": "todo",
    # ── Misc ───────────────────────────────────────────────────────────
    "journal": "default",
    "personal": "default",
}


def parse_metadata(audio_path: str | Path) -> dict[str, Any]:
    """
    Find and parse the sidecar file for `audio_path`.

    Returns a normalised metadata dict with at least these keys:
        title, category, template_key, date, duration, location, notes, raw
    """
    audio = Path(audio_path)
    stem = audio.stem

    meta: dict[str, Any] = {}

    # 1. Try JSON sidecar
    json_path = audio.with_suffix(".json")
    if json_path.exists():
        meta = _parse_json(json_path)
        logger.debug("Loaded JSON sidecar: %s", json_path)

    # 2. Try XML sidecar
    elif (xml_path := audio.with_suffix(".xml")).exists():
        meta = _parse_xml(xml_path)
        logger.debug("Loaded XML sidecar: %s", xml_path)

    # 3. Try Voice Record Pro plain-text .meta.txt sidecar
    #    The app writes either  <stem>.m4a.meta.txt  or  <stem>.meta.txt
    elif (meta_txt := audio.parent / (audio.name + ".meta.txt")).exists():
        meta = _parse_meta_txt(meta_txt)
        logger.debug("Loaded .meta.txt sidecar: %s", meta_txt)

    elif (meta_txt2 := audio.with_suffix(".meta.txt")).exists():
        meta = _parse_meta_txt(meta_txt2)
        logger.debug("Loaded .meta.txt sidecar: %s", meta_txt2)

    # 4. Fallback – metadata from filename + mtime
    else:
        logger.info("No sidecar found for %s – using filename fallback", audio.name)
        mtime = datetime.fromtimestamp(audio.stat().st_mtime)
        meta = {
            "title": stem.replace("_", " ").replace("-", " ").title(),
            "date": mtime.isoformat(),
            "category": "",
            "duration": 0.0,
            "location": "",
            "notes": "",
        }

    return _normalise(meta, stem)


# ── Internal parsers ─────────────────────────────────────────────────────────


def _parse_meta_txt(path: Path) -> dict[str, Any]:
    """
    Parse a Voice Record Pro plain-text .meta.txt sidecar.

    Lines before the VOICE-RECORD-PRO-META-START sentinel are key-value pairs
    separated by " : " (with surrounding whitespace).  The binary blob after
    the sentinel is ignored – we already have everything we need above it.
    """
    data: dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read .meta.txt (%s): %s", path, exc)
        return data

    for line in text.splitlines():
        # Stop at the binary blob sentinel
        if line.strip().startswith("------VOICE-RECORD-PRO-META"):
            break
        if " : " not in line:
            continue
        key, _, value = line.partition(" : ")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key and value:
            data[key] = value

    # Map .meta.txt field names → canonical metadata keys
    mapped: dict[str, Any] = {}

    # Title
    mapped["title"] = data.get("title", "")

    # Category (as-is; _normalise will lower-case it)
    mapped["category"] = data.get("category", "")

    # Creation date – try parsing the verbose string VRP writes
    # e.g. "Wednesday, 25 February 2026 at 09:46:01 Central European Standard Time"
    raw_date = data.get("creation_date", "")
    if raw_date:
        import re
        # Remove day-of-week prefix ("Wednesday, ")
        cleaned = re.sub(r"^\w+,\s*", "", raw_date.strip())
        # Replace " at " connector with a space
        cleaned = re.sub(r"\s+at\s+", " ", cleaned)
        # Keep only the first 4 whitespace-separated tokens: DD Month YYYY HH:MM:SS
        # Everything after (timezone name) is discarded.
        tokens = cleaned.split()
        cleaned = " ".join(tokens[:4]) if len(tokens) >= 4 else cleaned
        for fmt in ("%d %B %Y %H:%M:%S", "%d %b %Y %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                mapped["date"] = dt.isoformat()
                break
            except ValueError:
                continue
        else:
            logger.debug("Could not parse creation_date from .meta.txt: %r", raw_date)

    # Duration – convert HH:MM:SS to total seconds (float)
    raw_dur = data.get("duration", "")
    if raw_dur:
        parts = raw_dur.split(":")
        try:
            if len(parts) == 3:
                mapped["duration"] = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                mapped["duration"] = int(parts[0]) * 60 + int(parts[1])
            else:
                mapped["duration"] = float(raw_dur)
        except ValueError:
            pass

    mapped["location"] = ""
    mapped["notes"] = ""
    return mapped

def _parse_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("JSON sidecar parse error (%s): %s", path, exc)
        return {}


def _parse_xml(path: Path) -> dict[str, Any]:
    """Parse a generic key-value XML sidecar as produced by Voice Record Pro."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        data: dict[str, Any] = {}

        # Handle both flat <recording><title>…</title></recording>
        # and attribute-based layouts
        for child in root:
            tag = child.tag.lower().strip()
            text = (child.text or "").strip()
            data[tag] = text

        # Some VRP XML versions use <entry key="…">value</entry>
        for entry in root.findall(".//entry"):
            key = (entry.get("key") or entry.get("name") or "").lower().strip()
            if key:
                data[key] = (entry.text or "").strip()

        return data
    except ET.ParseError as exc:
        logger.warning("XML sidecar parse error (%s): %s", path, exc)
        return {}


def _normalise(raw: dict[str, Any], stem: str) -> dict[str, Any]:
    """Return a clean, consistent metadata dict regardless of source format."""
    category_raw = str(raw.get("category", "")).strip().lower()
    template_key = CATEGORY_MAP.get(category_raw, "default")

    # Parse date – try ISO first, then common formats
    date_str = str(raw.get("date", "")).strip()
    try:
        parsed_date = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        parsed_date = datetime.now()

    # Duration – accept int/float seconds or a raw HH:MM:SS string
    raw_duration = raw.get("duration", 0.0)
    if isinstance(raw_duration, str) and ":" in raw_duration:
        parts = raw_duration.split(":")
        try:
            raw_duration = (
                int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                if len(parts) == 3
                else int(parts[0]) * 60 + int(parts[1])
            )
        except (ValueError, IndexError):
            raw_duration = 0.0
    duration_sec = float(raw_duration or 0.0)

    # Human-readable duration string  MM:SS  or  HH:MM:SS
    total_s = int(duration_sec)
    hours, remainder = divmod(total_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_display = (
        f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours
        else f"{minutes:02d}:{seconds:02d}"
    )

    return {
        "title": str(raw.get("title", stem)).strip() or stem,
        "category": category_raw,
        "template_key": template_key,
        "date": parsed_date.isoformat(),
        "date_display": parsed_date.strftime("%Y-%m-%d"),
        "duration": duration_sec,
        "duration_display": duration_display,
        "location": str(raw.get("location", "")).strip(),
        "notes": str(raw.get("notes", "")).strip(),
        "raw": raw,
    }
