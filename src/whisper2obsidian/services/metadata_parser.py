"""
metadata_parser.py – Parse Voice Record Pro companion sidecar files.

Voice Record Pro saves audio as .m4a and optionally writes a companion
file with recording metadata. Supported formats:
  - JSON sidecar  (.json same stem as .m4a)
  - XML sidecar   (.xml same stem as .m4a)
  - Fallback:     empty metadata dict when no sidecar exists

Example JSON structure (Voice Record Pro):
{
  "title": "Meeting with team",
  "category": "Meeting",
  "date": "2024-01-15T10:30:00",
  "duration": 125.4,
  "location": "",
  "notes": ""
}
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known categories from Voice Record Pro (mapped to template keys)
CATEGORY_MAP: dict[str, str] = {
    "meeting": "meeting",
    "idea": "idea",
    "research": "research",
    "lecture": "research",
    "note": "default",
    "memo": "default",
    "reminder": "default",
    "": "default",
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

    # 3. Fallback – metadata from filename + mtime
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

    return {
        "title": str(raw.get("title", stem)).strip() or stem,
        "category": category_raw,
        "template_key": template_key,
        "date": parsed_date.isoformat(),
        "date_display": parsed_date.strftime("%Y-%m-%d"),
        "duration": float(raw.get("duration", 0.0) or 0.0),
        "location": str(raw.get("location", "")).strip(),
        "notes": str(raw.get("notes", "")).strip(),
        "raw": raw,
    }
