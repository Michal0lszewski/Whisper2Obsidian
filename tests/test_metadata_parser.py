"""Tests for metadata_parser.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper2obsidian.services.metadata_parser import parse_metadata


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    """Create a dummy .m4a file."""
    f = tmp_path / "test_memo.m4a"
    f.write_bytes(b"\x00" * 100)
    return f


def test_parse_json_sidecar(audio_file: Path) -> None:
    data = {
        "title": "Team Standup",
        "category": "Meeting",
        "date": "2024-06-01T09:00:00",
        "duration": 300.0,
        "location": "Office",
        "notes": "Sprint review",
    }
    audio_file.with_suffix(".json").write_text(json.dumps(data))
    meta = parse_metadata(audio_file)

    assert meta["title"] == "Team Standup"
    assert meta["category"] == "meeting"
    assert meta["template_key"] == "meeting"
    assert meta["duration"] == 300.0
    assert meta["location"] == "Office"


def test_parse_xml_sidecar(audio_file: Path) -> None:
    xml_content = """<recording>
        <title>Research Notes</title>
        <category>Research</category>
        <date>2024-06-15T14:30:00</date>
        <duration>450</duration>
    </recording>"""
    audio_file.with_suffix(".xml").write_text(xml_content)
    meta = parse_metadata(audio_file)

    assert meta["title"] == "Research Notes"
    assert meta["category"] == "research"
    assert meta["template_key"] == "research"


def test_fallback_no_sidecar(audio_file: Path) -> None:
    meta = parse_metadata(audio_file)
    # Fallback uses filename
    assert "test" in meta["title"].lower() or "memo" in meta["title"].lower()
    assert meta["template_key"] == "default"
    assert meta["duration"] == 0.0


def test_malformed_json_sidecar(audio_file: Path) -> None:
    audio_file.with_suffix(".json").write_text("{bad json")
    # Should not raise; falls back to empty
    meta = parse_metadata(audio_file)
    assert meta["template_key"] == "default"


def test_unknown_category_maps_to_default(audio_file: Path) -> None:
    data = {"title": "Misc", "category": "grocery_list"}
    audio_file.with_suffix(".json").write_text(json.dumps(data))
    meta = parse_metadata(audio_file)
    assert meta["template_key"] == "default"
