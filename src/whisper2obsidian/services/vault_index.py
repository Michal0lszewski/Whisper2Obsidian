"""
vault_index.py – SQLite-backed store for Obsidian vault tags and wiki-links.

Schema
------
  notes  (stem TEXT PK, title TEXT, path TEXT, updated_at TEXT)
  tags   (tag TEXT, stem TEXT)
  links  (from_stem TEXT, to_stem TEXT)

Used by:
  - vault_indexer node  (reads existing tags/links into state)
  - file_writer node    (inserts new note after writing)
  - vault_harvest.py    (bulk-index whole vault on first run / cron)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex to find [[wiki-links]] in Markdown (with optional |alias)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")
# Regex for YAML front-matter tags list
_YAML_TAGS_RE = re.compile(r"^tags:\s*\[?([^\]\n]+)\]?", re.MULTILINE | re.IGNORECASE)
# Regex for inline tags  #tag-name
_INLINE_TAG_RE = re.compile(r"(?<!\w)#([\w/-]+)")
# Regex for Dataview inline fields  key:: value
_DATAVIEW_RE = re.compile(r"^(\w[\w ]*)::\s*(.+)$", re.MULTILINE)


class VaultIndex:
    """Thin wrapper around a SQLite DB for vault metadata."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    # ── Schema ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS notes (
                    stem       TEXT PRIMARY KEY,
                    title      TEXT,
                    path       TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tags (
                    tag  TEXT NOT NULL,
                    stem TEXT NOT NULL,
                    UNIQUE(tag, stem)
                );
                CREATE TABLE IF NOT EXISTS links (
                    from_stem TEXT NOT NULL,
                    to_stem   TEXT NOT NULL,
                    UNIQUE(from_stem, to_stem)
                );
                CREATE INDEX IF NOT EXISTS idx_tags_tag  ON tags(tag);
                CREATE INDEX IF NOT EXISTS idx_links_from ON links(from_stem);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_note(self, stem: str, title: str, path: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO notes (stem, title, path, updated_at) VALUES (?,?,?,?)",
                (stem, title, path, now),
            )

    def upsert_tags(self, stem: str, tags: list[str]) -> None:
        with self._connect() as conn:
            for tag in tags:
                tag = tag.strip().lower().lstrip("#")
                if tag:
                    conn.execute(
                        "INSERT OR IGNORE INTO tags (tag, stem) VALUES (?, ?)", (tag, stem)
                    )

    def upsert_links(self, from_stem: str, to_stems: list[str]) -> None:
        with self._connect() as conn:
            for to in to_stems:
                if to:
                    conn.execute(
                        "INSERT OR IGNORE INTO links (from_stem, to_stem) VALUES (?, ?)",
                        (from_stem, to),
                    )

    def delete_note(self, stem: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM notes WHERE stem = ?", (stem,))
            conn.execute("DELETE FROM tags  WHERE stem = ?", (stem,))
            conn.execute("DELETE FROM links WHERE from_stem = ?", (stem,))

    # ── Read ─────────────────────────────────────────────────────────────────

    def all_tags(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT tag FROM tags ORDER BY tag").fetchall()
        return [r["tag"] for r in rows]

    def all_notes(self) -> dict[str, str]:
        """Return {stem: title} for all indexed notes."""
        with self._connect() as conn:
            rows = conn.execute("SELECT stem, title FROM notes ORDER BY stem").fetchall()
        return {r["stem"]: r["title"] for r in rows}

    def tags_for_note(self, stem: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag FROM tags WHERE stem = ?", (stem,)
            ).fetchall()
        return [r["tag"] for r in rows]

    def notes_with_tag(self, tag: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stem FROM tags WHERE tag = ?", (tag.lower(),)
            ).fetchall()
        return [r["stem"] for r in rows]

    def mark_processed(self, stem: str) -> None:
        """Record that an audio file has been processed (reuses notes table stem)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO notes (stem, title, path, updated_at) VALUES (?,?,?,?)",
                (stem, stem, "", datetime.now(UTC).isoformat()),
            )

    def is_processed(self, stem: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM notes WHERE stem = ?", (stem,)
            ).fetchone()
        return row is not None

    def processed_stems(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT stem FROM notes").fetchall()
        return [r["stem"] for r in rows]

    # ── Vault harvest helpers ─────────────────────────────────────────────────

    def index_markdown_file(self, md_path: Path) -> None:
        """Parse a single .md file and update all tables."""
        stem = md_path.stem
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", md_path, exc)
            return

        title = _extract_title(content, stem)
        tags = _extract_tags(content)
        links = [m.strip().lower().replace(" ", "-") for m in _WIKILINK_RE.findall(content)]

        self.upsert_note(stem, title, str(md_path))
        self.upsert_tags(stem, tags)
        self.upsert_links(stem, links)


# ── Module-level helpers ─────────────────────────────────────────────────────

def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _extract_tags(content: str) -> list[str]:
    tags: set[str] = set()
    # YAML front-matter tags
    for m in _YAML_TAGS_RE.finditer(content):
        for t in re.split(r"[,\s]+", m.group(1)):
            t = t.strip().strip('"').strip("'")
            if t:
                tags.add(t.lower())
    # Inline #tags (but not headings)
    for m in _INLINE_TAG_RE.finditer(content):
        tags.add(m.group(1).lower())
    return sorted(tags)


def iter_vault_md(vault_path: Path) -> Generator[Path, None, None]:
    """Yield all .md files in the vault, skipping hidden dirs."""
    for md in vault_path.rglob("*.md"):
        if not any(part.startswith(".") for part in md.parts):
            yield md
