"""
vault_harvest.py – Standalone script to (re)index the entire Obsidian vault.

Run this once on first setup, then periodically (e.g. via launchd/cron)
to keep the SQLite tag/link index up to date with notes added outside
the Whisper2Obsidian pipeline.

Usage:
    python -m whisper2obsidian.scripts.vault_harvest
    # or via installed entrypoint:
    w2o-harvest
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex, iter_vault_md

console = Console()
logging.basicConfig(level=logging.WARNING)


def main() -> None:
    vault_path = settings.vault_path
    if not vault_path.exists():
        console.print(f"[red]Vault not found: {vault_path}[/red]")
        sys.exit(1)

    index = VaultIndex(settings.processed_db)
    md_files = list(iter_vault_md(vault_path))

    console.print(f"[bold blue]Vault Harvest[/bold blue] – {len(md_files)} notes found")

    for md_path in track(md_files, description="Indexing…"):
        index.index_markdown_file(md_path)

    tags = index.all_tags()
    notes = index.all_notes()
    console.print(
        f"[green]✓[/green] Indexed {len(notes)} notes, {len(tags)} unique tags. "
        f"DB: {settings.processed_db}"
    )


if __name__ == "__main__":
    main()
