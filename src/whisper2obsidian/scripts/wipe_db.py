"""
wipe_db.py – Command-line utility to clear the w2o tracking database.
"""

from __future__ import annotations

import logging
import sqlite3
import sys

from rich.console import Console

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex

console = Console()

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    db_path = settings.processed_db

    if not db_path.exists():
        console.print(f"[yellow]Database not found at {db_path}[/yellow]")
        return

    console.print(f"[bold red]WARNING:[/] This will delete all tracking records in [cyan]{db_path}[/]")
    console.print("This forces whisper2obsidian to completely re-process all audio files next time it runs.")
    
    try:
        response = console.input("\nAre you sure you want to completely wipe the database? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)
            
        # Connect directly to clear the tables
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM notes;")
            conn.execute("DELETE FROM tags;")
            conn.execute("DELETE FROM links;")
            conn.commit()
            
        console.print("[bold green]✓ Database successfully wiped.[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error wiping database:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
