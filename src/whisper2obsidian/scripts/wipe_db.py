"""
wipe_db.py – Command-line utility to clear the w2o tracking database.
"""

from __future__ import annotations

import logging
import sqlite3
import sys

from rich.console import Console

from whisper2obsidian.config import settings

console = Console()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    db_path = settings.processed_db

    if not db_path.exists() and not settings.chroma_db_dir.exists():
        console.print(
            f"[yellow]Databases not found at {db_path} or {settings.chroma_db_dir}[/yellow]"
        )
        return

    console.print(
        f"[bold red]WARNING:[/] This will delete all tracking records in [cyan]{db_path}[/]"
    )
    console.print(
        f"It will also wipe all semantic vector embeddings in [cyan]{settings.chroma_db_dir}[/]"
    )
    console.print(
        "This forces whisper2obsidian to completely re-process all audio files "
        "and re-index your entire vault next time it runs."
    )

    try:
        response = console.input("\nAre you sure you want to completely wipe the database? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)

        # Connect directly to clear the tables
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM notes;")
                conn.execute("DELETE FROM tags;")
                conn.execute("DELETE FROM links;")
                conn.commit()
            console.print("[bold green]✓ SQLite database successfully wiped.[/bold green]")

        # Clear ChromaDB vector collection
        if settings.chroma_db_dir.exists():
            from whisper2obsidian.services.vector_index import VectorIndex

            try:
                vector_index = VectorIndex(settings.chroma_db_dir)
                vector_index.client.delete_collection("vault_notes")
                console.print(
                    "[bold green]✓ ChromaDB vector embeddings successfully wiped.[/bold green]"
                )
            except ValueError:
                # Occurs if the collection does not exist
                console.print(
                    "[yellow]ChromaDB collection 'vault_notes' was already empty "
                    "or did not exist.[/yellow]"
                )
            except Exception as e:
                console.print(f"[bold red]Error wiping ChromaDB:[/] {e}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error wiping database:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
