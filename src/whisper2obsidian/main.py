"""
main.py – Entry point for the Whisper2Obsidian daemon.

Runs the LangGraph pipeline in a loop, checking for new memos
on every iteration (configurable interval).

Usage:
    python -m whisper2obsidian.main            # default 60-second poll
    python -m whisper2obsidian.main --once     # process one memo and exit
    python -m whisper2obsidian.main --interval 120
    python -m whisper2obsidian.main --show-rate-usage
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.logging import RichHandler

from whisper2obsidian.config import settings
from whisper2obsidian.graph import compile_graph
from whisper2obsidian.services.vault_index import VaultIndex

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper2Obsidian – voice memo to Obsidian note pipeline"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single memo and exit (no loop)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--show-rate-usage",
        action="store_true",
        help="Print Groq rate-usage table after each processed memo",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def run_once(show_rate_usage: bool = False) -> dict:
    """Run the full graph for a single memo.  Returns final state."""
    # Override show_rate_usage at runtime if CLI flag is set
    if show_rate_usage:
        import os
        os.environ["SHOW_RATE_USAGE"] = "true"

    graph = compile_graph()
    initial_state: dict = {
        "already_processed": [],
        "existing_tags": [],
        "existing_links": {},
        "errors": [],
        "messages": [],
    }

    log = logging.getLogger(__name__)
    log.info("Running Whisper2Obsidian pipeline…")

    final_state = graph.invoke(initial_state)

    if final_state.get("note_path"):
        console.print(
            f"[bold green]✓[/] Note written: {final_state['note_path']}"
        )
    elif final_state.get("audio_path") == "":
        console.print("[yellow]No new voice memos found.[/yellow]")
    else:
        console.print("[bold red]✗ Pipeline completed with errors[/]")
        for err in final_state.get("errors", []):
            console.print(f"  [red]• {err}[/red]")

    return final_state


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    log = logging.getLogger(__name__)

    # Ensure DB and vault dirs are initialised
    VaultIndex(settings.processed_db)

    console.rule("[bold blue]Whisper2Obsidian[/bold blue]")
    console.print(f"  Audio folder : {settings.audio_folder}")
    console.print(f"  Vault inbox  : {settings.inbox_path}")
    console.print(f"  Whisper model: {settings.whisper_model}")
    console.print(f"  Groq model   : {settings.groq_model}")
    console.print(
        f"  Rate limits  : {settings.groq_rpm_limit} RPM / "
        f"{settings.groq_tpm_limit} TPM / {settings.groq_rpd_limit} RPD"
    )
    console.rule()

    if args.once:
        run_once(show_rate_usage=args.show_rate_usage)
        return

    log.info("Starting daemon loop (interval=%ds). Ctrl-C to stop.", args.interval)
    try:
        while True:
            state = run_once(show_rate_usage=args.show_rate_usage)
            # If no new memo was found, sleep and try again
            if not state.get("audio_path"):
                time.sleep(args.interval)
            # If a memo was processed successfully, check immediately for the next one
            # (in case several memos were dropped at once); only sleep after none found
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
