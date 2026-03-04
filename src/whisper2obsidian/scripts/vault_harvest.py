#!/usr/bin/env python3
"""
vault_harvest.py – CLI script to bulk-sync the Obsidian vault.
It scans all .md files, creates missing database entries, computes 
semantic summaries using the LLM, and populates ChromaDB.
"""

import logging
import sys

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex

logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vault_harvest")

def main() -> int:
    index = VaultIndex(settings.processed_db)
    
    logger.info("Starting vault harvest process...")
    logger.info("Target vault: %s", settings.vault_path)
    logger.info("This will summarize and embed any new or changed .md files.")
    
    try:
        index.sync_vault(settings.vault_path)
        logger.info("Vault harvest complete!")
        return 0
    except KeyboardInterrupt:
        logger.info("Harvest interrupted by user.")
        return 1
    except Exception as e:
        logger.error("Error during harvest: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
