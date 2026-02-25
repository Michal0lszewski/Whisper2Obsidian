"""
vault_indexer_node – Reads the SQLite vault index to provide existing
tags and links to downstream nodes for cross-referencing.
"""

from __future__ import annotations

import logging

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex
from whisper2obsidian.state import W2OState

logger = logging.getLogger(__name__)


def vault_indexer_node(state: W2OState) -> W2OState:
    """
    Load existing vault tags and note links from the SQLite index.
    These are injected into the analysis prompt so the LLM can suggest
    relevant cross-links and reuse existing tags.
    """
    index = VaultIndex(settings.processed_db)

    existing_tags = index.all_tags()
    existing_links = index.all_notes()   # {stem: title}

    logger.info(
        "Vault index: %d tags, %d notes available for cross-linking",
        len(existing_tags),
        len(existing_links),
    )

    return {
        **state,
        "existing_tags": existing_tags,
        "existing_links": existing_links,
    }
