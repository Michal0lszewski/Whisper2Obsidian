"""
vector_index.py – ChromaDB wrapper for semantic search.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from whisper2obsidian.config import settings

logger = logging.getLogger(__name__)


class VectorIndex:
    """Wrapper around ChromaDB for embedding and semantic search of notes."""

    def __init__(self, db_dir: Path | None = None) -> None:
        if db_dir is None:
            db_dir = settings.chroma_db_dir
        
        db_dir.parent.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        # We use default all-MiniLM-L6-v2 embedding function built into Chroma
        self.collection = self.client.get_or_create_collection(name="vault_notes")

    def upsert_note(self, stem: str, title: str, summary: str, rel_path: str) -> None:
        """Add or update a note's embedding in the vector database."""
        if not summary.strip():
            logger.warning("Empty summary provided for %s, skipping embedding.", stem)
            return

        try:
            self.collection.upsert(
                documents=[summary],
                ids=[stem],
                metadatas=[{
                    "title": title,
                    "path": rel_path
                }]
            )
            logger.debug("Vector upserted for %s", stem)
        except Exception as e:
            logger.error("Failed to upsert vector for %s: %s", stem, e)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search the vault for semantically similar notes."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Reformat ChromaDB output into a list of dicts
            matches = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][i]
                    metadata = results["metadatas"][0][i] or {}
                    document = results["documents"][0][i] if results["documents"] else ""
                    # The distance is typically L2 or cosine distance
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    matches.append({
                        "stem": doc_id,
                        "title": metadata.get("title", doc_id),
                        "path": metadata.get("path", ""),
                        "summary": document,
                        "distance": distance
                    })
            return matches
        except Exception as e:
            logger.error("Vector search failed for query '%s': %s", query, e)
            return []

    def delete_note(self, stem: str) -> None:
        """Remove a note from the vector database."""
        try:
            self.collection.delete(ids=[stem])
        except Exception:
            pass
