from langchain_core.tools import tool

from whisper2obsidian.config import settings
from whisper2obsidian.services.vault_index import VaultIndex
from whisper2obsidian.services.vector_index import VectorIndex


@tool
def search_similar_notes(query: str, n_results: int = 5) -> str:
    """
    Search the existing Obsidian vault for notes that are semantically similar to the query.
    Call this tool when you want to find related notes to link to.

    Args:
        query: A semantic search string describing the concepts you are looking for.
        n_results: Max number of notes to return (default 5).

    Returns:
        A formatted string listing the top matching notes with their summaries.
    """
    vector_db = VectorIndex()
    matches = vector_db.search(query, n_results=n_results)

    if not matches:
        return "No similar notes found in the vault."

    lines = ["Found the following similar notes:"]
    for i, m in enumerate(matches, 1):
        lines.append(f"\n{i}. Note: [[{m['stem']}]] - Title: {m['title']}")
        lines.append(f"   Summary: {m['summary']}")

    return "\n".join(lines)


@tool
def get_known_tags() -> str:
    """
    Retrieve all existing tags used in the Vault.
    Call this tool to avoid creating new tags when an existing equivalent tag already exists.

    Returns:
        A comma-separated string of all known tags.
    """
    index = VaultIndex(settings.processed_db)
    tags = index.all_tags()
    if not tags:
        return "No tags currently exist in the vault."
    return "Existing tags: " + ", ".join(tags)
