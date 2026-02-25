"""
graph.py – Compiles the Whisper2Obsidian LangGraph.

Graph topology:
    START
      └─ watcher_node
          ├─ (no new memo) → END
          └─ (new memo) → transcription_node
                            └─ vault_indexer_node
                                └─ analysis_node
                                    └─ note_writer_node
                                        └─ file_writer_node
                                            └─ END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from whisper2obsidian.nodes.analysis import analysis_node
from whisper2obsidian.nodes.file_writer import file_writer_node
from whisper2obsidian.nodes.note_writer import note_writer_node
from whisper2obsidian.nodes.transcription import transcription_node
from whisper2obsidian.nodes.vault_indexer import vault_indexer_node
from whisper2obsidian.nodes.watcher import has_new_memo, watcher_node
from whisper2obsidian.state import W2OState


def compile_graph():
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(W2OState)

    # Register nodes
    builder.add_node("watcher", watcher_node)
    builder.add_node("transcription", transcription_node)
    builder.add_node("vault_indexer", vault_indexer_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("note_writer", note_writer_node)
    builder.add_node("file_writer", file_writer_node)

    # Entry point
    builder.add_edge(START, "watcher")

    # Conditional: new memo found → full pipeline, else → END
    builder.add_conditional_edges(
        "watcher",
        has_new_memo,
        {
            "transcription": "transcription",
            "end": END,
        },
    )

    # Linear pipeline
    builder.add_edge("transcription", "vault_indexer")
    builder.add_edge("vault_indexer", "analysis")
    builder.add_edge("analysis", "note_writer")
    builder.add_edge("note_writer", "file_writer")
    builder.add_edge("file_writer", END)

    return builder.compile()
