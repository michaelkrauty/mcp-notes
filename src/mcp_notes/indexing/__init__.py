"""Note indexing for vector search."""

from mcp_notes.indexing.chunker import chunk_note
from mcp_notes.indexing.indexer import NoteIndexer

__all__ = ["chunk_note", "NoteIndexer"]
