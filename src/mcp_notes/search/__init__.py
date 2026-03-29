"""Search functionality for notes."""

from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.search.filters import SearchFilters, parse_search_query

__all__ = ["parse_search_query", "SearchFilters", "NoteSearchEngine"]
