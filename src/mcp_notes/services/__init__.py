"""Service layer for mcp-notes.

Provides clean separation between MCP tool handlers and business logic.
"""

from mcp_notes.services.note_service import (
    NoteService,
    NoteOperationResult,
)

__all__ = [
    "NoteService",
    "NoteOperationResult",
]
