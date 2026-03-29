"""Note linking operations.

Tools:
- get_note_links: Get incoming and outgoing links for a note
"""

from uuid import UUID

from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.singletons import get_links


@mcp.tool()
async def get_note_links(note_id: str) -> dict:
    """
    Get incoming and outgoing links for a note.

    Args:
        note_id: Note UUID string

    Returns:
        NoteLinks with outgoing, incoming (backlinks), and broken links
    """
    links = get_links()

    try:
        uuid = UUID(note_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

    note_links = links.get_note_links(uuid)
    return note_links.model_dump(mode="json")
