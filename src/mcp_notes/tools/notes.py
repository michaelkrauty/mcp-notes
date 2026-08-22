"""Core note CRUD operations.

Tools:
- create_note: Create a new note with auto-generated UUID
- read_note: Read a note by its UUID
- update_note: Update an existing note
- delete_note: Delete a note
"""

from uuid import UUID

from mcp.server.mcpserver import Context
from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp, notify_note_resources
from mcp_notes.singletons import get_note_service


@mcp.tool()
async def create_note(
    title: str,
    content: str,
    tags: list[str] | None = None,
    category: str | None = None,
    context: Context | None = None,
) -> dict:
    """
    Create a new note with auto-generated UUID.

    Args:
        title: Note title
        content: Note body content (markdown)
        tags: Optional list of tags (lowercase, hyphenated)
        category: Optional category path (e.g., "work/projects")

    Returns:
        Created note as dict
    """
    service = await get_note_service()
    result = await service.create(title=title, content=content, tags=tags, category=category)
    if result.success:
        await notify_note_resources(context)
    return result.to_dict()


@mcp.tool()
async def read_note(note_id: str) -> dict:
    """
    Read a note by its UUID.

    Args:
        note_id: Note UUID string

    Returns:
        Note as dict
    """
    try:
        uuid = UUID(note_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

    service = await get_note_service()
    result = await service.read(uuid)
    return result.to_dict()


@mcp.tool()
async def update_note(
    note_id: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    category: str | None = None,
    context: Context | None = None,
) -> dict:
    """
    Update an existing note. Only provided fields are updated.

    Args:
        note_id: Note UUID string
        title: New title (optional, may cause file rename)
        content: New body content (optional)
        tags: New tags (optional, pass empty list to clear)
        category: New category (optional, may cause file move, pass empty string to clear)

    Returns:
        Updated note as dict
    """
    try:
        uuid = UUID(note_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

    service = await get_note_service()
    result = await service.update(
        note_id=uuid,
        title=title,
        content=content,
        tags=tags,
        category=category,
    )
    if result.success:
        await notify_note_resources(context)
    return result.to_dict()


@mcp.tool()
async def delete_note(note_id: str, context: Context | None = None) -> dict:
    """
    Delete a note. The note is removed from the filesystem and search index,
    but remains recoverable from git history.

    Args:
        note_id: Note UUID string

    Returns:
        Success status
    """
    try:
        uuid = UUID(note_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

    service = await get_note_service()
    result = await service.delete(uuid)

    if result.success:
        await notify_note_resources(context)
        return {"success": True, "deleted_id": note_id}
    return result.to_dict()
