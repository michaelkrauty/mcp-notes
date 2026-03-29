"""Version history operations for notes.

Tools:
- get_note_history: Get version history from git
- restore_note_version: Restore a note to a previous version
"""

import logging
from uuid import UUID

from vector_core import validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.singletons import get_git, get_indexer, get_store
from mcp_notes.storage.filesystem import NoteNotFoundError

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_note_history(note_id: str, limit: int = 10) -> list[dict]:
    """
    Get version history for a note from git.

    Uses --follow to track history across file moves/renames.

    Args:
        note_id: Note UUID string
        limit: Max versions to return (default 10, max 100)

    Returns:
        List of version info
    """
    store = get_store()
    git = get_git()
    limit = validate_limit(limit, default=10)

    try:
        uuid = UUID(note_id)
    except ValueError:
        return [error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")]

    # Get current path for history tracking
    note_path = store.get_note_path(uuid)
    versions = git.get_history(uuid, limit=limit, path=note_path)
    return [v.model_dump(mode="json") for v in versions]


@mcp.tool()
async def restore_note_version(note_id: str, version_id: str) -> dict:
    """
    Restore a note to a previous version (creates new commit).

    Args:
        note_id: Note UUID string
        version_id: Git commit SHA to restore from

    Returns:
        Restored note
    """
    store = get_store()
    git = get_git()
    indexer = await get_indexer()

    try:
        uuid = UUID(note_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

    # Get title and current path
    try:
        note = store.read(uuid)
        title = note.title
        current_path = store.get_note_path(uuid)
    except NoteNotFoundError:
        title = "Unknown"
        current_path = None

    # Restore via git with current path
    commit_sha = git.restore_version(uuid, version_id, title, current_path=current_path)
    if not commit_sha:
        return error_response(ErrorCode.INTERNAL_ERROR, f"Failed to restore version {version_id}")

    # Re-index
    try:
        await indexer.index_note(uuid)
    except Exception as e:
        logger.warning(f"Failed to index restored note: {e}")

    # Return restored note
    try:
        note = store.read(uuid)
        return note.model_dump(mode="json")
    except NoteNotFoundError:
        return error_response(ErrorCode.NOTE_NOT_FOUND, "Restored note not found")
