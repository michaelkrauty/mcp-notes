"""Search and list operations for notes.

Tools:
- search_notes: Hybrid semantic + keyword search
- list_notes: List notes with filtering
- find_similar_notes: Find semantically similar notes
"""

from uuid import UUID

from vector_core import validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.constants import normalize_tag
from mcp_notes.singletons import get_search, get_store
from mcp_notes.storage.slugify import slugify_category_path


@mcp.tool()
async def search_notes(
    query: str,
    limit: int = 10,
    tags: list[str] | None = None,
    category: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """
    Hybrid semantic + keyword search across all notes.

    Query syntax supports filters:
    - tag:tagname - Filter by tag
    - category:path - Filter by category
    - after:YYYY-MM-DD - Created after date
    - before:YYYY-MM-DD - Created before date
    - title:text - Title contains text

    Args:
        query: Search query with optional filters
        limit: Max results (default 10, max 100)
        tags: Additional tag filters
        category: Additional category filter
        after: Created after date (ISO format)
        before: Created before date (ISO format)

    Returns:
        List of search results
    """
    search = await get_search()
    limit = validate_limit(limit, default=10)

    results = await search.search(
        query=query,
        limit=limit,
        tags=tags,
        category=category,
        after=after,
        before=before,
    )

    return [r.model_dump(mode="json") for r in results]


@mcp.tool()
async def list_notes(
    tags: list[str] | None = None,
    category: str | None = None,
    sort_by: str = "modified",
    limit: int = 50,
) -> list[dict]:
    """
    List notes with optional filtering.

    Args:
        tags: Filter by tags (all must match)
        category: Filter by category prefix
        sort_by: Sort field - "modified" (default), "created", or "title"
        limit: Max results (default 50, max 100)

    Returns:
        List of note summaries, or a single-item list with an error dict if
        sort_by is not one of the supported fields.
    """
    valid_sort_fields = ("modified", "created", "title")
    if sort_by not in valid_sort_fields:
        return [error_response(
            ErrorCode.INVALID_INPUT,
            f"Invalid sort_by: {sort_by}. Valid values: {', '.join(valid_sort_fields)}",
        )]

    store = get_store()
    limit = validate_limit(limit, default=50)

    summaries = store.list_all()

    # Filter by tags (normalized to stored form: lowercase, hyphenated)
    if tags:
        tags_norm = [t for t in (normalize_tag(t) for t in tags) if t]
        summaries = [
            s for s in summaries
            if all(t in s.tags for t in tags_norm)
        ]

    # Filter by category (exact match or child categories), normalized to the
    # stored slug form so a caller passing "Work"/"Finance" matches the stored
    # "work"/"finance" instead of silently returning nothing. "work" matches
    # "work" and "work/projects" but NOT "work-related".
    category_norm = slugify_category_path(category) if category else None
    if category_norm:
        summaries = [
            s for s in summaries
            if s.category and (
                s.category == category_norm or
                s.category.startswith(category_norm + "/")
            )
        ]

    # Sort
    if sort_by == "modified":
        summaries.sort(key=lambda s: s.modified, reverse=True)
    elif sort_by == "created":
        summaries.sort(key=lambda s: s.created, reverse=True)
    elif sort_by == "title":
        summaries.sort(key=lambda s: s.title.lower())

    # Limit
    summaries = summaries[:limit]

    return [s.model_dump(mode="json") for s in summaries]


@mcp.tool()
async def find_similar_notes(note_id: str, limit: int = 5) -> list[dict]:
    """
    Find notes semantically similar to the given note.

    Args:
        note_id: Source note UUID string
        limit: Max results (default 5, max 100)

    Returns:
        List of similar notes
    """
    search = await get_search()
    limit = validate_limit(limit, default=5)

    try:
        uuid = UUID(note_id)
    except ValueError:
        return [error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")]

    results = await search.find_similar(uuid, limit=limit)
    return [r.model_dump(mode="json") for r in results]
