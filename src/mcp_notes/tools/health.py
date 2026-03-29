"""Health and maintenance operations for notes.

Tools:
- reindex_notes: Force reindex all notes
- check_note_health: Check health of all notes
"""

from typing import Any

from mcp_notes.app import mcp
from mcp_notes.singletons import get_indexer, get_links, get_store


@mcp.tool()
async def reindex_notes() -> dict:
    """
    Force reindex all notes (useful after manual file edits).

    Returns:
        Index status
    """
    indexer = await get_indexer()

    status = await indexer.index_all(force=True)
    return status.model_dump(mode="json")


@mcp.tool()
async def check_note_health() -> dict:
    """
    Check health of all notes, reporting parse errors and issues.

    Returns:
        Health report with total notes, parse errors, and recommendations
    """
    store = get_store()
    links = get_links()

    # Get all notes (this populates parse errors)
    summaries = store.list_all()
    parse_errors = store.get_parse_errors()

    # Get broken links
    broken = links.get_all_broken_links()

    # Get orphans
    orphans = links.get_orphan_notes()

    health: dict[str, Any] = {
        "total_notes": len(summaries),
        "parse_errors": len(parse_errors),
        "broken_links": len(broken),
        "orphan_notes": len(orphans),
        "is_healthy": len(parse_errors) == 0 and len(broken) == 0,
    }

    # Add details if there are issues
    if parse_errors:
        health["parse_error_details"] = [
            {
                "path": str(e.path),
                "error_type": e.error_type,
                "message": e.message,
            }
            for e in parse_errors
        ]

    if broken:
        health["broken_link_details"] = [
            {
                "source_note_id": str(b.source_note_id),
                "source_note_title": b.source_note_title,
                "broken_target_id": str(b.broken_target_id),
            }
            for b in broken[:10]  # Limit to first 10
        ]
        if len(broken) > 10:
            health["broken_link_note"] = f"Showing first 10 of {len(broken)} broken links"

    return health
