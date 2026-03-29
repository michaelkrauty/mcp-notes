"""Tag management operations for notes.

Tools:
- list_tags: List all tags with note counts
- rename_tag: Rename a tag across all notes
- merge_tags: Merge multiple tags into one
"""

from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.constants import validate_tag as _validate_tag
from mcp_notes.models import TagInfo
from mcp_notes.singletons import get_git, get_indexer, get_store


@mcp.tool()
async def list_tags() -> list[dict]:
    """
    List all tags with note counts.

    Returns:
        List of tag info
    """
    store = get_store()

    tag_counts: dict[str, int] = {}
    for summary in store.list_all():
        for tag in summary.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    tags = [
        TagInfo(name=name, count=count)
        for name, count in sorted(tag_counts.items())
    ]

    return [t.model_dump(mode="json") for t in tags]


@mcp.tool()
async def rename_tag(old_tag: str, new_tag: str) -> dict:
    """
    Rename a tag across all notes.

    Args:
        old_tag: Tag to rename
        new_tag: New tag name

    Returns:
        Count of notes updated, or error dict if validation fails
    """
    # Validate old tag (just normalize)
    old_normalized = old_tag.lower().strip()
    if not old_normalized:
        return error_response(ErrorCode.VALIDATION_FAILED, "Old tag cannot be empty")

    # Validate new tag
    new_normalized, error = _validate_tag(new_tag)
    if error:
        return error_response(ErrorCode.VALIDATION_FAILED, f"Invalid new tag: {error}")

    store = get_store()
    git = get_git()
    indexer = await get_indexer()

    updated = 0
    for summary in store.list_all():
        if old_normalized in summary.tags:
            # Update tags
            new_tags = [new_normalized if t == old_normalized else t for t in summary.tags]

            store.update(
                note_id=summary.id,
                tags=new_tags,
            )

            note_path = store.get_note_path(summary.id)
            git.commit_update(summary.id, summary.title, path=note_path)
            updated += 1

    # Re-index all
    if updated > 0:
        await indexer.index_all()

    return {"updated_count": updated}


@mcp.tool()
async def merge_tags(source_tags: list[str], target_tag: str) -> dict:
    """
    Merge multiple tags into one.

    Args:
        source_tags: Tags to merge from
        target_tag: Tag to merge into

    Returns:
        Count of notes updated, or error dict if validation fails
    """
    # Validate target tag
    target_normalized, error = _validate_tag(target_tag)
    if error:
        return error_response(ErrorCode.VALIDATION_FAILED, f"Invalid target tag: {error}")

    # Normalize source tags
    source_normalized = [t.lower().strip() for t in source_tags if t.strip()]
    if not source_normalized:
        return error_response(ErrorCode.VALIDATION_FAILED, "Source tags list cannot be empty")

    store = get_store()
    git = get_git()
    indexer = await get_indexer()

    updated = 0
    for summary in store.list_all():
        has_source = any(t in source_normalized for t in summary.tags)
        if has_source:
            # Remove source tags, add target
            new_tags = [t for t in summary.tags if t not in source_normalized]
            if target_normalized not in new_tags:
                new_tags.append(target_normalized)

            store.update(
                note_id=summary.id,
                tags=new_tags,
            )

            note_path = store.get_note_path(summary.id)
            git.commit_update(summary.id, summary.title, path=note_path)
            updated += 1

    # Re-index all
    if updated > 0:
        await indexer.index_all()

    return {"updated_count": updated}
