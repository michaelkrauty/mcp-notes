"""Category management operations for notes.

Tools:
- list_categories: List all categories with note counts and hierarchy
- move_category: Move/rename a category
"""

from mcp_notes.app import mcp
from mcp_notes.models import CategoryInfo, CategoryTree
from mcp_notes.singletons import get_git, get_indexer, get_store
from mcp_notes.storage.slugify import slugify_category_path


@mcp.tool()
async def list_categories() -> dict:
    """
    List all categories with note counts and hierarchy.

    Returns:
        Category tree
    """
    store = get_store()

    category_counts: dict[str, int] = {}
    for summary in store.list_all():
        if summary.category:
            category_counts[summary.category] = category_counts.get(summary.category, 0) + 1

    # Build flat list
    categories = [
        CategoryInfo(path=path, count=count, children=[])
        for path, count in sorted(category_counts.items())
    ]

    return CategoryTree(
        categories=categories,
        total_notes=store.count(),
    ).model_dump(mode="json")


@mcp.tool()
async def move_category(old_path: str, new_path: str) -> dict:
    """
    Move/rename a category.

    Args:
        old_path: Current category path
        new_path: New category path

    Returns:
        Count of notes updated
    """
    store = get_store()
    git = get_git()
    indexer = await get_indexer()

    # Normalize to the stored slug form: categories are slugified on write, so
    # a raw "Work"/"Work & Projects" old_path must be normalized the same way
    # or it matches nothing and the move silently does nothing. new_path is
    # normalized too so the rewritten prefix is deterministic (store.update
    # re-slugifies on write regardless).
    old_norm = slugify_category_path(old_path)
    new_norm = slugify_category_path(new_path)

    updated = 0
    for summary in store.list_all():
        if summary.category and old_norm:
            # Match exact category OR category that starts with old_norm/
            # This prevents "work" from matching "work-related"
            is_exact_match = summary.category == old_norm
            is_child_match = summary.category.startswith(old_norm + "/")

            if is_exact_match or is_child_match:
                # Replace prefix (preserve the child suffix)
                new_category = new_norm + summary.category[len(old_norm):]

                store.update(
                    note_id=summary.id,
                    category=new_category,
                )

                note_path = store.get_note_path(summary.id)
                git.commit_update(summary.id, summary.title, path=note_path)
                updated += 1

    # Re-index all
    if updated > 0:
        await indexer.index_all()

    return {"updated_count": updated}
