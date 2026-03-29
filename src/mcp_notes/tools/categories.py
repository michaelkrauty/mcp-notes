"""Category management operations for notes.

Tools:
- list_categories: List all categories with note counts and hierarchy
- move_category: Move/rename a category
"""

from mcp_notes.app import mcp
from mcp_notes.models import CategoryInfo, CategoryTree
from mcp_notes.singletons import get_git, get_indexer, get_store


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

    updated = 0
    for summary in store.list_all():
        if summary.category:
            # Match exact category OR category that starts with old_path/
            # This prevents "work" from matching "work-related"
            is_exact_match = summary.category == old_path
            is_child_match = summary.category.startswith(old_path + "/")

            if is_exact_match or is_child_match:
                # Replace prefix
                new_category = new_path + summary.category[len(old_path):]

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
