"""MCP Notes Server - Semantic note management with vector search."""

import json
import logging
from datetime import UTC, datetime

from vector_core import (
    UNSET,
    UnsetType,
    validate_limit,
)
from vector_core.errors import ErrorCode, error_response
from vector_core.glossary import (
    GlossaryNotFoundError,
    TermExistsError,
)

from mcp_notes.constants import validate_tag as _validate_tag
from mcp_notes.facts import (
    DuplicateFactError,
    FactNotFoundError,
    FactSource,
    SourceType,
)
from mcp_notes.models import (
    CategoryInfo,
    CategoryTree,
    NotesIndex,
    TagInfo,
)
from mcp_notes.settings import settings
from mcp_notes.singletons import (
    cleanup_async_resources,
    get_fact_indexer,
    get_fact_store,
    get_git,
    get_glossary_indexer,
    get_glossary_store,
    get_indexer,
    get_integrity_manager,
    get_links,
    get_note_service,
    get_search,
    get_store,
)
from mcp_notes.storage.filesystem import NoteNotFoundError

logger = logging.getLogger(__name__)

# Import mcp instance from app module (shared across tool modules)
from mcp_notes.app import mcp  # noqa: E402

# Import tool modules to register tools with mcp instance
# Note: tools/notes.py contains core note CRUD operations
from mcp_notes import tools  # noqa: E402, F401

# Re-export tools for backward compatibility with tests
from mcp_notes.tools.notes import (  # noqa: E402, F401
    create_note,
    read_note,
    update_note,
    delete_note,
)
from mcp_notes.tools.search import (  # noqa: E402, F401
    search_notes,
    list_notes,
    find_similar_notes,
)
from mcp_notes.tools.versioning import (  # noqa: E402, F401
    get_note_history,
    restore_note_version,
)
from mcp_notes.tools.links import get_note_links  # noqa: E402, F401
from mcp_notes.tools.tags import (  # noqa: E402, F401
    list_tags,
    rename_tag,
    merge_tags,
)
from mcp_notes.tools.categories import (  # noqa: E402, F401
    list_categories,
    move_category,
)
from mcp_notes.tools.health import (  # noqa: E402, F401
    reindex_notes,
    check_note_health,
)
from mcp_notes.tools.glossary import (  # noqa: E402, F401
    add_glossary_entry,
    lookup_term,
    search_glossary,
    list_glossary,
    update_glossary_entry,
    delete_glossary_entry,
)
from mcp_notes.tools.facts import (  # noqa: E402, F401
    add_fact,
    add_facts_batch,
    update_fact,
    delete_fact,
    query_facts,
    get_entity,
    list_facts,
    search_facts,
    index_facts,
    find_connections,
    get_neighbors,
)
from mcp_notes.tools.integrity import (  # noqa: E402, F401
    get_facts_with_stale_sources,
    get_source_statistics,
    check_fact_integrity,
    revalidate_fact_sources,
)


# ============= MCP Resources =============


@mcp.resource("notes://index")
async def get_notes_index() -> str:
    """Full index of all notes."""
    store = get_store()

    summaries = store.list_all()
    index = NotesIndex(
        notes=summaries,
        total=len(summaries),
        last_indexed=datetime.now(UTC),
    )

    return index.model_dump_json()


@mcp.resource("notes://tags")
async def get_tags_resource() -> str:
    """All tags with counts."""
    store = get_store()

    tag_counts: dict[str, int] = {}
    for summary in store.list_all():
        for tag in summary.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    tags = [
        TagInfo(name=name, count=count)
        for name, count in sorted(tag_counts.items())
    ]

    return json.dumps([t.model_dump(mode="json") for t in tags])


@mcp.resource("notes://categories")
async def get_categories_resource() -> str:
    """Category hierarchy with counts."""
    store = get_store()

    category_counts: dict[str, int] = {}
    for summary in store.list_all():
        if summary.category:
            category_counts[summary.category] = category_counts.get(summary.category, 0) + 1

    categories = [
        CategoryInfo(path=path, count=count, children=[])
        for path, count in sorted(category_counts.items())
    ]

    tree = CategoryTree(categories=categories, total_notes=store.count())
    return tree.model_dump_json()


@mcp.resource("notes://recent")
async def get_recent_notes() -> str:
    """Recently modified notes (last 20)."""
    store = get_store()

    summaries = store.list_all()
    summaries.sort(key=lambda s: s.modified, reverse=True)
    recent = summaries[:20]

    return json.dumps([s.model_dump(mode="json") for s in recent])


@mcp.resource("notes://orphans")
async def get_orphan_notes() -> str:
    """Notes with no incoming links."""
    links = get_links()

    orphans = links.get_orphan_notes()

    return json.dumps([o.model_dump(mode="json") for o in orphans])


@mcp.resource("notes://broken-links")
async def get_broken_links_resource() -> str:
    """All broken [[uuid]] references."""
    links = get_links()

    broken = links.get_all_broken_links()

    return json.dumps([b.model_dump(mode="json") for b in broken])


@mcp.resource("notes://parse-errors")
async def get_parse_errors_resource() -> str:
    """Notes that failed to parse (corrupted or malformed)."""
    store = get_store()

    # Trigger a listing to detect parse errors
    store.list_all()

    errors = store.get_parse_errors()

    return json.dumps([
        {
            "path": str(e.path),
            "error_type": e.error_type,
            "message": e.message,
        }
        for e in errors
    ])
