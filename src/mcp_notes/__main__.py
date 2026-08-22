"""Entry point for mcp-notes server."""

import logging

from vector_core import verify_tools_registered

from mcp_notes.server import get_indexer, mcp
from mcp_notes.settings import settings

logger = logging.getLogger(__name__)

# Expected tools for verification (catches silent import failures)
EXPECTED_TOOLS = [
    # Notes CRUD
    "create_note",
    "read_note",
    "update_note",
    "delete_note",
    # Search
    "search_notes",
    "list_notes",
    "find_similar_notes",
    # Versioning
    "get_note_history",
    "restore_note_version",
    # Links
    "get_note_links",
    # Tags
    "list_tags",
    "rename_tag",
    "merge_tags",
    # Categories
    "list_categories",
    "move_category",
    # Health
    "reindex_notes",
    "check_note_health",
    # Glossary
    "add_glossary_entry",
    "lookup_term",
    "search_glossary",
    "list_glossary",
    "update_glossary_entry",
    "delete_glossary_entry",
    # Facts
    "add_fact",
    "add_facts_batch",
    "update_fact",
    "delete_fact",
    "query_facts",
    "get_entity",
    "list_facts",
    "search_facts",
    "index_facts",
    "find_connections",
    "get_neighbors",
    # Integrity
    "get_facts_with_stale_sources",
    "get_source_statistics",
    "check_fact_integrity",
    "revalidate_fact_sources",
]


async def startup() -> None:
    """Run startup tasks."""
    # Auto-index on startup if enabled
    if settings.auto_index:
        logger.info("Auto-indexing notes on startup...")
        try:
            indexer = await get_indexer()
            status = await indexer.index_all()
            logger.info(f"Indexed {status.indexed_notes}/{status.total_notes} notes")
        except Exception as e:
            logger.warning(f"Auto-index failed: {e}")


def main() -> None:
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting mcp-notes server")
    logger.info(f"Notes directory: {settings.dir}")
    logger.info(f"Git enabled: {settings.git_enabled}")
    logger.info(f"Auto-index: {settings.auto_index}")

    # Verify all expected tools are registered (catches silent import failures)
    verify_tools_registered(mcp, EXPECTED_TOOLS, "mcp-notes")

    # The MCPServer lifespan runs startup and cleanup on the serving event loop.
    mcp.run()


if __name__ == "__main__":
    main()
