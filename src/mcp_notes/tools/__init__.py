"""MCP Notes tool modules.

Modularized tool implementations for the mcp-notes server.
Each module handles a specific category of tools:

- notes.py: Core note CRUD operations (create, read, update, delete)
- search.py: Search and list operations (search_notes, list_notes, find_similar)
- versioning.py: Version control (get_note_history, restore_note_version)
- links.py: Note linking (get_note_links)
- tags.py: Tag management (list_tags, rename_tag, merge_tags)
- categories.py: Category management (list_categories, move_category)
- glossary.py: Glossary tools (add, lookup, search, list, update, delete)
- facts.py: Fact graph tools (add, query, search, connections, neighbors)
- health.py: Health and maintenance (check_note_health, reindex_notes)

Tools are registered via @mcp.tool() decorator when modules are imported.
Import all modules in server.py to register all tools.
"""

# Import all tool modules to register their tools with mcp
from mcp_notes.tools import notes
from mcp_notes.tools import search
from mcp_notes.tools import versioning
from mcp_notes.tools import links
from mcp_notes.tools import tags
from mcp_notes.tools import categories
from mcp_notes.tools import health
from mcp_notes.tools import glossary
from mcp_notes.tools import facts
from mcp_notes.tools import integrity

__all__ = [
    "notes",
    "search",
    "versioning",
    "links",
    "tags",
    "categories",
    "health",
    "glossary",
    "facts",
    "integrity",
]
