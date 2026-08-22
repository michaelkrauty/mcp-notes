"""MCPServer application instance for mcp-notes.

This module creates the shared MCPServer instance that all tool modules import.
Separating this avoids circular imports when modularizing tools.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from mcp_notes import __version__

NOTE_RESOURCE_URIS = (
    "notes://broken-links",
    "notes://categories",
    "notes://index",
    "notes://orphans",
    "notes://parse-errors",
    "notes://recent",
    "notes://tags",
)


async def notify_note_resources(context: Context | None) -> None:
    """Notify modern subscribers after a note mutation."""
    if context is None:
        return

    for uri in NOTE_RESOURCE_URIS:
        await context.notify_resource_updated(uri)


@asynccontextmanager
async def lifespan(_: MCPServer[None]) -> AsyncIterator[None]:
    """Run startup and cleanup on the server's event loop."""
    # Lazy imports avoid the app -> singletons/tools -> app registration cycle.
    from mcp_notes.__main__ import startup  # noqa: PLC0415
    from mcp_notes.singletons import cleanup_async_resources  # noqa: PLC0415

    try:
        await startup()
        yield None
    finally:
        with anyio.CancelScope(shield=True):
            await cleanup_async_resources()


# Initialize the MCP server shared across all tool modules.
mcp: MCPServer[None] = MCPServer("mcp-notes", version=__version__, lifespan=lifespan)
