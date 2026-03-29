"""FastMCP application instance for mcp-notes.

This module creates the shared FastMCP instance that all tool modules import.
Separating this avoids circular imports when modularizing tools.
"""

from mcp.server.fastmcp import FastMCP

from mcp_notes import __version__

# Initialize FastMCP server - shared across all tool modules
mcp = FastMCP("mcp-notes")
mcp._mcp_server.version = __version__
