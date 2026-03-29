"""Note storage operations."""

from mcp_notes.storage.filesystem import NoteNotFoundError, NoteStore
from mcp_notes.storage.git import GitManager
from mcp_notes.storage.parser import ParsedNote, parse_note, serialize_note
from mcp_notes.storage.slugify import build_filename, generate_slug, slugify_category_path
from mcp_notes.storage.uuid_index import UUIDIndex

__all__ = [
    "parse_note",
    "serialize_note",
    "ParsedNote",
    "NoteStore",
    "NoteNotFoundError",
    "GitManager",
    "UUIDIndex",
    "generate_slug",
    "build_filename",
    "slugify_category_path",
]
