"""Data models for mcp-notes."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Note(BaseModel):
    """Full note representation."""

    id: UUID
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    links: list[UUID] = Field(default_factory=list)
    created: datetime
    modified: datetime


class NoteSummary(BaseModel):
    """Lightweight note summary for listings."""

    id: UUID
    title: str
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    created: datetime
    modified: datetime
    excerpt: str = ""  # First N chars of content


class SearchResult(BaseModel):
    """Search result with relevance info."""

    note: NoteSummary
    score: float
    highlights: list[str] = Field(default_factory=list)  # Matching snippets
    result_type: str = "note"  # "note", "chunk", or "glossary"
    degraded: bool = False  # True when using sparse-only fallback (embedding service unavailable)


class NoteVersion(BaseModel):
    """Git version information for a note."""

    commit_sha: str
    timestamp: datetime
    message: str
    author: str


class NoteLinks(BaseModel):
    """Link analysis for a note."""

    outgoing: list[NoteSummary] = Field(default_factory=list)
    incoming: list[NoteSummary] = Field(default_factory=list)  # Backlinks
    broken: list[UUID] = Field(default_factory=list)


class TagInfo(BaseModel):
    """Tag with usage count."""

    name: str
    count: int


class CategoryInfo(BaseModel):
    """Category with hierarchy."""

    path: str
    count: int
    children: list["CategoryInfo"] = Field(default_factory=list)


class NotesIndex(BaseModel):
    """Full notes index for MCP resource."""

    notes: list[NoteSummary]
    total: int
    last_indexed: datetime | None = None


class IndexStatus(BaseModel):
    """Indexing status information."""

    total_notes: int
    indexed_notes: int
    last_indexed: datetime | None = None
    index_healthy: bool


class BrokenLink(BaseModel):
    """Broken link reference."""

    source_note_id: UUID
    source_note_title: str
    broken_target_id: UUID


class NoteChunk(BaseModel):
    """Chunk of a note for indexing."""

    note_id: UUID
    chunk_index: int
    content: str
    section_title: str | None = None
    start_line: int
    end_line: int


class CategoryTree(BaseModel):
    """Hierarchical category structure."""

    categories: list[CategoryInfo]
    total_notes: int
