"""Configuration for mcp-notes via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from vector_core.settings import VectorCoreSettingsMixin, settings as vector_settings


class NotesSettings(VectorCoreSettingsMixin, BaseSettings):
    """Notes-specific settings.

    Inherits vector-core settings (embedding_url, qdrant_url, etc.) via mixin.
    """

    model_config = SettingsConfigDict(env_prefix="NOTES_")

    # Storage
    dir: Path = Path.home() / "notes"

    # Git versioning
    git_enabled: bool = True
    git_user_name: str = "Notes MCP"
    git_user_email: str = "notes@localhost"

    # Search defaults
    search_limit_default: int = 10
    excerpt_length: int = 200

    # Indexing
    auto_index: bool = True  # Index on startup

    # Tag validation
    max_tag_length: int = 50  # Maximum characters allowed in a tag

    # Chunking thresholds
    max_chunk_chars: int = 80000  # ~20k tokens
    section_overlap_chars: int = 500  # Overlap between sections

    # File size limits (protection against memory exhaustion)
    max_note_size_kb: int = 10240  # 10 MB default limit

    # Collection name prefix for Qdrant
    collection_prefix: str = "notes"

    # Derived paths (not in vector-core mixin)
    @property
    def facts_db_path(self) -> Path:
        """Path to facts database (shared across MCP servers for verification)."""
        return vector_settings.shared_data_dir / "facts.db"


settings = NotesSettings()
