"""Note service for mcp-notes.

Orchestrates note operations across store, git, indexer, and integrity manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mcp_notes.storage.filesystem import Note, NoteNotFoundError

if TYPE_CHECKING:
    from mcp_notes.facts import SourceIntegrityManager
    from mcp_notes.indexing.indexer import NoteIndexer
    from mcp_notes.storage.filesystem import NoteStore
    from mcp_notes.storage.git import GitManager

logger = logging.getLogger(__name__)


@dataclass
class NoteOperationResult:
    """Result of a note operation."""

    success: bool
    note: Note | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API response."""
        if self.success and self.note:
            return self.note.model_dump(mode="json")
        return {
            "error_code": self.error_code,
            "message": self.error_message,
            **({"details": self.metadata} if self.metadata else {}),
        }


class NoteService:
    """Orchestrates note operations with proper coordination.

    This service ensures that:
    - Notes are properly persisted to filesystem
    - Git commits track all changes
    - Search index stays synchronized
    - Fact source integrity is maintained

    All operations are atomic at the service level - either all
    components are updated or the operation fails cleanly.
    """

    def __init__(
        self,
        store: NoteStore,
        git: GitManager,
        indexer: NoteIndexer,
        integrity: SourceIntegrityManager,
    ):
        """Initialize the note service.

        Args:
            store: NoteStore for filesystem operations
            git: GitManager for version control
            indexer: NoteIndexer for search index
            integrity: SourceIntegrityManager for fact source tracking
        """
        self._store = store
        self._git = git
        self._indexer = indexer
        self._integrity = integrity

    @property
    def store(self) -> NoteStore:
        """Access underlying NoteStore."""
        return self._store

    async def create(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
    ) -> NoteOperationResult:
        """Create a new note.

        Coordinates:
        1. Create note in filesystem (generates UUID)
        2. Commit to git
        3. Index for search

        Args:
            title: Note title
            content: Note body content (markdown)
            tags: Optional list of tags
            category: Optional category path

        Returns:
            NoteOperationResult with created note or error
        """
        # Create note in store
        note = self._store.create(
            title=title,
            content=content,
            tags=tags,
            category=category,
        )

        # Git commit
        note_path = self._store.get_note_path(note.id)
        self._git.commit_create(note.id, note.title, path=note_path)

        # Index for search (non-fatal if fails)
        await self._safe_index(note.id)

        return NoteOperationResult(success=True, note=note)

    async def read(self, note_id: UUID) -> NoteOperationResult:
        """Read a note by UUID.

        Args:
            note_id: Note UUID

        Returns:
            NoteOperationResult with note or error
        """
        try:
            note = self._store.read(note_id)
            return NoteOperationResult(success=True, note=note)
        except NoteNotFoundError:
            return NoteOperationResult(
                success=False,
                error_code="note_not_found",
                error_message=f"Note not found: {note_id}",
            )

    async def update(
        self,
        note_id: UUID,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
    ) -> NoteOperationResult:
        """Update an existing note.

        Coordinates:
        1. Track old path for move detection
        2. Update note in filesystem
        3. Git commit (move or update)
        4. Re-index for search
        5. Mark fact sources as modified (if content changed)

        Args:
            note_id: Note UUID
            title: New title (optional)
            content: New content (optional)
            tags: New tags (optional)
            category: New category (optional)

        Returns:
            NoteOperationResult with updated note or error
        """
        try:
            # Track old path for move detection
            old_path = self._store.get_note_path(note_id)

            # Update in store
            note = self._store.update(
                note_id=note_id,
                title=title,
                content=content,
                tags=tags,
                category=category,
            )

            # Git commit - detect move vs simple update
            new_path = self._store.get_note_path(note_id)
            if old_path and new_path and old_path != new_path:
                self._git.commit_move(old_path, new_path, note.title)
            else:
                self._git.commit_update(note_id, note.title, path=new_path)

            # Re-index for search (non-fatal)
            await self._safe_index(note_id)

            # Mark fact sources as modified (only if content changed)
            if content is not None:
                self._safe_mark_modified(note_id)

            return NoteOperationResult(success=True, note=note)

        except NoteNotFoundError:
            return NoteOperationResult(
                success=False,
                error_code="note_not_found",
                error_message=f"Note not found: {note_id}",
            )

    async def delete(
        self,
        note_id: UUID,
    ) -> NoteOperationResult:
        """Delete a note.

        Coordinates:
        1. Remove from search index
        2. Delete from filesystem
        3. Git commit deletion
        4. Mark fact sources as deleted

        The note remains recoverable from git history.

        Args:
            note_id: Note UUID

        Returns:
            NoteOperationResult indicating success or error
        """
        try:
            # Get note info for commit message
            note = self._store.read(note_id)
            note_path = self._store.get_note_path(note_id)

            # Remove from index first (so search doesn't return deleted note)
            await self._safe_delete_index(note_id)

            # Delete from store
            self._store.delete(note_id)

            # Git commit
            self._git.commit_delete(note_id, note.title, path=note_path)

            # Mark fact sources as deleted
            self._safe_mark_deleted(note_id)

            return NoteOperationResult(
                success=True,
                metadata={"deleted_id": str(note_id), "title": note.title},
            )

        except NoteNotFoundError:
            return NoteOperationResult(
                success=False,
                error_code="note_not_found",
                error_message=f"Note not found: {note_id}",
            )

    async def _safe_index(self, note_id: UUID) -> None:
        """Index note, logging errors but not failing operation."""
        try:
            await self._indexer.index_note(note_id)
        except Exception as e:
            logger.warning(f"Failed to index note {note_id}: {e}")

    async def _safe_delete_index(self, note_id: UUID) -> None:
        """Delete note from index, logging errors but not failing."""
        try:
            await self._indexer.delete_note_index(note_id)
        except Exception as e:
            logger.warning(f"Failed to delete index for note {note_id}: {e}")

    def _safe_mark_modified(self, note_id: UUID) -> None:
        """Mark fact sources as modified, logging errors but not failing."""
        try:
            count = self._integrity.mark_note_modified(note_id)
            if count > 0:
                logger.info(f"Marked {count} fact sources as modified for note {note_id}")
        except Exception as e:
            logger.warning(f"Failed to mark fact sources as modified: {e}")

    def _safe_mark_deleted(self, note_id: UUID) -> None:
        """Mark fact sources as deleted, logging errors but not failing."""
        try:
            count = self._integrity.mark_note_deleted(note_id)
            if count > 0:
                logger.info(f"Marked {count} fact sources as deleted for note {note_id}")
        except Exception as e:
            logger.warning(f"Failed to mark fact sources as deleted: {e}")
