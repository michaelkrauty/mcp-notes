"""Source-backed recovery of legacy note vectors into isolated generations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    EmbeddingMigrationError,
    ensure_embedding_collection,
    resolve_shared_embedding_text,
)

from mcp_notes.storage.parser import ParsedNote, parse_note

if TYPE_CHECKING:
    from mcp_notes.indexing.indexer import NoteIndexer


class NotesMigration:
    """Replace legacy note groups from a complete, readable source snapshot.

    Legacy summaries do not retain their embedding input, and chunk payloads
    may be truncated. Replacing the whole group also reconciles external edits
    and deletions without attaching fresh vectors to stale payloads.
    """

    def __init__(self, indexer: NoteIndexer):
        self.indexer = indexer
        self._rebuild_ids: set[UUID] = set()
        self._notes: dict[UUID, tuple[ParsedNote, str | None]] | None = None

    def _source_notes(self) -> dict[UUID, tuple[ParsedNote, str | None]]:
        # Each ensure_notes_collection call owns a fresh coordinator, including
        # retries. Read once per attempt, never once per retained point.
        if self._notes is None:
            self._notes = self._snapshot()
        return self._notes

    async def resolve_text(self, payload: dict[str, Any]) -> str | None:
        if payload.get("type") == "note":
            note_id = UUID(payload["note_id"])
            # Shared collections can contain notes owned by another directory.
            # Absence from this store is not evidence that those notes were deleted.
            path = self.indexer.note_store.get_note_path(note_id)
            if path is None or any(part.is_symlink() for part in (path, *path.parents)):
                return await resolve_shared_embedding_text(payload)
            source = self._source_notes().get(note_id)
            if source is None or self.indexer._hash_note(*source) != payload.get("note_hash"):
                self._rebuild_ids.add(note_id)
                return None
        return await resolve_shared_embedding_text(payload)

    def _snapshot(self) -> dict[UUID, tuple[ParsedNote, str | None]]:
        root = self.indexer.note_store.notes_dir
        # A missing/unreadable root is not evidence that every note was deleted.
        if not root.is_dir():
            raise EmbeddingMigrationError("The notes source directory is unavailable")

        def on_error(error: OSError) -> None:
            raise error

        notes: dict[UUID, tuple[ParsedNote, str | None]] = {}
        for directory, directories, files in os.walk(root, onerror=on_error):
            base = Path(directory)
            directories[:] = [name for name in directories if not (base / name).is_symlink()]
            for name in files:
                if not name.endswith(".md") or (base / name).is_symlink():
                    continue
                path = base / name
                parsed = parse_note(path.read_text(encoding="utf-8"))
                if parsed.id in notes:
                    raise EmbeddingMigrationError("Duplicate note identifiers in source directory")
                category = str(base.relative_to(root))
                notes[parsed.id] = (parsed, None if category == "." else category)
        return notes

    async def finalize(self, physical_name: str) -> None:
        if not self._rebuild_ids:
            return
        # Complete the scan before changing candidate points. Unlike iter_all,
        # this scan propagates traversal and parsing errors instead of skipping.
        notes = self._source_notes()
        await self.indexer._ensure_global_vocab()
        # Use existing stable token IDs without changing live corpus statistics.
        # A failed candidate must not rewrite the active generation's vocabulary.

        # This temporary indexer only writes the candidate and does not resolve
        # a generation or own the shared clients.
        from mcp_notes.indexing.indexer import NoteIndexer  # noqa: PLC0415

        candidate = NoteIndexer(
            note_store=self.indexer.note_store,
            storage=self.indexer.storage,
            embedder=self.indexer.embedder,
            global_vocab=self.indexer.global_vocab,
        )
        candidate._collection_name = physical_name
        for note_id in self._rebuild_ids:
            await candidate._delete_note_points(note_id)
            if note_id in notes:
                await candidate._index_note(*notes[note_id])
                # New source tokens have not joined the live vocabulary yet.
                # Keep this note eligible for the ordinary full index pass,
                # which registers the source corpus before rewriting sparse data.
                await candidate.storage.update_payload(
                    physical_name,
                    [
                        FieldCondition(key="type", match=MatchValue(value="note")),
                        FieldCondition(key="note_id", match=MatchValue(value=str(note_id))),
                    ],
                    {"note_hash": "", "source_reindex_pending": True},
                )
        self._rebuild_ids.clear()


async def ensure_notes_collection(
    indexer: NoteIndexer, *, lock_held: bool = False
) -> CollectionGeneration:
    """Make the complete mixed collection ready before an operation."""
    migration = NotesMigration(indexer)
    return await ensure_embedding_collection(
        indexer.storage,
        indexer.logical_collection_name,
        indexer.embedder,
        migration.resolve_text,
        finalize_candidate=migration.finalize,
        lock_held=lock_held,
    )
