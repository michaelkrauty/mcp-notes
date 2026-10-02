"""Source-backed recovery of legacy note vectors into isolated generations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from qdrant_client.models import FieldCondition, MatchValue
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    EmbeddingMigrationError,
    ensure_embedding_collection,
    resolve_shared_embedding_text,
)
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.storage.parser import ParsedNote, parse_note

if TYPE_CHECKING:
    from mcp_notes.indexing.indexer import NoteIndexer


class NotesMigration:
    """Replace legacy note groups from a complete, readable source snapshot.

    Legacy summaries do not retain their embedding input, and chunk payloads
    may be truncated. Replacing readable source groups reconciles external edits
    without attaching fresh vectors to stale payloads. Missing files are not
    deletion evidence; complete retained inputs remain authoritative.
    """

    def __init__(self, indexer: NoteIndexer):
        self.indexer = indexer
        self._rebuild_ids: set[UUID] = set()
        self._notes: dict[UUID, tuple[ParsedNote, str | None]] | None = None
        self._chunk_texts: dict[UUID, list[str]] = {}

    def _source_notes(self) -> dict[UUID, tuple[ParsedNote, str | None]]:
        # Each ensure_notes_collection call owns a fresh coordinator, including
        # retries. Read once per attempt, never once per retained point.
        if self._notes is None:
            self._notes = self._snapshot()
        return self._notes

    def owns_note(self, note_id: UUID) -> bool:
        path = self.indexer.note_store.get_note_path(note_id)
        root = self.indexer.note_store.notes_dir
        if path is None or not path.is_relative_to(root):
            return False
        # The configured root may itself be a symlink. Only entries beneath
        # that root are excluded, matching the source snapshot's traversal.
        return not any(
            part.is_symlink()
            for part in (path, *path.parents)
            if part != root and part.is_relative_to(root)
        )

    async def resolve_text(self, payload: dict[str, Any]) -> str | None:
        point_type = payload.get("type")
        if point_type in {"note", "chunk"}:
            note_id = UUID(payload["note_id"])
            # Shared collections can contain notes owned by another directory.
            # Absence from this store is not evidence that those notes were deleted.
            if self.owns_note(note_id):
                return self._resolve_owned_note(note_id, payload)
        return await resolve_shared_embedding_text(payload)

    def _resolve_owned_note(self, note_id: UUID, payload: dict[str, Any]) -> str | None:
        path = self.indexer.note_store.get_note_path(note_id)
        source = self._source_notes().get(note_id) if path is not None and path.is_file() else None
        if source is None:
            return self._retained_note_text(payload)
        if payload["type"] == "note":
            if (
                self.indexer._hash_note(*source) != payload.get("note_hash")
                or payload.get("index_policy") != self.indexer._index_policy()
            ):
                self._rebuild_ids.add(note_id)
                return None
            return generate_note_summary(source[0])
        if note_id not in self._chunk_texts:
            self._chunk_texts[note_id] = [
                chunk.content for chunk in chunk_note(source[0], self.indexer.embedder)
            ]
        texts = self._chunk_texts[note_id]
        index = payload.get("chunk_index")
        if (
            isinstance(index, int)
            and 0 <= index < len(texts)
            and texts[index] == payload.get("embedding_text", payload.get("content"))
        ):
            return texts[index]
        self._rebuild_ids.add(note_id)
        return None

    @staticmethod
    def _retained_note_text(payload: dict[str, Any]) -> str:
        text = stored_embedding_text(payload)
        if text is not None and payload.get("embedding_text_source") != "legacy-note-metadata":
            return text
        raise EmbeddingMigrationError(
            "Original note source is unavailable and complete retained embedding input is missing; "
            "legacy metadata-only summaries or potentially truncated chunks "
            "cannot establish coverage"
        )

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
        # Core can copy points with retained embedding_text without invoking
        # our resolver. Reconcile those too, including orphan chunks left by
        # an interrupted index, before publishing the candidate.
        payloads = await self.indexer.storage.scroll_points(physical_name, max_results=0)
        summaries: set[UUID] = set()
        chunks: set[UUID] = set()
        for payload in payloads:
            if is_derived_fragment(payload):
                continue
            if payload.get("type") not in {"note", "chunk"}:
                continue
            note_id = UUID(payload["note_id"])
            if not self.owns_note(note_id):
                continue
            (summaries if payload["type"] == "note" else chunks).add(note_id)
            await self.resolve_text(payload)
        readable_orphans = {
            note_id
            for note_id in chunks - summaries
            if (path := self.indexer.note_store.get_note_path(note_id)) is not None
            and path.is_file()
        }
        if readable_orphans:
            self._rebuild_ids.update(readable_orphans & self._source_notes().keys())
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
            if note_id in notes:
                await candidate._delete_note_points(note_id)
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
    await indexer._ensure_global_vocab()
    return await ensure_embedding_collection(
        indexer.storage,
        indexer.logical_collection_name,
        indexer.embedder,
        migration.resolve_text,
        finalize_candidate=migration.finalize,
        vectorize=indexer.global_vocab.vectorize_document,
        lock_held=lock_held,
    )
