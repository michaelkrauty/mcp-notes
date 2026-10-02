"""Note indexer using vector-core."""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from uuid import UUID

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
)
from vector_core import (
    EmbeddingClient,
    QdrantStorage,
    create_hybrid_point_with_key,
    generate_collection_name,
    generate_point_id,
)
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.storage.embedding_fragments import (
    fragment_point,
    is_derived_fragment,
    upsert_fragment_group,
)
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    embedding_collection_lock,
)

from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.indexing.migration import NotesMigration
from mcp_notes.models import IndexStatus
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.parser import ParsedNote, parse_note

logger = logging.getLogger(__name__)

# Codebase ID for GlobalVocabulary registration
NOTES_CODEBASE_ID = "notes"
NOTE_INDEX_VERSION = "complete-spans-v1"


class NoteIndexer:
    """
    Indexes notes into Qdrant for hybrid search.

    Uses vector-core components:
    - EmbeddingClient for dense vectors
    - GlobalVocabulary for cross-codebase sparse vectors (two-pass indexing)
    - QdrantStorage for vector storage
    """

    def __init__(
        self,
        note_store: NoteStore | None = None,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
    ):
        """
        Initialize indexer.

        Args:
            note_store: NoteStore instance (created if not provided)
            storage: QdrantStorage instance (created if not provided)
            embedder: EmbeddingClient instance (created if not provided)
            global_vocab: GlobalVocabulary instance (uses singleton if not provided)
        """
        self.note_store = note_store or NoteStore()
        self.storage = storage or QdrantStorage()
        self.embedder = embedder or EmbeddingClient()
        self._global_vocab = global_vocab  # Use singleton if None
        self._collection_name: str | None = None
        self._operation_generation: ContextVar[tuple[object, CollectionGeneration] | None] = (
            ContextVar("notes_collection_generation", default=None)
        )

    @property
    def global_vocab(self) -> GlobalVocabulary:
        """Get GlobalVocabulary instance.

        Returns the instance passed to __init__, or the singleton after
        _ensure_global_vocab() is called.

        Raises:
            RuntimeError: If accessed before async initialization.
        """
        if self._global_vocab is None:
            raise RuntimeError(
                "GlobalVocabulary not initialized. Call await _ensure_global_vocab() first, "
                "or pass an instance to __init__."
            )
        return self._global_vocab

    async def _ensure_global_vocab(self) -> None:
        """Ensure GlobalVocabulary is initialized using singleton."""
        if self._global_vocab is None:
            self._global_vocab = GlobalVocabulary.get_instance()

    @property
    def collection_name(self) -> str:
        """Get the physical target captured by the current operation."""
        binding = self._operation_generation.get()
        if binding is not None and binding[0] is asyncio.current_task():
            return binding[1].physical_name
        return self.logical_collection_name

    @property
    def logical_collection_name(self) -> str:
        """Get collection name based on notes directory."""
        if self._collection_name is None:
            self._collection_name = settings.collection_name or generate_collection_name(
                str(self.note_store.base_dir),
                prefix=settings.collection_prefix,
            )
        return self._collection_name

    @asynccontextmanager
    async def collection_operation(self) -> AsyncIterator[CollectionGeneration]:
        """Hold the migration lock and pin one physical target for a mutation."""
        from mcp_notes.indexing.migration import ensure_notes_collection  # noqa: PLC0415

        binding = self._operation_generation.get()
        if binding is not None and binding[0] is asyncio.current_task():
            yield binding[1]
            return
        async with embedding_collection_lock(self.storage, self.logical_collection_name):
            generation = await ensure_notes_collection(self, lock_held=True)
            token = self._operation_generation.set((asyncio.current_task(), generation))
            try:
                yield generation
            finally:
                self._operation_generation.reset(token)

    async def ensure_collection(self) -> None:
        """Ensure a compatible generation and its note payload indexes."""
        async with self.collection_operation():
            await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Ensure Qdrant collection exists with payload indexes."""
        if not await self.storage.collection_exists(self.collection_name):
            await self.storage.create_collection(self.collection_name)
            logger.info(f"Created collection: {self.collection_name}")

        # Ensure payload indexes for efficient filtering (idempotent)
        await self.storage.ensure_payload_indexes(
            self.collection_name,
            [
                ("type", PayloadSchemaType.KEYWORD),
                ("note_id", PayloadSchemaType.KEYWORD),
                ("tags", PayloadSchemaType.KEYWORD),
                ("category", PayloadSchemaType.KEYWORD),
            ],
        )

    async def index_all(self, force: bool = False) -> IndexStatus:
        """Index notes into a compatible generation under the collection lock."""
        async with self.collection_operation():
            return await self._index_all(force=force)

    async def _index_all(self, force: bool = False) -> IndexStatus:
        """
        Index all notes using two-pass GlobalVocabulary pattern.

        Pass 1: Collect tokens from all content and register with GlobalVocabulary
        Pass 2: Generate embeddings and sparse vectors, upsert to Qdrant

        Args:
            force: If True, reindex everything. If False, incremental update.

        Returns:
            IndexStatus with results
        """
        await self._ensure_global_vocab()
        await self.ensure_collection()

        # Get existing indexed notes
        if not force:
            indexed = await self._get_indexed_hashes()
        else:
            indexed = {}
            # Shared collections can contain notes from other source roots.
            # Clear only groups whose ownership is known to this source store.
            ownership = NotesMigration(self)
            payloads = await self.storage.scroll_points(
                self.collection_name, payload_fields=["type", "note_id"], max_results=0
            )
            owned = {
                UUID(payload["note_id"])
                for payload in payloads
                if payload.get("type") in {"note", "chunk"}
                and ownership.owns_note(UUID(payload["note_id"]))
            }
            for note_id in owned:
                await self._delete_note_points(note_id)

        pending = {note_id for note_id, note_hash in indexed.items() if not note_hash}

        # Single pass: collect notes to index AND tokens for GlobalVocabulary
        # (Previously iterated twice - once for notes, once for tokens)
        notes_to_index: list[tuple[ParsedNote, str | None]] = []
        tokens_per_doc: list[set[str]] = []
        total_notes = 0

        for parsed, category in self.note_store.iter_all():
            total_notes += 1
            note_hash = self._hash_note(parsed, category)

            # Check if this note needs indexing
            if force or str(parsed.id) not in indexed or indexed[str(parsed.id)] != note_hash:
                notes_to_index.append((parsed, category))

            # Collect tokens for GlobalVocabulary (needs ALL notes for IDF statistics)
            summary = generate_note_summary(parsed)
            tokens_per_doc.append(set(self.global_vocab.tokenize(summary)))

            for chunk in chunk_note(parsed, self.embedder):
                tokens_per_doc.append(set(self.global_vocab.tokenize(chunk.content)))

        # Register this codebase's vocabulary
        self.global_vocab.register_codebase(NOTES_CODEBASE_ID, tokens_per_doc)

        if not notes_to_index:
            logger.info("No notes to index")
            return IndexStatus(
                total_notes=total_notes,
                indexed_notes=total_notes,
                last_indexed=datetime.now(UTC),
                index_healthy=not pending,
            )

        # Pass 2: Index notes with embeddings and sparse vectors
        indexed_count = 0
        for parsed, category in notes_to_index:
            try:
                new_chunk_count = await self._index_note(parsed, category)
                if not force:
                    # Incremental re-index can shrink a note; upsert replaces
                    # same-ID chunks but cannot remove the chunks left over from a
                    # previous, larger version. Prune them. In force mode the
                    # owned groups were cleared, so no orphans are possible.
                    await self._delete_orphan_chunks(parsed.id, new_chunk_count)
                await self._complete_note_index(parsed, category)
                pending.discard(str(parsed.id))
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index note {parsed.id}: {e}")

        logger.info(f"Indexed {indexed_count}/{len(notes_to_index)} notes")

        # Account for notes that were already up to date (skipped this pass) plus
        # the ones we just indexed successfully. A note whose indexing raised is
        # not counted as indexed, and the index is only healthy if every note that
        # needed indexing succeeded.
        previously_indexed = total_notes - len(notes_to_index)
        indexed_notes = previously_indexed + indexed_count
        index_healthy = indexed_count == len(notes_to_index) and not pending
        if not index_healthy:
            logger.warning(
                f"Indexing incomplete: {indexed_count}/{len(notes_to_index)} "
                "notes indexed successfully"
            )

        return IndexStatus(
            total_notes=total_notes,
            indexed_notes=indexed_notes,
            last_indexed=datetime.now(UTC),
            index_healthy=index_healthy,
        )

    async def index_note(self, note_id: UUID) -> None:
        """Index one note into a compatible generation."""
        async with self.collection_operation():
            await self._index_one_note(note_id)

    async def _index_one_note(self, note_id: UUID) -> None:
        """
        Index a single note using upsert-then-delete-orphans pattern.

        Uses incremental vocabulary update for efficiency - only adds tokens
        from this note instead of re-reading the entire corpus.

        Data integrity: Uses upsert-then-delete-orphans to ensure notes remain
        searchable even if indexing partially fails. Old chunks are only deleted
        AFTER new chunks are successfully upserted.

        Args:
            note_id: Note UUID to index
        """
        await self._ensure_global_vocab()
        await self.ensure_collection()

        # Load note (category comes from path via read())
        note = self.note_store.read(note_id)
        parsed = parse_note(note.content)
        category = note.category  # Category from path

        # Collect tokens for incremental vocabulary update
        chunks = chunk_note(parsed, self.embedder)
        summary = generate_note_summary(parsed)

        tokens_for_doc: list[set[str]] = [set(self.global_vocab.tokenize(summary))]
        for chunk in chunks:
            tokens_for_doc.append(set(self.global_vocab.tokenize(chunk.content)))

        # Update vocabulary incrementally (O(1) per note, not O(n))
        # This handles both empty vocabulary and adding to existing vocabulary
        self.global_vocab.update_codebase_incremental(
            NOTES_CODEBASE_ID,
            added_tokens=tokens_for_doc,
            removed_tokens=[],
            net_doc_change=len(tokens_for_doc),
        )

        # Index with category from path - returns new chunk count
        # Note: _index_note uses upsert, so existing chunks with same IDs are replaced
        new_chunk_count = await self._index_note(parsed, category)

        # Delete orphaned chunks (old chunks beyond new chunk count)
        # This is safe because new chunks are already upserted
        await self._delete_orphan_chunks(note_id, new_chunk_count)
        await self._complete_note_index(parsed, category)

    async def delete_note_index(self, note_id: UUID) -> None:
        """
        Remove a note from the index.

        Args:
            note_id: Note UUID to remove
        """
        async with self.collection_operation():
            await self._delete_note_points(note_id)

    async def _index_note(self, parsed: ParsedNote, category: str | None) -> int:
        """Index a single parsed note with category from path.

        Returns:
            Number of chunks indexed (for orphan cleanup)
        """
        # Note: Vocabulary should be updated before calling this method
        # via update_codebase_incremental() in index_note() or register_codebase() in index_all()

        # Generate chunks
        chunks = chunk_note(parsed, self.embedder)

        # Prepare texts for embedding
        texts = [c.content for c in chunks]

        # Add file-level summary
        summary = generate_note_summary(parsed)
        # Get embeddings
        embeddings = await self.embedder.embed_all(texts, role="document")

        # Prepare points
        points = []

        # File-level point - use category from path, not frontmatter
        file_points = await fragment_point(
            self.embedder,
            point_id=generate_point_id(f"note:{parsed.id}"),
            text=summary,
            sparse=self.global_vocab.vectorize_document(summary),
            vectorize=self.global_vocab.vectorize_document,
            payload={
                "type": "note",
                "note_id": str(parsed.id),
                "title": parsed.title,
                "tags": parsed.tags,
                "category": category,  # From path, not frontmatter
                "created": parsed.created.isoformat(),
                "modified": parsed.modified.isoformat(),
                "note_hash": "",
                "index_policy": "",
                "source_reindex_pending": True,
                "embedding_text": summary,
            },
        )

        # Chunk points
        for i, chunk in enumerate(chunks):
            chunk_point = self._create_point(
                point_type="chunk",
                note_id=parsed.id,
                chunk_index=i,
                content=chunk.content,
                embedding=embeddings[i],
                payload={
                    "type": "chunk",
                    "note_id": str(parsed.id),
                    "title": parsed.title,
                    "chunk_index": i,
                    "section_title": chunk.section_title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "content": chunk.content,
                    "tags": parsed.tags,
                    "category": category,  # From path, not frontmatter
                    "created": parsed.created.isoformat(),
                    "modified": parsed.modified.isoformat(),
                },
            )
            points.append(chunk_point)

        # Upsert
        await upsert_fragment_group(self.storage, self.collection_name, file_points)
        for point in points:
            await upsert_fragment_group(self.storage, self.collection_name, [point])
        logger.debug(f"Indexed note {parsed.id} with {len(chunks)} chunks")

        return len(chunks)

    async def _complete_note_index(self, parsed: ParsedNote, category: str | None) -> None:
        """Publish the current source hash only after writes and orphan cleanup succeed."""
        await self.storage.update_payload(
            self.collection_name,
            [
                FieldCondition(key="type", match=MatchValue(value="note")),
                FieldCondition(key="note_id", match=MatchValue(value=str(parsed.id))),
            ],
            {
                "note_hash": self._hash_note(parsed, category),
                "index_policy": self._index_policy(),
                "source_reindex_pending": False,
            },
        )

    def _create_point(
        self,
        point_type: str,
        note_id: UUID,
        chunk_index: int | None,
        content: str,
        embedding: list[float],
        payload: dict,
    ) -> PointStruct:
        """Create a Qdrant point with dense + sparse vectors."""
        # Generate deterministic key for point ID
        if chunk_index is not None:
            key = f"{point_type}:{note_id}:{chunk_index}"
        else:
            key = f"{point_type}:{note_id}"

        # Generate sparse vector using GlobalVocabulary
        sparse = self.global_vocab.vectorize_document(content)

        retained = (
            {"embedding_text_field": "content"}
            if payload.get("content") == content
            else {"embedding_text": content}
        )
        return create_hybrid_point_with_key(key, embedding, sparse, {**payload, **retained})

    async def _delete_note_points(self, note_id: UUID) -> None:
        """Delete all points for a note."""
        await self.storage.delete_by_filter(
            self.collection_name,
            field="note_id",
            value=str(note_id),
        )

    async def _delete_orphan_chunks(self, note_id: UUID, new_chunk_count: int) -> None:
        """Retire obsolete canonical chunks and their derived fragments together."""
        client = await self.storage.get_client()
        await client.delete(
            self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="note_id", match=MatchValue(value=str(note_id))),
                    FieldCondition(key="type", match=MatchValue(value="chunk")),
                    FieldCondition(key="chunk_index", range=Range(gte=new_chunk_count)),
                ]
            ),
            wait=True,
        )

    async def _get_indexed_hashes(self) -> dict[str, str]:
        """Get map of note_id -> hash for indexed notes."""
        try:
            points = await self.storage.scroll_points(
                self.collection_name,
                filter_conditions=[
                    FieldCondition(key="type", match=MatchValue(value="note")),
                ],
                payload_fields=[
                    "note_id",
                    "note_hash",
                    "embedding_text_source",
                    "index_policy",
                    "embedding_fragment",
                    "source_reindex_pending",
                ],
            )

            return {
                p["note_id"]: (
                    ""
                    if (
                        p.get("embedding_text_source") == "legacy-note-metadata"
                        or p.get("source_reindex_pending") is True
                        or p.get("index_policy") != self._index_policy()
                    )
                    else p.get("note_hash", "")
                )
                for p in points
                if p.get("note_id") and not is_derived_fragment(p)
            }
        except Exception as e:
            logger.debug(f"Could not retrieve indexed hashes (collection may not exist): {e}")
            return {}

    def _hash_note(self, parsed: ParsedNote, category: str | None) -> str:
        """Generate hash of note content including category from path."""
        content = (
            f"{NOTE_INDEX_VERSION}:{parsed.title}:{parsed.body}:{','.join(parsed.tags)}:{category}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _index_policy() -> str:
        return f"{NOTE_INDEX_VERSION}:{settings.max_chunk_chars}:{settings.section_overlap_chars}"

    async def get_status(self) -> IndexStatus:
        """Report status for the compatible physical generation."""
        async with self.collection_operation():
            return await self._get_status()

    async def _get_status(self) -> IndexStatus:
        """Get current index status."""
        total_notes = self.note_store.count()

        indexed: dict[str, str] = {}
        try:
            indexed = await self._get_indexed_hashes()
            indexed_notes = sum(bool(note_hash) for note_hash in indexed.values())
        except Exception as e:
            logger.debug(f"Could not get indexed count: {e}")
            indexed_notes = 0

        # Check if collection exists
        try:
            exists = await self.storage.collection_exists(self.collection_name)
        except Exception as e:
            logger.debug(f"Could not check collection existence: {e}")
            exists = False

        return IndexStatus(
            total_notes=total_notes,
            indexed_notes=indexed_notes,
            last_indexed=None,  # Would need to track this
            index_healthy=exists and indexed_notes > 0 and indexed_notes == len(indexed),
        )

    async def close(self) -> None:
        """Close connections safely."""
        if self.storage is not None:
            await self.storage.close()
        if self.embedder is not None:
            await self.embedder.close()
