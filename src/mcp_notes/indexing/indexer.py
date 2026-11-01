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
    MatchAny,
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
    FRAGMENT_KEY,
    fragment_point,
    is_derived_fragment,
    upsert_fragment_group,
)
from vector_core.storage.embedding_migration import (
    CollectionGeneration,
    embedding_collection_lock,
)
from vector_core.storage.embedding_sources import stored_embedding_text

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

# Physical names of migrated generations this process published and then
# registered with the vocabulary. Process-wide, because the indexer a search
# creates for one query can be the one that publishes.
_registered_generations: set[str] = set()


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
        """Hold the migration lock and pin one physical target for a mutation.

        A generation this process just published is registered with the
        vocabulary before the operation runs.
        """
        from mcp_notes.indexing.migration import ensure_notes_collection  # noqa: PLC0415

        binding = self._operation_generation.get()
        if binding is not None and binding[0] is asyncio.current_task():
            yield binding[1]
            return
        async with embedding_collection_lock(self.storage, self.logical_collection_name):
            generation = await ensure_notes_collection(self, lock_held=True)
            token = self._operation_generation.set((asyncio.current_task(), generation))
            try:
                await self._register_published_generation(generation)
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

        Notes pass 2 does not finish keep their registration corrected to what
        is actually stored for them.

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
            # Notes with points before pass 2, for vocabulary reconciliation
            previously_stored = set(indexed)
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
            # Only groups owned by another source root are still stored
            previously_stored = {
                payload["note_id"]
                for payload in payloads
                if payload.get("type") in {"note", "chunk"}
                and UUID(payload["note_id"]) not in owned
            }

        pending = {note_id for note_id, note_hash in indexed.items() if not note_hash}

        # Single pass: collect notes to index AND tokens for GlobalVocabulary
        # (Previously iterated twice - once for notes, once for tokens)
        notes_to_index: list[tuple[ParsedNote, str | None, list[set[str]]]] = []
        tokens_per_doc: list[set[str]] = []
        total_notes = 0

        for parsed, category in self.note_store.iter_all():
            total_notes += 1
            note_hash = self._hash_note(parsed, category)

            # Collect tokens for GlobalVocabulary (needs ALL notes for IDF statistics)
            note_tokens = self._note_token_sets(parsed)
            tokens_per_doc.extend(note_tokens)

            # Check if this note needs indexing; keep its registered sets in
            # case pass 2 does not finish it
            if force or str(parsed.id) not in indexed or indexed[str(parsed.id)] != note_hash:
                notes_to_index.append((parsed, category, note_tokens))

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
        started = 0
        finished: set[UUID] = set()
        try:
            for parsed, category, _ in notes_to_index:
                started += 1
                try:
                    new_chunk_count = await self._index_note(parsed, category)
                    if not force:
                        # Incremental re-index can shrink a note; upsert replaces
                        # same-ID chunks but cannot remove the chunks left over from a
                        # previous, larger version. Prune them. In force mode the
                        # owned groups were cleared, so no orphans are possible.
                        await self._delete_orphan_chunks(parsed.id, new_chunk_count)
                    await self._complete_note_index(parsed, category)
                    finished.add(parsed.id)
                    pending.discard(str(parsed.id))
                    indexed_count += 1
                except Exception as e:
                    logger.error(f"Failed to index note {parsed.id}: {e}")
        finally:
            # Runs on cancellation too, best-effort: an interrupted pass leaves
            # the current note part written and later notes unwritten, all
            # registered by pass 1 as if indexed.
            await self._reconcile_unfinished_notes(
                [
                    (parsed.id, registered, position < started)
                    for position, (parsed, _, registered) in enumerate(notes_to_index)
                    if parsed.id not in finished
                ],
                previously_stored,
            )

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

    # ------------------------------------------------------------------
    # Sparse vocabulary accounting
    #
    # Single-note writes keep the notes contribution to GlobalVocabulary
    # current with incremental deltas instead of re-registering the whole
    # corpus on every write; index_all re-registers it from the source notes
    # and then corrects it for any note it failed to write, and a process that
    # publishes a migrated generation registers what that generation stores.
    # Two invariants keep sparse retrieval honest:
    #
    # 1. A note's terms are in the vocabulary before the note is vectorized.
    #    ``vectorize_document`` silently drops unknown tokens, so a term
    #    registered late never reaches the sparse vectors of the note that
    #    introduced it.
    # 2. The contribution describes exactly the indexed notes. Its document
    #    count and document frequencies feed the IDF weights of every codebase
    #    sharing the vocabulary database.
    #
    # As in index_all, each canonical point is one vocabulary document: the
    # summary, whose text ``_index_note`` stores in ``embedding_text``, and
    # each chunk, stored in ``content``. The removal side reads that stored
    # text back, so adding and retiring a note tokenize the same text and
    # repeated edits cannot leak document frequency. Derived fragments are
    # dense slices of the same text, not further documents.
    # ------------------------------------------------------------------

    def _note_token_sets(self, parsed: ParsedNote) -> list[set[str]]:
        """Token sets for the points ``_index_note`` writes: the summary, then each chunk."""
        texts = [generate_note_summary(parsed)]
        texts.extend(chunk.content for chunk in chunk_note(parsed, self.embedder))
        return [set(self.global_vocab.tokenize(text)) for text in texts]

    async def _stored_token_sets(self, *conditions: FieldCondition) -> list[tuple[str, set[str]]]:
        """``(note_id, tokens)`` for each stored canonical note or chunk point that matches.

        A point whose payload predates retained embedding text still counts as
        a document, just one without tokens.
        """
        payloads = await self.storage.scroll_points(
            self.collection_name,
            filter_conditions=[
                *conditions,
                FieldCondition(key="type", match=MatchAny(any=["note", "chunk"])),
            ],
            # note_id keeps the projection non-empty for a point without text;
            # scroll_points drops points whose payload comes back empty.
            payload_fields=[
                "note_id",
                "embedding_text",
                "embedding_text_field",
                "content",
                FRAGMENT_KEY,
            ],
            max_results=0,
        )
        token_sets: list[tuple[str, set[str]]] = []
        for payload in payloads:
            if is_derived_fragment(payload):
                continue
            try:
                text = stored_embedding_text(payload)
            except ValueError:
                text = None
            if text is None:
                text = payload.get("content")
            tokens = set(self.global_vocab.tokenize(text)) if isinstance(text, str) else set()
            token_sets.append((payload["note_id"], tokens))
        return token_sets

    async def _indexed_token_sets(self, note_id: UUID) -> list[set[str]]:
        """Token sets currently registered for a note, one per stored canonical point.

        An unindexed note has no points and contributes nothing.
        """
        stored = await self._stored_token_sets(
            FieldCondition(key="note_id", match=MatchValue(value=str(note_id)))
        )
        return [tokens for _, tokens in stored]

    async def _register_published_generation(self, generation: CollectionGeneration) -> None:
        """Register the notes contribution from a generation this process published.

        A migration rebuilds changed notes from their current source in the
        candidate but leaves the vocabulary alone, so a failed candidate can
        never change it. Once published, the generation is what is stored,
        and the single-note deltas that follow read it back as the registered
        state, so register its canonical points owned by this store, as
        index_all would. vector-core keeps returning the publishing binding,
        so a process-wide memo makes this run once per generation. A failure
        is logged and the next operation retries.
        """
        if not generation.migrated or generation.physical_name in _registered_generations:
            return
        try:
            await self._ensure_global_vocab()
            stored = await self._stored_token_sets()
            ownership = NotesMigration(self)
            owned = {note_id for note_id, _ in stored if ownership.owns_note(UUID(note_id))}
            self.global_vocab.register_codebase(
                NOTES_CODEBASE_ID, [tokens for note_id, tokens in stored if note_id in owned]
            )
        except Exception as e:
            logger.warning(
                f"Could not register the published notes generation "
                f"{generation.physical_name} with the vocabulary; retrying next operation: {e}"
            )
            return
        _registered_generations.add(generation.physical_name)

    def _update_vocabulary(
        self,
        added_tokens: list[set[str]],
        removed_tokens: list[set[str]],
        net_doc_change: int,
    ) -> bool:
        """Apply a notes vocabulary delta, logging rather than raising on failure.

        Returns whether the delta reached the database. Callers must not
        compensate for a delta that was never applied: inverting one that
        failed would subtract tokens and documents that were never added.

        A note write must not fail because the shared vocabulary database was
        busy: a missed delta degrades sparse ranking, while a raised error would
        lose the note. ``index_all`` re-registers the whole corpus and repairs
        any drift.
        """
        try:
            self.global_vocab.update_codebase_incremental(
                NOTES_CODEBASE_ID,
                added_tokens=added_tokens,
                removed_tokens=removed_tokens,
                net_doc_change=net_doc_change,
            )
        except Exception as e:
            logger.warning(f"Failed to update the notes vocabulary contribution: {e}")
            return False
        return True

    async def _revert_vocabulary_update(
        self, note_id: UUID, registered: list[set[str]], previous: list[set[str]]
    ) -> None:
        """Make the contribution describe what a failed note write left stored.

        ``_index_note`` embeds everything before writing, then replaces the
        summary group and each chunk group in turn, and each group write
        restores its previous points when it fails. A failure before the first
        group write commits leaves the previous points intact. A later one
        leaves a mix: the new summary and leading chunks beside the old trailing
        ones, every new point plus orphaned old chunks if cleanup fails, or just
        the new points if publishing the hash fails. Whatever is stored is what
        the next write or delete reads back and retires, so read it back and
        register that. If it cannot be read, assume nothing was written, as is
        the case for a failed embed or a failure at the first Qdrant request.
        """
        try:
            stored = await self._indexed_token_sets(note_id)
        except Exception as e:
            logger.warning(
                f"Could not read note {note_id} back after a failed write; "
                f"restoring its previous vocabulary contribution: {e}"
            )
            stored = previous
        self._update_vocabulary(stored, registered, len(stored) - len(registered))

    async def _reconcile_unfinished_notes(
        self,
        unfinished: list[tuple[UUID, list[set[str]], bool]],
        previously_stored: set[str],
    ) -> None:
        """Make index_all's registration describe what its unfinished notes left stored.

        Pass 1 registers every source note's new token sets so pass 2 can
        vectorize them, but a note whose write raised, or that a cancelled pass
        interrupted or never reached, keeps whatever is stored: its old points,
        nothing, or a partial mix. A later ``index_note`` reads that back and
        retires it, so the contribution must hold the stored sets rather than
        the registered ones. All corrections go out as one delta.

        Each entry is ``(note_id, registered sets, whether pass 2 started it)``;
        ``previously_stored`` holds the IDs of notes with points before pass 2.
        A note pass 2 never started that had no points still has none, so it
        needs no read. A failed read falls back to that for a note that had no
        points. One that had points keeps its registered sets, since its old
        ones are unknown, and the next ``index_all`` repairs it.
        """
        added: list[set[str]] = []
        removed: list[set[str]] = []
        for note_id, registered, started in unfinished:
            had_points = str(note_id) in previously_stored
            stored: list[set[str]] = []
            if started or had_points:
                try:
                    stored = await self._indexed_token_sets(note_id)
                except Exception as e:
                    if had_points:
                        logger.warning(
                            f"Could not read unfinished note {note_id} back after "
                            f"index_all; its vocabulary contribution stays as registered: {e}"
                        )
                        continue
                    logger.warning(
                        f"Could not read unfinished note {note_id} back after "
                        f"index_all; assuming it stored nothing: {e}"
                    )
            added.extend(stored)
            removed.extend(registered)
        if removed:
            self._update_vocabulary(added, removed, len(added) - len(removed))

    async def index_note(self, note_id: UUID) -> None:
        """Index one note into a compatible generation."""
        async with self.collection_operation():
            await self._index_one_note(note_id)

    async def _index_one_note(self, note_id: UUID) -> None:
        """
        Index a single note using upsert-then-delete-orphans pattern.

        The note's stored token sets are swapped for its new ones in a single
        vocabulary delta before the note is vectorized, so terms it introduces
        reach its own sparse vectors and an edit never counts the note twice.
        A failed vocabulary write is logged and indexing continues; a failed
        note write makes the contribution match what was left stored.

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

        # Register the delta before _index_note vectorizes (invariant 1)
        added = self._note_token_sets(parsed)
        removed = await self._indexed_token_sets(note_id)
        applied = self._update_vocabulary(added, removed, len(added) - len(removed))

        try:
            # Index with category from path - returns new chunk count
            # Note: _index_note uses upsert, so existing chunks with same IDs are replaced
            new_chunk_count = await self._index_note(parsed, category)

            # Delete orphaned chunks (old chunks beyond new chunk count)
            # This is safe because new chunks are already upserted
            await self._delete_orphan_chunks(note_id, new_chunk_count)
            await self._complete_note_index(parsed, category)
        except BaseException:
            # Cancellation counts: like any failure it can land between the
            # group writes and leave a partially replaced note.
            if applied:
                await self._revert_vocabulary_update(note_id, added, removed)
            raise

    async def delete_note_index(self, note_id: UUID) -> None:
        """
        Remove a note from the index.

        The note's vocabulary documents are retired only once its points are
        gone. Subtracting after a failed delete would leave the note searchable
        while its terms had been written off.

        Args:
            note_id: Note UUID to remove
        """
        async with self.collection_operation():
            await self._delete_one_note(note_id)

    async def _delete_one_note(self, note_id: UUID) -> None:
        await self._ensure_global_vocab()
        try:
            stored = await self._indexed_token_sets(note_id)
        except Exception as e:
            # Deletion must proceed even if the stored text cannot be read; the
            # cost is a stale contribution that the next index_all repairs.
            logger.warning(
                f"Could not read note {note_id} back before deleting it from the "
                f"index; its vocabulary contribution is left in place: {e}"
            )
            stored = []
        await self._delete_note_points(note_id)
        if stored:
            self._update_vocabulary([], stored, -len(stored))

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
