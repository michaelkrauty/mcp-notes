"""Tests for note indexer."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from vector_core.storage.qdrant import QdrantStorage

from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.models import IndexStatus


@pytest.fixture(autouse=True)
def ready_generation(monkeypatch):
    """Indexing unit tests isolate writes from the migration coordinator."""

    @asynccontextmanager
    async def collection_operation(indexer):
        yield

    monkeypatch.setattr(NoteIndexer, "collection_operation", collection_operation)


class TestNoteIndexerInit:
    """Tests for NoteIndexer initialization."""

    def test_init_default(self):
        """Creates default components if not provided."""
        with (
            patch("mcp_notes.indexing.indexer.NoteStore") as mock_store,
            patch("mcp_notes.indexing.indexer.QdrantStorage") as mock_storage,
            patch("mcp_notes.indexing.indexer.EmbeddingClient") as mock_embedder,
        ):
            NoteIndexer()

            mock_store.assert_called_once()
            mock_storage.assert_called_once()
            mock_embedder.assert_called_once()

    def test_init_custom(self, tmp_path):
        """Uses provided components."""
        mock_store = MagicMock()
        mock_storage = MagicMock()
        mock_embedder = MagicMock()

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=mock_embedder,
        )

        assert indexer.note_store is mock_store
        assert indexer.storage is mock_storage
        assert indexer.embedder is mock_embedder

    def test_global_vocab_initialized(self):
        """GlobalVocabulary is initialized."""
        mock_store = MagicMock()
        mock_global_vocab = MagicMock()
        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
            global_vocab=mock_global_vocab,
        )

        assert indexer.global_vocab is mock_global_vocab


class TestNoteIndexerCollectionName:
    """Tests for collection_name property."""

    def test_collection_name_generated(self):
        """Collection name is generated from base_dir."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")

        with patch("mcp_notes.indexing.indexer.generate_collection_name") as mock_gen:
            mock_gen.return_value = "notes_abc123"

            indexer = NoteIndexer(
                note_store=mock_store,
                storage=MagicMock(),
                embedder=MagicMock(),
            )
            name = indexer.collection_name

            mock_gen.assert_called_once()
            assert name == "notes_abc123"

    def test_collection_name_cached(self):
        """Collection name is cached after first access."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")

        with patch("mcp_notes.indexing.indexer.generate_collection_name") as mock_gen:
            mock_gen.return_value = "notes_abc123"

            indexer = NoteIndexer(
                note_store=mock_store,
                storage=MagicMock(),
                embedder=MagicMock(),
            )
            first_name = indexer.collection_name
            second_name = indexer.collection_name

            # Should only generate once and return same value
            assert mock_gen.call_count == 1
            assert first_name == second_name == "notes_abc123"


class TestNoteIndexerEnsureCollection:
    """Tests for ensure_collection method."""

    @pytest.mark.asyncio
    async def test_creates_if_not_exists(self):
        """Creates collection if it doesn't exist."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = False

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        await indexer.ensure_collection()

        mock_storage.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_if_exists(self):
        """Skips creation if collection exists."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        await indexer.ensure_collection()

        mock_storage.create_collection.assert_not_called()


class TestNoteIndexerHashNote:
    """Tests for _hash_note method."""

    def test_hash_note(self):
        """Generates consistent hash."""
        mock_store = MagicMock()
        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
        )

        parsed = MagicMock()
        parsed.title = "Test Title"
        parsed.body = "Test body content"
        parsed.tags = ["tag1", "tag2"]
        parsed.category = "work"

        hash1 = indexer._hash_note(parsed, "work")
        hash2 = indexer._hash_note(parsed, "work")

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated SHA256

    def test_hash_different_content(self):
        """Different content produces different hash."""
        mock_store = MagicMock()
        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
        )

        parsed1 = MagicMock()
        parsed1.title = "Title 1"
        parsed1.body = "Body 1"
        parsed1.tags = []
        parsed1.category = None

        parsed2 = MagicMock()
        parsed2.title = "Title 2"
        parsed2.body = "Body 2"
        parsed2.tags = []
        parsed2.category = None

        assert indexer._hash_note(parsed1, None) != indexer._hash_note(parsed2, None)


class TestNoteIndexerIndexAll:
    """Tests for index_all method."""

    @pytest.mark.asyncio
    async def test_index_all_empty(self):
        """Returns status when no notes."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_store.iter_all.return_value = iter([])
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
        )

        with patch.object(indexer, "_get_indexed_hashes", return_value={}):
            status = await indexer.index_all()

        assert isinstance(status, IndexStatus)
        assert status.total_notes == 0
        assert status.index_healthy is True

    @pytest.mark.asyncio
    async def test_index_all_force(self):
        """Force reindex clears only known note groups from this source root."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_store.iter_all.return_value = iter([])
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True
        mock_store.notes_dir = mock_store.base_dir / "notes"
        mock_store.get_note_path.return_value = mock_store.notes_dir / "known.md"
        mock_storage.scroll_points.return_value = [{"type": "note", "note_id": str(UUID(int=1))}]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
        )

        await indexer.index_all(force=True)

        # Must NOT nuke the shared collection (would destroy glossary + facts).
        mock_storage.delete_collection.assert_not_called()
        mock_storage.delete_by_filter.assert_awaited_once_with(
            indexer.collection_name, field="note_id", value=str(UUID(int=1))
        )

    @pytest.mark.asyncio
    async def test_index_all_partial_failure_accounting(self):
        """Failed notes are not counted as indexed and mark the index unhealthy."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        notes = [(MagicMock(), None), (MagicMock(), None), (MagicMock(), None)]
        for i, (parsed, _) in enumerate(notes):
            parsed.id = UUID(int=i)
        mock_store.iter_all.return_value = iter(notes)
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
            global_vocab=MagicMock(),
        )

        # Second note fails to index; first and third succeed.
        index_note = AsyncMock(side_effect=[0, RuntimeError("boom"), 0])
        with (
            patch("mcp_notes.indexing.indexer.generate_note_summary", return_value="s"),
            patch("mcp_notes.indexing.indexer.chunk_note", return_value=[]),
            patch.object(indexer, "_get_indexed_hashes", return_value={}),
            patch.object(indexer, "_index_note", index_note),
        ):
            status = await indexer.index_all()

        assert status.total_notes == 3
        assert status.indexed_notes == 2  # only the two that succeeded
        assert status.index_healthy is False

    @pytest.mark.asyncio
    async def test_index_all_all_succeed_is_healthy(self):
        """When every note indexes, the status is healthy and counts everything."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        notes = [(MagicMock(), None), (MagicMock(), None)]
        for i, (parsed, _) in enumerate(notes):
            parsed.id = UUID(int=i)
        mock_store.iter_all.return_value = iter(notes)
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
            global_vocab=MagicMock(),
        )

        with (
            patch("mcp_notes.indexing.indexer.generate_note_summary", return_value="s"),
            patch("mcp_notes.indexing.indexer.chunk_note", return_value=[]),
            patch.object(indexer, "_get_indexed_hashes", return_value={}),
            patch.object(indexer, "_index_note", new=AsyncMock(return_value=0)),
        ):
            status = await indexer.index_all()

        assert status.total_notes == 2
        assert status.indexed_notes == 2
        assert status.index_healthy is True

    @pytest.mark.asyncio
    async def test_index_all_incremental_prunes_orphan_chunks(self):
        """A note re-indexed via index_all (incremental) has its orphans pruned."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        parsed = MagicMock()
        note_id = UUID("12345678-1234-5678-1234-567812345678")
        parsed.id = note_id
        mock_store.iter_all.return_value = iter([(parsed, None)])
        mock_storage = AsyncMock(spec=QdrantStorage)
        mock_storage.collection_exists.return_value = True
        # The note previously had 4 chunks; it now has 2 -> chunks 2,3 are orphans.
        mock_storage.scroll_points.return_value = [
            {"chunk_index": 0},
            {"chunk_index": 1},
            {"chunk_index": 2},
            {"chunk_index": 3},
        ]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
            global_vocab=MagicMock(),
        )

        with (
            patch("mcp_notes.indexing.indexer.generate_note_summary", return_value="s"),
            patch("mcp_notes.indexing.indexer.chunk_note", return_value=[]),
            patch.object(indexer, "_get_indexed_hashes", return_value={}),
            patch.object(indexer, "_index_note", new=AsyncMock(return_value=2)),
        ):
            await indexer.index_all()

        client = await mock_storage.get_client()
        client.delete.assert_awaited_once()
        deletion = client.delete.call_args.kwargs["points_selector"]
        assert deletion.must[0].match.value == str(note_id)
        assert deletion.must[2].range.gte == 2

    @pytest.mark.asyncio
    async def test_index_all_force_skips_orphan_pruning(self):
        """Force clears owned groups, so no second orphan-chunk scan is needed."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        parsed = MagicMock()
        parsed.id = UUID(int=1)
        mock_store.iter_all.return_value = iter([(parsed, None)])
        mock_storage = AsyncMock(spec=QdrantStorage)
        mock_storage.collection_exists.return_value = True

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=AsyncMock(),
            global_vocab=MagicMock(),
        )

        with (
            patch("mcp_notes.indexing.indexer.generate_note_summary", return_value="s"),
            patch("mcp_notes.indexing.indexer.chunk_note", return_value=[]),
            patch.object(indexer, "_index_note", new=AsyncMock(return_value=2)),
        ):
            await indexer.index_all(force=True)

        mock_storage.delete_points.assert_not_called()
        mock_storage.scroll_points.assert_awaited_once_with(
            indexer.collection_name, payload_fields=["type", "note_id"], max_results=0
        )


class TestNoteIndexerDeleteOrphanChunks:
    """Tests for _delete_orphan_chunks (pruning chunks when a note shrinks)."""

    @pytest.mark.asyncio
    async def test_deletes_only_orphan_chunks(self):
        """A scoped range deletes obsolete canonical chunks and their children."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        # spec=QdrantStorage guards against delete_points disappearing from
        # vector-core: the call would raise AttributeError if the method is gone.
        mock_storage = AsyncMock(spec=QdrantStorage)
        mock_storage.scroll_points.return_value = [
            {"chunk_index": 0},
            {"chunk_index": 1},
            {"chunk_index": 2},
            {"chunk_index": 3},
        ]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        note_id = UUID("12345678-1234-5678-1234-567812345678")
        # New version keeps chunks 0 and 1; chunks 2 and 3 are orphans.
        await indexer._delete_orphan_chunks(note_id, new_chunk_count=2)

        client = await mock_storage.get_client()
        client.delete.assert_awaited_once()
        assert client.delete.call_args.args == (indexer.collection_name,)
        deletion = client.delete.call_args.kwargs["points_selector"]
        assert deletion.must[0].match.value == str(note_id)
        assert deletion.must[1].match.value == "chunk"
        assert deletion.must[2].range.gte == 2

    @pytest.mark.asyncio
    async def test_cleanup_failure_is_visible(self):
        """A failed cleanup must not report a stale fragment group as healthy."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock(spec=QdrantStorage)
        mock_storage.scroll_points.return_value = [
            {"chunk_index": 0},
            {"chunk_index": 1},
        ]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        client = await mock_storage.get_client()
        client.delete.side_effect = RuntimeError("cleanup failed")
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await indexer._delete_orphan_chunks(UUID(int=0), new_chunk_count=2)


class TestNoteIndexerDeleteNoteIndex:
    """Tests for delete_note_index method."""

    @pytest.mark.asyncio
    async def test_delete_note_index(self):
        """Deletes note points from index."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock()

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        note_id = uuid4()
        await indexer.delete_note_index(note_id)

        mock_storage.delete_by_filter.assert_called_once()
        call_args = mock_storage.delete_by_filter.call_args
        assert call_args[1]["field"] == "note_id"
        assert call_args[1]["value"] == str(note_id)


class TestNoteIndexerGetIndexedHashes:
    """Tests for _get_indexed_hashes method."""

    @pytest.mark.asyncio
    async def test_returns_hashes(self):
        """Returns dict of note_id to hash."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock()
        mock_storage.scroll_points.return_value = [
            {"note_id": "uuid1", "note_hash": "hash1", "index_policy": NoteIndexer._index_policy()},
            {"note_id": "uuid2", "note_hash": "hash2", "index_policy": NoteIndexer._index_policy()},
        ]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        hashes = await indexer._get_indexed_hashes()

        assert hashes == {"uuid1": "hash1", "uuid2": "hash2"}

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """Returns empty dict on error."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_storage = AsyncMock()
        mock_storage.scroll_points.side_effect = Exception("Connection error")

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        hashes = await indexer._get_indexed_hashes()

        assert hashes == {}


class TestNoteIndexerGetStatus:
    """Tests for get_status method."""

    @pytest.mark.asyncio
    async def test_returns_status(self):
        """Returns IndexStatus with counts."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_store.count.return_value = 5
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = True
        mock_storage.scroll_points.return_value = [
            {"note_id": "1", "note_hash": "h1", "index_policy": NoteIndexer._index_policy()},
            {"note_id": "2", "note_hash": "h2", "index_policy": NoteIndexer._index_policy()},
        ]

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        status = await indexer.get_status()

        assert status.total_notes == 5
        assert status.indexed_notes == 2
        assert status.index_healthy is True

    @pytest.mark.parametrize(
        "incomplete", [{"source_reindex_pending": True}, {"index_policy": "old"}]
    )
    async def test_incomplete_note_is_not_counted_or_reported_healthy(self, incomplete):
        store = MagicMock(base_dir=Path("/isolated/notes"))
        store.count.return_value = 2
        storage = AsyncMock()
        storage.collection_exists.return_value = True
        current = {"note_hash": "complete", "index_policy": NoteIndexer._index_policy()}
        storage.scroll_points.return_value = [
            {"note_id": "complete", **current},
            {"note_id": "retry", **current, **incomplete},
        ]
        indexer = NoteIndexer(note_store=store, storage=storage, embedder=MagicMock())
        assert (await indexer._get_indexed_hashes())["retry"] == ""
        status = await indexer.get_status()
        assert status.indexed_notes == 1
        assert status.index_healthy is False

    @pytest.mark.asyncio
    async def test_status_unhealthy_no_collection(self):
        """Status unhealthy when collection doesn't exist."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_store.count.return_value = 5
        mock_storage = AsyncMock()
        mock_storage.collection_exists.return_value = False
        mock_storage.scroll_points.return_value = []

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        status = await indexer.get_status()

        assert status.index_healthy is False

    @pytest.mark.asyncio
    async def test_status_handles_errors(self):
        """Status handles errors gracefully."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_store.count.return_value = 5
        mock_storage = AsyncMock()
        mock_storage.collection_exists.side_effect = Exception("Error")
        mock_storage.scroll_points.side_effect = Exception("Error")

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=MagicMock(),
        )

        status = await indexer.get_status()

        assert status.total_notes == 5
        assert status.indexed_notes == 0
        assert status.index_healthy is False


class TestNoteIndexerClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_closes_connections(self):
        """Closes storage and embedder connections."""
        mock_store = MagicMock()
        mock_storage = AsyncMock()
        mock_embedder = AsyncMock()

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=mock_storage,
            embedder=mock_embedder,
        )

        await indexer.close()

        mock_storage.close.assert_called_once()
        mock_embedder.close.assert_called_once()


class TestNoteIndexerCreatePoint:
    """Tests for _create_point method."""

    def test_create_point_with_chunk_index(self):
        """Creates point with chunk index in ID."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_global_vocab = MagicMock()
        mock_global_vocab.vectorize_document.return_value = MagicMock(
            indices=[1, 2], values=[0.5, 0.5]
        )

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
            global_vocab=mock_global_vocab,
        )

        note_id = uuid4()
        point = indexer._create_point(
            point_type="chunk",
            note_id=note_id,
            chunk_index=0,
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            payload={"type": "chunk"},
        )

        assert point is not None
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert point.payload == {"type": "chunk", "embedding_text": "test content"}

    def test_create_point_without_chunk_index(self):
        """Creates point without chunk index in ID."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_global_vocab = MagicMock()
        mock_global_vocab.vectorize_document.return_value = MagicMock(indices=[1], values=[1.0])

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
            global_vocab=mock_global_vocab,
        )

        note_id = uuid4()
        point = indexer._create_point(
            point_type="note",
            note_id=note_id,
            chunk_index=None,
            content="test content",
            embedding=[0.1, 0.2],
            payload={"type": "note"},
        )

        assert point is not None

    def test_create_point_generates_uuid_id(self):
        """Creates point with deterministic UUID from key."""
        mock_store = MagicMock()
        mock_store.base_dir = Path("/home/user/notes")
        mock_global_vocab = MagicMock()
        mock_global_vocab.vectorize_document.return_value = MagicMock(indices=[1], values=[1.0])

        indexer = NoteIndexer(
            note_store=mock_store,
            storage=MagicMock(),
            embedder=MagicMock(),
            global_vocab=mock_global_vocab,
        )

        note_id = uuid4()
        point = indexer._create_point(
            point_type="note",
            note_id=note_id,
            chunk_index=None,
            content="test",
            embedding=[0.1],
            payload={},
        )

        # Point ID should be a valid UUID string format
        assert isinstance(point.id, str)
        assert len(point.id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
