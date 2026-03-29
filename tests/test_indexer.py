"""Tests for note indexer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.models import IndexStatus


class TestNoteIndexerInit:
    """Tests for NoteIndexer initialization."""

    def test_init_default(self):
        """Creates default components if not provided."""
        with patch("mcp_notes.indexing.indexer.NoteStore") as mock_store, \
             patch("mcp_notes.indexing.indexer.QdrantStorage") as mock_storage, \
             patch("mcp_notes.indexing.indexer.EmbeddingClient") as mock_embedder:

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
        """Force reindex clears collection."""
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

        await indexer.index_all(force=True)

        mock_storage.delete_collection.assert_called_once()
        mock_storage.create_collection.assert_called_once()


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
            {"note_id": "uuid1", "note_hash": "hash1"},
            {"note_id": "uuid2", "note_hash": "hash2"},
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
            {"note_id": "1", "note_hash": "h1"},
            {"note_id": "2", "note_hash": "h2"},
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
        mock_global_vocab.vectorize_document.return_value = MagicMock(indices=[1, 2], values=[0.5, 0.5])

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
        assert point.payload == {"type": "chunk"}

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
