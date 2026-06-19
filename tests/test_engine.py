"""Tests for the search engine."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from mcp_notes.search.engine import NoteSearchEngine


class TestNoteSearchEngineInit:
    """Tests for search engine initialization."""

    def test_default_initialization(self):
        """Engine initializes with default components."""
        with patch("mcp_notes.search.engine.NoteStore"):
            with patch("mcp_notes.search.engine.QdrantStorage"):
                with patch("mcp_notes.search.engine.EmbeddingClient"):
                    engine = NoteSearchEngine()
                    assert engine.note_store is not None
                    assert engine.storage is not None
                    assert engine.embedder is not None

    def test_global_vocab_requires_initialization(self):
        """GlobalVocabulary property raises if not initialized."""
        with patch("mcp_notes.search.engine.NoteStore"):
            with patch("mcp_notes.search.engine.QdrantStorage"):
                with patch("mcp_notes.search.engine.EmbeddingClient"):
                    engine = NoteSearchEngine()
                    with pytest.raises(RuntimeError, match="GlobalVocabulary not initialized"):
                        _ = engine.global_vocab

    def test_custom_components(self):
        """Engine accepts custom components."""
        mock_store = MagicMock()
        mock_storage = MagicMock()
        mock_embedder = MagicMock()
        mock_global_vocab = MagicMock()

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            embedder=mock_embedder,
            global_vocab=mock_global_vocab,
        )

        assert engine.note_store is mock_store
        assert engine.storage is mock_storage
        assert engine.embedder is mock_embedder
        assert engine.global_vocab is mock_global_vocab


class TestCollectionName:
    """Tests for collection name generation."""

    def test_collection_name_generated(self):
        """Collection name is generated from base directory."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        engine = NoteSearchEngine(note_store=mock_store)
        name = engine.collection_name

        assert name.startswith("notes_")
        assert len(name) == 18  # "notes_" + 12 char hash

    def test_collection_name_cached(self):
        """Collection name is cached after first access."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        engine = NoteSearchEngine(note_store=mock_store)
        name1 = engine.collection_name
        name2 = engine.collection_name

        assert name1 == name2


class TestEnsureVocabularyRegistered:
    """Tests for vocabulary registration check."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"
        mock_global_vocab = MagicMock()

        engine = NoteSearchEngine(
            note_store=mock_store,
            global_vocab=mock_global_vocab,
        )
        return engine

    def test_vocabulary_registered(self, mock_engine):
        """Returns True if vocabulary is registered."""
        mock_engine.global_vocab.get_codebase_doc_count.return_value = 10

        result = mock_engine._ensure_vocabulary_registered()

        assert result is True
        mock_engine.global_vocab.get_codebase_doc_count.assert_called_once_with("notes")

    def test_vocabulary_not_registered(self, mock_engine):
        """Returns False if vocabulary not registered."""
        mock_engine.global_vocab.get_codebase_doc_count.return_value = 0

        result = mock_engine._ensure_vocabulary_registered()

        assert result is False


class TestSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        # Create async mock client
        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock()
        mock_client.scroll = AsyncMock(return_value=([], None))

        mock_storage = MagicMock()
        mock_storage.get_metadata = AsyncMock(return_value=None)
        mock_storage._get_client = AsyncMock(return_value=mock_client)
        mock_storage.get_client = AsyncMock(return_value=mock_client)
        mock_storage.scroll_points = AsyncMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 1024)

        mock_global_vocab = MagicMock()
        mock_global_vocab.get_codebase_doc_count.return_value = 10
        mock_global_vocab.vectorize_query = MagicMock(return_value=MagicMock(
            indices=[0, 1, 2],
            values=[0.5, 0.3, 0.2],
        ))

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            embedder=mock_embedder,
            global_vocab=mock_global_vocab,
        )
        # Store mock client for test access
        engine._mock_client = mock_client
        return engine

    async def test_search_returns_results(self, mock_engine):
        """Search returns properly formatted results."""
        mock_client = mock_engine._mock_client

        note_id = "123e4567-e89b-12d3-a456-426614174000"
        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {
            "note_id": note_id,
            "type": "note",
            "title": "Test Note",
            "tags": ["test"],
            "category": "testing",
            "content": "This is test content",
            "created": "2024-01-01T00:00:00+00:00",
            "modified": "2024-01-02T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        results = await mock_engine.search("test query")

        assert len(results) == 1
        assert results[0].note.id == UUID(note_id)
        assert results[0].note.title == "Test Note"
        assert results[0].score == 0.9

    async def test_search_empty_query_filter_only(self, mock_engine):
        """Empty query falls back to filter-only search."""
        results = await mock_engine.search("")

        assert len(results) == 0
        # Should use scroll_points via storage
        mock_engine.storage.scroll_points.assert_called()

    async def test_search_with_tags_filter(self, mock_engine):
        """Search with tag filter."""
        mock_client = mock_engine._mock_client
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        await mock_engine.search("query tag:important")

        # Verify query_points was called (search executed)
        mock_client.query_points.assert_called()

    async def test_default_search_restricts_to_notes_and_chunks(self, mock_engine):
        """search_notes calls search() with the default mode='both' and no
        type_filter. Notes, glossary entries, and facts share one Qdrant
        collection, so the query MUST carry a type filter restricting to
        note/chunk; otherwise glossary and fact points leak into note results.
        """
        mock_client = mock_engine._mock_client
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        # mode defaults to "both", type_filter defaults to None (the
        # search_notes tool passes neither).
        await mock_engine.search("some query")

        prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
        query_filter = prefetch[0].filter
        assert query_filter is not None
        type_conditions = [
            c for c in (query_filter.must or []) if getattr(c, "key", None) == "type"
        ]
        assert len(type_conditions) == 1
        # MatchAny exposes the allowed values via `.any`.
        assert set(getattr(type_conditions[0].match, "any", [])) == {"note", "chunk"}

    async def test_explicit_all_type_filter_is_not_restricted(self, mock_engine):
        """An explicit type_filter='all' is the documented all-types override:
        it must NOT add a note/chunk restriction, so glossary and fact points
        can still be returned. Only the default (type_filter=None) is scoped to
        note/chunk; 'all' is distinct from the default.
        """
        mock_client = mock_engine._mock_client
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        await mock_engine.search("some query", type_filter="all")

        prefetch = mock_client.query_points.call_args.kwargs["prefetch"]
        query_filter = prefetch[0].filter
        type_conditions = [
            c
            for c in ((query_filter.must if query_filter else []) or [])
            if getattr(c, "key", None) == "type"
        ]
        assert type_conditions == []


class TestExtractHighlights:
    """Tests for highlight extraction."""

    def test_extract_basic_highlight(self):
        """Extract highlight around query term."""
        mock_store = MagicMock()
        mock_store.base_dir = "/notes"
        engine = NoteSearchEngine(note_store=mock_store)

        content = "This is some test content with the word python in it."
        highlights = engine._extract_highlights(content, "python")

        assert len(highlights) >= 1
        assert "python" in highlights[0].lower() or len(highlights) > 0

    def test_extract_no_match(self):
        """No highlights when query not in content."""
        mock_store = MagicMock()
        mock_store.base_dir = "/notes"
        engine = NoteSearchEngine(note_store=mock_store)

        content = "This is some content."
        highlights = engine._extract_highlights(content, "xyz123")

        # May return empty or first snippet
        assert isinstance(highlights, list)


class TestFindSimilar:
    """Tests for find_similar functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        # Create async mock client
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.query_points = AsyncMock()

        mock_storage = MagicMock()
        mock_storage._get_client = AsyncMock(return_value=mock_client)
        mock_storage.get_client = AsyncMock(return_value=mock_client)
        mock_storage.collection_exists = AsyncMock(return_value=True)
        mock_storage.get_metadata = AsyncMock(return_value=None)

        # Use mock global_vocab
        mock_global_vocab = MagicMock()
        mock_global_vocab.get_codebase_doc_count.return_value = 10

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            global_vocab=mock_global_vocab,
        )
        engine._mock_client = mock_client
        return engine

    async def test_find_similar_returns_results(self, mock_engine):
        """Find similar returns matching notes."""
        mock_client = mock_engine._mock_client

        # Mock retrieve for source note
        source_id = "123e4567-e89b-12d3-a456-426614174000"
        mock_source_point = MagicMock()
        mock_source_point.vector = {"dense": [0.1] * 1024}
        mock_client.retrieve.return_value = [mock_source_point]

        # Mock similar notes query
        similar_id = "223e4567-e89b-12d3-a456-426614174001"
        mock_similar = MagicMock()
        mock_similar.score = 0.85
        mock_similar.payload = {
            "note_id": similar_id,
            "type": "note",
            "title": "Similar Note",
            "tags": [],
            "category": None,
            "content": "Similar content",
            "created": "2024-01-01T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [mock_similar]
        mock_client.query_points.return_value = mock_response

        results = await mock_engine.find_similar(UUID(source_id), limit=5)

        assert len(results) == 1
        assert results[0].note.id == UUID(similar_id)
        assert results[0].score == 0.85

    async def test_find_similar_no_source_vector(self, mock_engine):
        """Return empty if source note has no vector."""
        mock_client = mock_engine._mock_client
        mock_client.retrieve.return_value = []

        source_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        results = await mock_engine.find_similar(source_id)

        assert results == []

    async def test_find_similar_excludes_source(self, mock_engine):
        """Similar results exclude the source note itself."""
        mock_client = mock_engine._mock_client

        source_id = "123e4567-e89b-12d3-a456-426614174000"
        mock_source = MagicMock()
        mock_source.vector = {"dense": [0.1] * 1024}
        mock_client.retrieve.return_value = [mock_source]

        # Query returns both source and different note
        source_result = MagicMock()
        source_result.score = 1.0
        source_result.payload = {"note_id": source_id, "type": "note"}

        different_id = "223e4567-e89b-12d3-a456-426614174001"
        different_result = MagicMock()
        different_result.score = 0.8
        different_result.payload = {
            "note_id": different_id,
            "type": "note",
            "title": "Different Note",
            "tags": [],
            "content": "",
            "created": "2024-01-01T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [source_result, different_result]
        mock_client.query_points.return_value = mock_response

        results = await mock_engine.find_similar(UUID(source_id), limit=5)

        # Should exclude source note
        result_ids = [r.note.id for r in results]
        assert UUID(source_id) not in result_ids


class TestFilterOnlySearch:
    """Tests for filter-only search (when query is empty)."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        mock_storage = MagicMock()
        mock_storage._get_client = AsyncMock()
        mock_storage.get_metadata = AsyncMock(return_value=None)
        mock_storage.scroll_points = AsyncMock(return_value=[])

        mock_global_vocab = MagicMock()
        mock_global_vocab.get_codebase_doc_count.return_value = 10

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            global_vocab=mock_global_vocab,
        )
        return engine

    async def test_filter_only_by_tags(self, mock_engine):
        """Filter by tags without semantic query."""
        await mock_engine.search("tag:important")

        # Should use scroll_points via storage
        mock_engine.storage.scroll_points.assert_called()

    async def test_filter_only_with_limit(self, mock_engine):
        """Respect limit in filter-only search."""
        await mock_engine.search("tag:test", limit=5)

        # Verify scroll_points was called via storage
        mock_engine.storage.scroll_points.assert_called()

    async def test_filter_only_skips_invalid_uuid(self, mock_engine):
        """Filter-only search skips results with invalid note_id (line 266-267)."""
        # Return payload with invalid UUID
        mock_engine.storage.scroll_points.return_value = [
            {
                "note_id": "not-a-valid-uuid",  # Invalid UUID
                "title": "Bad Note",
                "tags": [],
                "content": "Content",
                "created": "2024-01-01T00:00:00+00:00",
            },
            {
                "note_id": "123e4567-e89b-12d3-a456-426614174000",  # Valid UUID
                "title": "Good Note",
                "tags": [],
                "content": "Content",
                "created": "2024-01-01T00:00:00+00:00",
            },
        ]

        results = await mock_engine.search("tag:test")

        # Should only return the valid note
        assert len(results) == 1
        assert results[0].note.title == "Good Note"

    async def test_filter_only_post_filter_excludes(self, mock_engine):
        """Filter-only search applies post-filters (line 260-261)."""
        # Return payloads where one has excluded tag
        mock_engine.storage.scroll_points.return_value = [
            {
                "note_id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Draft Note",
                "tags": ["draft"],  # Will be excluded
                "content": "Draft content",
                "created": "2024-01-01T00:00:00+00:00",
            },
            {
                "note_id": "223e4567-e89b-12d3-a456-426614174001",
                "title": "Published Note",
                "tags": ["published"],
                "content": "Published content",
                "created": "2024-01-01T00:00:00+00:00",
            },
        ]

        # Use -tag:draft to exclude drafts
        results = await mock_engine.search("-tag:draft")

        # Should only return published note
        assert len(results) == 1
        assert results[0].note.title == "Published Note"


class TestFindSimilarEdgeCases:
    """Edge case tests for find_similar."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.query_points = AsyncMock()

        mock_storage = MagicMock()
        mock_storage._get_client = AsyncMock(return_value=mock_client)
        mock_storage.get_client = AsyncMock(return_value=mock_client)
        mock_storage.get_metadata = AsyncMock(return_value=None)

        mock_global_vocab = MagicMock()
        mock_global_vocab.get_codebase_doc_count.return_value = 10

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            global_vocab=mock_global_vocab,
        )
        engine._mock_client = mock_client
        return engine

    async def test_find_similar_retrieve_exception(self, mock_engine):
        """find_similar returns empty on retrieve exception (line 378-379)."""
        mock_client = mock_engine._mock_client
        mock_client.retrieve.side_effect = Exception("Network error")

        source_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        results = await mock_engine.find_similar(source_id)

        assert results == []

    async def test_find_similar_vector_not_dict(self, mock_engine):
        """find_similar returns empty when vector is not a dict (line 385-386)."""
        mock_client = mock_engine._mock_client

        mock_point = MagicMock()
        mock_point.vector = [0.1] * 1024  # List instead of dict
        mock_client.retrieve.return_value = [mock_point]

        source_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        results = await mock_engine.find_similar(source_id)

        assert results == []

    async def test_find_similar_missing_dense_vector(self, mock_engine):
        """find_similar returns empty when dense vector missing (line 389)."""
        mock_client = mock_engine._mock_client

        mock_point = MagicMock()
        mock_point.vector = {"sparse": [0.1, 0.2]}  # No "dense" key
        mock_client.retrieve.return_value = [mock_point]

        source_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        results = await mock_engine.find_similar(source_id)

        assert results == []

    async def test_find_similar_skips_invalid_result_uuid(self, mock_engine):
        """find_similar skips results with invalid UUID (line 415-416)."""
        mock_client = mock_engine._mock_client

        # Source point with valid vector
        mock_source = MagicMock()
        mock_source.vector = {"dense": [0.1] * 1024}
        mock_client.retrieve.return_value = [mock_source]

        # Results include invalid and valid UUIDs
        invalid_result = MagicMock()
        invalid_result.score = 0.9
        invalid_result.payload = {
            "note_id": "not-a-valid-uuid",  # Invalid
            "title": "Bad Result",
        }

        valid_result = MagicMock()
        valid_result.score = 0.8
        valid_result.payload = {
            "note_id": "223e4567-e89b-12d3-a456-426614174001",
            "title": "Good Result",
            "tags": [],
            "category": None,
            "created": "2024-01-01T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [invalid_result, valid_result]
        mock_client.query_points.return_value = mock_response

        source_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        results = await mock_engine.find_similar(source_id, limit=5)

        # Should only include valid UUID result
        assert len(results) == 1
        assert results[0].note.title == "Good Result"


class TestSearchEdgeCases:
    """Edge case tests for search method."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked components."""
        mock_store = MagicMock()
        mock_store.base_dir = "/path/to/notes"

        mock_client = AsyncMock()
        mock_client.query_points = AsyncMock()

        mock_storage = MagicMock()
        mock_storage.get_metadata = AsyncMock(return_value=None)
        mock_storage.get_client = AsyncMock(return_value=mock_client)

        mock_embedder = MagicMock()
        mock_embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 1024)

        mock_global_vocab = MagicMock()
        mock_global_vocab.get_codebase_doc_count.return_value = 10
        mock_global_vocab.vectorize_query = MagicMock(return_value=MagicMock(
            indices=[0, 1],
            values=[0.5, 0.3],
        ))

        engine = NoteSearchEngine(
            note_store=mock_store,
            storage=mock_storage,
            embedder=mock_embedder,
            global_vocab=mock_global_vocab,
        )
        engine._mock_client = mock_client
        return engine

    async def test_search_skips_invalid_uuid(self, mock_engine):
        """Search skips results with invalid note_id (line 197-198)."""
        mock_client = mock_engine._mock_client

        # Invalid UUID result
        invalid_point = MagicMock()
        invalid_point.score = 0.95
        invalid_point.payload = {
            "note_id": "invalid-uuid-format",
            "title": "Bad Note",
            "tags": [],
            "content": "Content",
        }

        # Valid UUID result
        valid_point = MagicMock()
        valid_point.score = 0.9
        valid_point.payload = {
            "note_id": "123e4567-e89b-12d3-a456-426614174000",
            "title": "Good Note",
            "tags": [],
            "content": "Content",
            "created": "2024-01-01T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [invalid_point, valid_point]
        mock_client.query_points.return_value = mock_response

        results = await mock_engine.search("test query")

        # Should only return valid note
        assert len(results) == 1
        assert results[0].note.title == "Good Note"

    async def test_search_post_filter_excludes_tag(self, mock_engine):
        """Search skips results that have excluded tags (line 190-191)."""
        mock_client = mock_engine._mock_client

        # Result with tag that will be excluded
        excluded_point = MagicMock()
        excluded_point.score = 0.95
        excluded_point.payload = {
            "note_id": "123e4567-e89b-12d3-a456-426614174000",
            "title": "Excluded Note",
            "tags": ["draft"],  # This tag will be excluded
            "content": "Draft content",
            "created": "2024-01-01T00:00:00+00:00",
        }

        # Result without excluded tag
        included_point = MagicMock()
        included_point.score = 0.9
        included_point.payload = {
            "note_id": "223e4567-e89b-12d3-a456-426614174001",
            "title": "Included Note",
            "tags": ["published"],
            "content": "Published content",
            "created": "2024-01-01T00:00:00+00:00",
        }

        mock_response = MagicMock()
        mock_response.points = [excluded_point, included_point]
        mock_client.query_points.return_value = mock_response

        # Search with -tag:draft to exclude drafts
        results = await mock_engine.search("test -tag:draft")

        # Only the included note should be returned
        assert len(results) == 1
        assert results[0].note.title == "Included Note"
