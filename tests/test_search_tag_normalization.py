"""Regression tests: tag filters must be normalized to their stored form.

Note tags are persisted as ``normalize_tag()`` output (lowercase, stripped,
spaces -> hyphens). A filter that is not normalized the same way silently
matches nothing. The ``tag:`` query syntax was already normalized, but the
explicit ``tags=[...]`` parameter to ``search_notes`` was passed through raw,
so e.g. ``tags=["Work"]`` returned zero results.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.search.filters import SearchFilters, filters_to_qdrant


def _tags_in_query_points(mock_client) -> list[str]:
    """Collect the 'tags' MatchValue values applied by the most recent
    query_points call.

    Hybrid search issues two prefetch branches (sparse + dense) that share the
    same Filter object, so inspecting a single branch reflects the actual set
    of tag conditions (looking at both would double-count)."""
    call = mock_client.query_points.call_args
    prefetch = call.kwargs["prefetch"]
    flt = getattr(prefetch[0], "filter", None)
    if flt is None:
        return []
    return [
        cond.match.value
        for cond in (flt.must or [])
        if getattr(cond, "key", None) == "tags"
    ]


@pytest.fixture
def mock_engine():
    """Engine with mocked Qdrant client/embedder/vocab (mirrors test_engine)."""
    mock_store = MagicMock()
    mock_store.base_dir = "/path/to/notes"

    mock_client = AsyncMock()
    mock_client.query_points = AsyncMock()
    mock_client.scroll = AsyncMock(return_value=([], None))

    mock_storage = MagicMock()
    mock_storage.get_client = AsyncMock(return_value=mock_client)
    mock_storage.scroll_points = AsyncMock(return_value=[])

    mock_embedder = MagicMock()
    mock_embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 1024)

    mock_global_vocab = MagicMock()
    mock_global_vocab.get_codebase_doc_count.return_value = 10
    mock_global_vocab.vectorize_query = MagicMock(
        return_value=MagicMock(indices=[0, 1, 2], values=[0.5, 0.3, 0.2])
    )

    engine = NoteSearchEngine(
        note_store=mock_store,
        storage=mock_storage,
        embedder=mock_embedder,
        global_vocab=mock_global_vocab,
    )
    engine._mock_client = mock_client
    return engine


class TestFiltersToQdrantEmptyTag:
    def test_empty_tag_skipped(self):
        """An empty tag must not become a MatchValue('') that matches nothing."""
        conditions = filters_to_qdrant(SearchFilters(tags=["work", ""]))

        assert len(conditions) == 1
        assert conditions[0].key == "tags"
        assert conditions[0].match.value == "work"


class TestSearchFiltersAddTags:
    """SearchFilters.add_tags normalizes, de-dupes, and ignores empties/None."""

    def test_normalizes_and_dedupes(self):
        filters = SearchFilters(tags=["work"])
        filters.add_tags(["Work", "My Tag", "my tag"])
        assert filters.tags == ["work", "my-tag"]

    def test_none_is_noop(self):
        filters = SearchFilters(tags=["work"])
        filters.add_tags(None)
        assert filters.tags == ["work"]

    def test_blank_tags_ignored(self):
        filters = SearchFilters()
        filters.add_tags(["   ", ""])
        assert filters.tags == []


class TestExplicitTagNormalization:
    """The explicit tags= parameter is normalized like tag: query syntax."""

    async def test_mixed_case_tag_normalized(self, mock_engine):
        mock_client = mock_engine._mock_client
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        await mock_engine.search("meeting", tags=["Work"])

        values = _tags_in_query_points(mock_client)
        assert "work" in values
        assert "Work" not in values

    async def test_spaced_tag_normalized_to_hyphen(self, mock_engine):
        mock_client = mock_engine._mock_client
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        await mock_engine.search("meeting", tags=["My Tag"])

        assert "my-tag" in _tags_in_query_points(mock_client)

    async def test_explicit_and_query_tags_do_not_duplicate(self, mock_engine):
        """Explicit tag equal (after normalization) to a tag: in the query is
        only applied once."""
        mock_client = mock_engine._mock_client
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        await mock_engine.search("meeting tag:work", tags=["Work"])

        assert _tags_in_query_points(mock_client).count("work") == 1
