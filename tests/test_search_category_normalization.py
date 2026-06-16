"""Regression tests: category filters must be normalized to their stored form.

Note categories are slugified before being written to disk and into the index
payload (e.g. "Work & Projects" -> "work-projects"). A filter that is not
slugified the same way silently matches nothing. The ``category:`` query syntax
and the explicit ``category=`` parameter to ``search_notes`` were both passed
through raw, so e.g. ``category="Finance"`` returned zero results even when a
note plainly lived under that category. This mirrors the tag-normalization fix.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.search.filters import SearchFilters, filters_to_qdrant


def _categories_in_query_points(mock_client) -> list[str]:
    """Collect the 'category' MatchValue values applied by the most recent
    query_points call. The sparse + dense prefetch branches share one Filter,
    so inspecting a single branch reflects the actual set of conditions."""
    call = mock_client.query_points.call_args
    prefetch = call.kwargs["prefetch"]
    flt = getattr(prefetch[0], "filter", None)
    if flt is None:
        return []
    return [
        cond.match.value
        for cond in (flt.must or [])
        if getattr(cond, "key", None) == "category"
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


class TestFiltersToQdrantCategory:
    def test_category_condition_uses_value(self):
        """A category filter becomes a single 'category' MatchValue condition."""
        conditions = filters_to_qdrant(SearchFilters(category="work-projects"))

        category_conditions = [c for c in conditions if c.key == "category"]
        assert len(category_conditions) == 1
        assert category_conditions[0].match.value == "work-projects"


class TestSearchFiltersSetCategory:
    """SearchFilters.set_category normalizes to the stored slug form."""

    def test_normalizes_case_and_spaces(self):
        filters = SearchFilters()
        filters.set_category("Work & Projects")
        assert filters.category == "work-projects"

    def test_normalizes_each_path_segment(self):
        filters = SearchFilters()
        filters.set_category("Work/Client X")
        assert filters.category == "work/client-x"

    def test_none_clears(self):
        filters = SearchFilters(category="work")
        filters.set_category(None)
        assert filters.category is None

    def test_value_normalizing_to_empty_clears(self):
        filters = SearchFilters(category="work")
        filters.set_category("")
        assert filters.category is None


class TestExplicitCategoryNormalization:
    """The explicit category= parameter is normalized like category: syntax."""

    async def test_mixed_case_category_normalized(self, mock_engine):
        mock_client = mock_engine._mock_client
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        await mock_engine.search("meeting", category="Finance")

        values = _categories_in_query_points(mock_client)
        assert "finance" in values
        assert "Finance" not in values

    async def test_spaced_category_normalized_to_slug(self, mock_engine):
        mock_client = mock_engine._mock_client
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response

        await mock_engine.search("meeting", category="Work & Projects")

        assert "work-projects" in _categories_in_query_points(mock_client)
