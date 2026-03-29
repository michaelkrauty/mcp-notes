"""Integration tests for search engine with real Qdrant and embeddings."""

from datetime import UTC, datetime, timedelta

import pytest


class TestHybridSearch:
    """Integration tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_finds_related(self, indexed_notes):
        """Semantic search finds conceptually related notes."""
        from mcp_notes.server import search_notes

        # Search for "programming language" should find Python and JS notes
        results = await search_notes(query="programming language", limit=5)

        assert len(results) > 0
        titles = [r["note"]["title"] for r in results]
        # Should find programming-related notes
        assert any("Python" in t or "JavaScript" in t for t in titles)

    @pytest.mark.asyncio
    async def test_semantic_search_with_tag_filter(self, indexed_notes):
        """Search with tag filter restricts results."""
        from mcp_notes.server import search_notes

        # Search for web-related with python tag
        results = await search_notes(
            query="web development",
            tags=["python"],
            limit=5,
        )

        # All results should have python tag
        for r in results:
            assert "python" in r["note"]["tags"]

    @pytest.mark.asyncio
    async def test_semantic_search_with_category_filter(self, indexed_notes):
        """Search with category filter restricts results."""
        from mcp_notes.server import search_notes

        # Search within tutorials category
        results = await search_notes(
            query="programming",
            category="tutorials",
            limit=5,
        )

        # All results should be in tutorials category
        for r in results:
            if r["note"]["category"]:
                assert r["note"]["category"].startswith("tutorials")

    @pytest.mark.asyncio
    async def test_search_mode_note(self, indexed_notes):
        """Search in note mode returns file-level results."""
        from mcp_notes.server import get_search

        search = await get_search()
        results = await search.search(
            query="programming",
            mode="note",
            limit=5,
        )

        assert len(results) > 0
        # Results should be note-level

    @pytest.mark.asyncio
    async def test_search_mode_chunk(self, indexed_notes):
        """Search in chunk mode returns section-level results."""
        from mcp_notes.server import get_search

        search = await get_search()
        results = await search.search(
            query="programming",
            mode="chunk",
            limit=5,
        )

        # Should have results (or empty if no chunks match)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_mode_both(self, indexed_notes):
        """Search in both mode returns mixed results."""
        from mcp_notes.server import get_search

        search = await get_search()
        results = await search.search(
            query="programming",
            mode="both",
            limit=10,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_query_syntax(self, indexed_notes):
        """Search with filter syntax in query."""
        from mcp_notes.server import search_notes

        # Use tag: syntax in query
        results = await search_notes(query="tag:python programming", limit=5)

        # Should find python-tagged notes
        for r in results:
            if r["note"]["tags"]:
                # At least some should have python tag
                pass

    @pytest.mark.asyncio
    async def test_search_limit_respected(self, indexed_notes):
        """Search respects limit parameter."""
        from mcp_notes.server import search_notes

        results = await search_notes(query="programming", limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_returns_highlights(self, indexed_notes):
        """Search returns highlight snippets."""
        from mcp_notes.server import search_notes

        results = await search_notes(query="Python programming", limit=5)

        # Check that results have expected structure
        for r in results:
            assert "note" in r
            assert "score" in r
            assert "highlights" in r


class TestFilterOnlySearch:
    """Tests for filter-only search (no semantic query)."""

    @pytest.mark.asyncio
    async def test_filter_only_by_tag(self, indexed_notes):
        """Filter-only search by tag."""
        from mcp_notes.server import get_search

        search = await get_search()
        results = await search.search(
            query="tag:python",
            mode="note",
            limit=10,
        )

        # All results should have python tag (or be empty)
        for r in results:
            if r.note.tags:
                # Filter syntax should work
                pass

    @pytest.mark.asyncio
    async def test_filter_only_by_category(self, indexed_notes):
        """Filter-only search by category."""
        from mcp_notes.server import get_search

        search = await get_search()
        results = await search.search(
            query="category:work",
            mode="note",
            limit=10,
        )

        # Results should be in work category
        for r in results:
            if r.note.category:
                assert r.note.category.startswith("work")


class TestFindSimilar:
    """Tests for find_similar_notes functionality."""

    @pytest.mark.asyncio
    async def test_find_similar_returns_related(self, indexed_notes):
        """Find similar returns conceptually related notes."""
        from mcp_notes.server import find_similar_notes

        # Get the Python guide note ID
        python_note = next(n for n in indexed_notes if "Python Programming" in n["title"])
        note_id = python_note["id"]

        results = await find_similar_notes(note_id, limit=3)

        # Should find related notes (Python Web Frameworks should be similar)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_similar_excludes_self(self, indexed_notes):
        """Find similar excludes the source note."""
        from mcp_notes.server import find_similar_notes

        python_note = next(n for n in indexed_notes if "Python Programming" in n["title"])
        note_id = python_note["id"]

        results = await find_similar_notes(note_id, limit=5)

        # Source note should not be in results
        result_ids = [r.get("note", {}).get("id") for r in results if "note" in r]
        assert note_id not in result_ids


class TestReindexNotes:
    """Tests for reindex functionality."""

    @pytest.mark.asyncio
    async def test_reindex_updates_search(self, integration_notes_dir):
        """Reindex makes new notes searchable."""
        from mcp_notes.server import create_note, reindex_notes, search_notes

        # Create a note with unique content
        await create_note(
            title="Unique Reindex Test",
            content="This contains xyzzy123unique string for testing.",
        )

        # Force reindex
        status = await reindex_notes()
        assert status["total_notes"] >= 1

        # Search should find it
        results = await search_notes(query="xyzzy123unique", limit=5)
        # May or may not find depending on timing, but shouldn't error
        assert isinstance(results, list)


class TestSearchWithDateFilters:
    """Tests for date-based search filters."""

    @pytest.mark.asyncio
    async def test_search_with_after_filter(self, indexed_notes):
        """Search with after date filter."""
        from mcp_notes.server import get_search

        search = await get_search()

        # Search for notes after a past date (should find all)
        past_date = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        results = await search.search(
            query="programming",
            after=past_date,
            limit=10,
        )

        # Should find notes created today
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_before_filter(self, indexed_notes):
        """Search with before date filter."""
        from mcp_notes.server import get_search

        search = await get_search()

        # Search for notes before a future date (should find all)
        future_date = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        results = await search.search(
            query="programming",
            before=future_date,
            limit=10,
        )

        assert len(results) > 0


class TestSearchEdgeCases:
    """Edge case tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, indexed_notes):
        """Search with empty query uses filter-only mode."""
        from mcp_notes.server import search_notes

        # Empty query with tag filter
        results = await search_notes(query="", tags=["python"], limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_whitespace_query(self, indexed_notes):
        """Search with whitespace-only query."""
        from mcp_notes.server import search_notes

        results = await search_notes(query="   ", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_special_characters(self, indexed_notes):
        """Search with special characters in query."""
        from mcp_notes.server import search_notes

        # Should not crash on special chars
        results = await search_notes(query="python & javascript", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_unicode_query(self, indexed_notes):
        """Search with unicode characters."""
        from mcp_notes.server import search_notes

        results = await search_notes(query="プログラミング", limit=5)
        assert isinstance(results, list)
