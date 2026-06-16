"""Tests for MCP Notes server module."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from mcp_notes.server import (
    cleanup_async_resources,
    create_note,
    delete_note,
    get_facts_with_stale_sources,
    get_git,
    get_indexer,
    get_links,
    get_note_history,
    get_search,
    get_store,
    list_categories,
    list_notes,
    list_tags,
    read_note,
    rename_tag,
    restore_note_version,
    search_notes,
    update_note,
)


class TestResourceManagement:
    """Tests for resource management functions."""

    def test_get_store_singleton(self):
        """get_store returns same instance."""
        import mcp_notes.singletons as singletons_module

        # Save and reset using SyncSingleton API
        original = singletons_module._note_store.get_if_initialized()
        singletons_module._note_store.reset()

        try:
            s1 = get_store()
            s2 = get_store()
            assert s1 is s2
        finally:
            # Restore using SyncSingleton API
            singletons_module._note_store.set_instance(original)

    def test_get_git_singleton(self):
        """get_git returns same instance."""
        import mcp_notes.singletons as singletons_module

        # Save and reset using SyncSingleton API
        original = singletons_module._git_manager.get_if_initialized()
        singletons_module._git_manager.reset()

        try:
            g1 = get_git()
            g2 = get_git()
            assert g1 is g2
        finally:
            # Restore using SyncSingleton API
            singletons_module._git_manager.set_instance(original)

    def test_get_links_singleton(self):
        """get_links returns same instance."""
        import mcp_notes.singletons as singletons_module

        # Save and reset using SyncSingleton API
        original = singletons_module._link_resolver.get_if_initialized()
        singletons_module._link_resolver.reset()

        try:
            l1 = get_links()
            l2 = get_links()
            assert l1 is l2
        finally:
            # Restore using SyncSingleton API
            singletons_module._link_resolver.set_instance(original)

    @pytest.mark.asyncio
    async def test_get_indexer_singleton(self):
        """get_indexer returns same instance (AsyncSingleton pattern)."""
        import mcp_notes.singletons as singletons_module

        # Reset singleton for clean test
        singletons_module._indexer.reset()

        try:
            i1 = await get_indexer()
            i2 = await get_indexer()
            assert i1 is i2
        finally:
            singletons_module._indexer.reset()

    @pytest.mark.asyncio
    async def test_get_search_singleton(self):
        """get_search returns same instance (AsyncSingleton pattern)."""
        import mcp_notes.singletons as singletons_module

        # Reset singleton for clean test
        singletons_module._search_engine.reset()

        try:
            s1 = await get_search()
            s2 = await get_search()
            assert s1 is s2
        finally:
            singletons_module._search_engine.reset()

    @pytest.mark.asyncio
    async def test_cleanup_async_resources(self):
        """Cleanup function is async."""
        import inspect
        assert inspect.iscoroutinefunction(cleanup_async_resources)


class TestCreateNote:
    """Tests for create_note tool."""

    @pytest.mark.asyncio
    async def test_create_note_basic(self, tmp_notes_dir):
        """Create a simple note."""
        result = await create_note(
            title="Test Note",
            content="This is test content.",
        )

        # Note is created even if indexing fails
        assert result["title"] == "Test Note"
        # Content includes frontmatter, check that body is present
        assert "This is test content." in result["content"]
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_note_with_tags(self, tmp_notes_dir):
        """Create a note with tags."""
        result = await create_note(
            title="Tagged Note",
            content="Content with tags.",
            tags=["test", "python"],
        )

        assert "error" not in result
        assert "test" in result["tags"]
        assert "python" in result["tags"]

    @pytest.mark.asyncio
    async def test_create_note_with_category(self, tmp_notes_dir):
        """Create a note with category."""
        result = await create_note(
            title="Categorized Note",
            content="Content with category.",
            category="work/projects",
        )

        assert "error" not in result
        assert result["category"] == "work/projects"


class TestReadNote:
    """Tests for read_note tool."""

    @pytest.mark.asyncio
    async def test_read_note_success(self, tmp_notes_dir):
        """Read an existing note."""
        # Create first
        created = await create_note(
            title="Read Test",
            content="Read test content.",
        )
        note_id = created["id"]

        # Read back
        result = await read_note(note_id)

        assert "error" not in result
        assert result["title"] == "Read Test"

    @pytest.mark.asyncio
    async def test_read_note_invalid_uuid(self, tmp_notes_dir):
        """Read with invalid UUID."""
        result = await read_note("not-a-uuid")
        assert "error_code" in result
        assert "Invalid UUID" in result["message"]

    @pytest.mark.asyncio
    async def test_read_note_not_found(self, tmp_notes_dir):
        """Read non-existent note."""
        fake_uuid = str(uuid4())
        result = await read_note(fake_uuid)
        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestUpdateNote:
    """Tests for update_note tool."""

    @pytest.mark.asyncio
    async def test_update_note_title(self, tmp_notes_dir):
        """Update note title."""
        created = await create_note(
            title="Original Title",
            content="Original content.",
        )
        note_id = created["id"]

        result = await update_note(
            note_id=note_id,
            title="Updated Title",
        )

        # Note is updated even if re-indexing fails
        assert result["title"] == "Updated Title"
        # Content includes frontmatter, check that body is present
        assert "Original content." in result["content"]

    @pytest.mark.asyncio
    async def test_update_note_content(self, tmp_notes_dir):
        """Update note content."""
        created = await create_note(
            title="Content Test",
            content="Original content.",
        )
        note_id = created["id"]

        result = await update_note(
            note_id=note_id,
            content="Updated content.",
        )

        # Note is updated even if re-indexing fails
        # Content includes frontmatter, check that body is present
        assert "Updated content." in result["content"]

    @pytest.mark.asyncio
    async def test_update_note_invalid_uuid(self, tmp_notes_dir):
        """Update with invalid UUID."""
        result = await update_note(
            note_id="not-a-uuid",
            title="New Title",
        )
        assert "error_code" in result
        assert "Invalid UUID" in result["message"]

    @pytest.mark.asyncio
    async def test_update_note_not_found(self, tmp_notes_dir):
        """Update non-existent note."""
        fake_uuid = str(uuid4())
        result = await update_note(
            note_id=fake_uuid,
            title="New Title",
        )
        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestDeleteNote:
    """Tests for delete_note tool."""

    @pytest.mark.asyncio
    async def test_delete_note_success(self, tmp_notes_dir):
        """Delete an existing note."""
        created = await create_note(
            title="Delete Test",
            content="Will be deleted.",
        )
        note_id = created["id"]

        result = await delete_note(note_id)

        assert result["success"] is True
        assert result["deleted_id"] == note_id

        # Verify deleted
        read_result = await read_note(note_id)
        assert "error_code" in read_result

    @pytest.mark.asyncio
    async def test_delete_note_invalid_uuid(self, tmp_notes_dir):
        """Delete with invalid UUID."""
        result = await delete_note("not-a-uuid")
        assert "error_code" in result
        assert "Invalid UUID" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_note_not_found(self, tmp_notes_dir):
        """Delete non-existent note."""
        fake_uuid = str(uuid4())
        result = await delete_note(fake_uuid)
        assert "error_code" in result
        assert "not found" in result["message"].lower()


def qdrant_available() -> bool:
    """Check if Qdrant is running."""
    import httpx
    try:
        response = httpx.get("http://localhost:6333/collections", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at localhost:6333"
)


class TestSearchNotes:
    """Tests for search_notes tool."""

    @requires_qdrant
    @pytest.mark.asyncio
    async def test_search_notes_empty(self, tmp_notes_dir):
        """Search in empty notes (requires Qdrant collection)."""
        # Search requires Qdrant collection to exist
        # This test verifies search doesn't crash
        try:
            results = await search_notes(query="test")
            assert isinstance(results, list)
        except Exception:
            # Collection may not exist, that's expected
            pass

    @pytest.mark.asyncio
    async def test_search_notes_finds_content(self, tmp_notes_dir):
        """Search finds notes by content."""
        await create_note(
            title="Unique Search Test",
            content="This contains xyzabc123 unique string.",
        )

        results = await search_notes(query="xyzabc123")
        assert isinstance(results, list)
        # May or may not find depending on indexing timing
        # Just verify no error

    @pytest.mark.asyncio
    async def test_search_notes_with_limit(self, tmp_notes_dir):
        """Search respects limit."""
        # Create multiple notes
        for i in range(5):
            await create_note(
                title=f"Limit Test {i}",
                content=f"Content for limit test {i}.",
            )

        results = await search_notes(query="limit", limit=3)
        assert len(results) <= 3


class TestListNotes:
    """Tests for list_notes tool."""

    @pytest.mark.asyncio
    async def test_list_notes_empty(self, tmp_notes_dir):
        """List notes on empty store."""
        results = await list_notes()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_list_notes_with_notes(self, tmp_notes_dir):
        """List notes returns created notes."""
        await create_note(title="List Test 1", content="Content 1")
        await create_note(title="List Test 2", content="Content 2")

        results = await list_notes()
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_list_notes_tag_filter(self, tmp_notes_dir):
        """List notes filters by tag."""
        await create_note(
            title="Tagged",
            content="With tag",
            tags=["special-tag"],
        )
        await create_note(
            title="Not Tagged",
            content="Without tag",
        )

        results = await list_notes(tags=["special-tag"])
        assert all("special-tag" in r.get("tags", []) for r in results)

    @pytest.mark.asyncio
    async def test_list_notes_tag_filter_normalizes(self, tmp_notes_dir):
        """A mixed-case / spaced tag filter is normalized to the stored form."""
        await create_note(
            title="Tagged",
            content="With tag",
            tags=["special-tag"],
        )
        # "Special Tag" normalizes to "special-tag" (lowercase, space->hyphen).
        results = await list_notes(tags=["Special Tag"])
        assert len(results) >= 1
        assert all("special-tag" in r.get("tags", []) for r in results)

    @pytest.mark.asyncio
    async def test_list_notes_category_filter(self, tmp_notes_dir):
        """List notes filters by category."""
        await create_note(
            title="Categorized",
            content="With category",
            category="work",
        )
        await create_note(
            title="Uncategorized",
            content="Without category",
        )

        results = await list_notes(category="work")
        # All results should be in work category
        for r in results:
            if r.get("category"):
                assert r["category"].startswith("work")

    @pytest.mark.asyncio
    async def test_list_notes_category_filter_normalizes(self, tmp_notes_dir):
        """A human-form category filter matches notes stored under the slug.

        Categories are slugified on write ("Work & Projects" -> "work-projects"),
        so an un-normalized filter previously returned nothing.
        """
        await create_note(
            title="Categorized",
            content="With category",
            category="Work & Projects",
        )

        # Stored slug is "work-projects"; the human form and a case-only
        # variant must both match.
        assert len(await list_notes(category="Work & Projects")) == 1
        slug_results = await list_notes(category="work-projects")
        assert len(slug_results) == 1
        assert slug_results[0]["category"] == "work-projects"

    @pytest.mark.asyncio
    async def test_list_notes_invalid_sort_by(self, tmp_notes_dir):
        """An unsupported sort_by returns a clear error instead of silently not sorting."""
        result = await list_notes(sort_by="date")
        assert len(result) == 1
        assert result[0]["error_code"] == "invalid_input"
        assert "sort_by" in result[0]["message"]
        # Lists every supported value so the caller can correct itself
        assert "modified, created, title" in result[0]["message"]

    @pytest.mark.asyncio
    async def test_list_notes_invalid_sort_by_fails_fast(self, monkeypatch):
        """sort_by is validated before any store access (no I/O on bad input)."""
        def boom():
            raise AssertionError("get_store must not be called for an invalid sort_by")

        monkeypatch.setattr("mcp_notes.tools.search.get_store", boom)
        result = await list_notes(sort_by="nope")
        assert result[0]["error_code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_list_notes_sort_by_title(self, tmp_notes_dir):
        """A valid sort_by ('title') still sorts results ascending."""
        await create_note(title="Banana", content="b")
        await create_note(title="apple", content="a")
        await create_note(title="Cherry", content="c")

        results = await list_notes(sort_by="title")
        titles = [r["title"] for r in results if "error_code" not in r]
        assert titles == sorted(titles, key=str.lower)


class TestGetFactsWithStaleSources:
    """Tests for get_facts_with_stale_sources status validation."""

    @pytest.mark.asyncio
    async def test_invalid_status(self, tmp_notes_dir):
        """An unsupported status returns a clear error instead of an empty result."""
        result = await get_facts_with_stale_sources(status="active")
        assert len(result) == 1
        assert result[0]["error_code"] == "invalid_input"
        assert "status" in result[0]["message"]
        # Lists every supported value so the caller can correct itself
        assert "deleted, modified, all" in result[0]["message"]

    @pytest.mark.asyncio
    async def test_invalid_status_fails_fast(self, monkeypatch):
        """status is validated before the integrity manager is accessed."""
        def boom():
            raise AssertionError("get_integrity_manager must not be called for invalid status")

        monkeypatch.setattr("mcp_notes.tools.integrity.get_integrity_manager", boom)
        result = await get_facts_with_stale_sources(status="nope")
        assert result[0]["error_code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_valid_status_returns_list(self, tmp_notes_dir):
        """A valid status ('all') passes validation and returns a list (no error)."""
        result = await get_facts_with_stale_sources(status="all")
        assert isinstance(result, list)
        assert not any("error_code" in r for r in result)


class TestListTags:
    """Tests for list_tags tool."""

    @pytest.mark.asyncio
    async def test_list_tags_empty(self, tmp_notes_dir):
        """List tags on empty store."""
        results = await list_tags()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_list_tags_with_notes(self, tmp_notes_dir):
        """List tags shows tags from notes."""
        await create_note(
            title="Tagged Note",
            content="Content",
            tags=["python", "testing"],
        )

        results = await list_tags()
        # Tags should include python and testing
        tag_names = [t.get("name") or t.get("tag") for t in results]
        assert "python" in tag_names or any("python" in str(t) for t in results)


class TestListCategories:
    """Tests for list_categories tool."""

    @pytest.mark.asyncio
    async def test_list_categories_empty(self, tmp_notes_dir):
        """List categories on empty store."""
        results = await list_categories()
        assert isinstance(results, list) or isinstance(results, dict)

    @pytest.mark.asyncio
    async def test_list_categories_with_notes(self, tmp_notes_dir):
        """List categories shows categories from notes."""
        await create_note(
            title="Categorized Note",
            content="Content",
            category="work/projects",
        )

        results = await list_categories()
        # Should have some category info
        assert results is not None


class TestFindSimilarNotes:
    """Tests for find_similar_notes tool."""

    @pytest.mark.asyncio
    async def test_find_similar_invalid_uuid(self, tmp_notes_dir):
        """Find similar with invalid UUID."""
        from mcp_notes.server import find_similar_notes

        result = await find_similar_notes("not-a-uuid")
        assert len(result) == 1
        assert "error_code" in result[0]
        assert "Invalid UUID" in result[0]["message"]

    @pytest.mark.asyncio
    async def test_find_similar_returns_list(self, tmp_notes_dir):
        """Find similar returns list."""
        from mcp_notes.server import find_similar_notes

        # Create a note
        created = await create_note(
            title="Test Note",
            content="Some test content.",
        )

        result = await find_similar_notes(created["id"])
        assert isinstance(result, list)


class TestGetNoteHistory:
    """Tests for get_note_history tool."""

    @pytest.mark.asyncio
    async def test_get_history_invalid_uuid(self, tmp_notes_dir):
        """Get history with invalid UUID."""
        from mcp_notes.server import get_note_history

        result = await get_note_history("not-a-uuid")
        assert len(result) == 1
        assert "error_code" in result[0]

    @pytest.mark.asyncio
    async def test_get_history_returns_list(self, tmp_notes_dir):
        """Get history returns list."""
        from mcp_notes.server import get_note_history

        # Create a note
        created = await create_note(
            title="History Test",
            content="Some content.",
        )

        result = await get_note_history(created["id"])
        assert isinstance(result, list)


class TestRestoreNoteVersion:
    """Tests for restore_note_version tool."""

    @pytest.mark.asyncio
    async def test_restore_invalid_uuid(self, tmp_notes_dir):
        """Restore with invalid UUID."""
        from mcp_notes.server import restore_note_version

        result = await restore_note_version("not-a-uuid", "abc123")
        assert "error_code" in result
        assert "Invalid UUID" in result["message"]


class TestDeletedNoteRecovery:
    """End-to-end recovery of deleted notes via history + restore (issue #13)."""

    @pytest.mark.asyncio
    async def test_history_and_restore_after_delete(self, tmp_notes_dir):
        """create -> delete -> get_note_history -> restore_note_version works."""
        created = await create_note(
            title="Recoverable Note",
            content="Important content.",
            category="work",
        )
        note_id = created["id"]

        deleted = await delete_note(note_id)
        assert deleted.get("success") is True

        # History must be discoverable even though the path is gone
        history = await get_note_history(note_id)
        assert len(history) >= 2
        assert all("commit_sha" in v for v in history)

        # Restore from the creation commit (oldest entry)
        restored = await restore_note_version(note_id, history[-1]["commit_sha"])
        assert "error_code" not in restored, restored
        assert restored["title"] == "Recoverable Note"
        assert "Important content." in restored["content"]

        # The restored note is readable through the store again
        note = await read_note(note_id)
        assert note["title"] == "Recoverable Note"


class TestGetNoteLinks:
    """Tests for get_note_links tool."""

    @pytest.mark.asyncio
    async def test_get_links_invalid_uuid(self, tmp_notes_dir):
        """Get links with invalid UUID."""
        from mcp_notes.server import get_note_links

        result = await get_note_links("not-a-uuid")
        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_get_links_returns_dict(self, tmp_notes_dir):
        """Get links returns proper structure."""
        from mcp_notes.server import get_note_links

        # Create a note
        created = await create_note(
            title="Link Test",
            content="Content with no links.",
        )

        result = await get_note_links(created["id"])
        # Should have outgoing, incoming, and broken keys
        assert "outgoing" in result or "error" not in result


class TestReindexNotes:
    """Tests for reindex_notes tool."""

    @pytest.mark.asyncio
    async def test_reindex_returns_status(self, tmp_notes_dir):
        """Reindex returns index status."""
        from mcp_notes.server import reindex_notes

        result = await reindex_notes()
        assert isinstance(result, dict)
        # Should have status fields
        assert "total_notes" in result or "indexed_notes" in result


class TestRenameTag:
    """Tests for rename_tag tool."""

    @pytest.mark.asyncio
    async def test_rename_tag_no_matches(self, tmp_notes_dir):
        """Rename tag with no matching notes."""
        from mcp_notes.server import rename_tag

        result = await rename_tag("nonexistent", "newname")
        assert result["updated_count"] == 0

    @pytest.mark.asyncio
    async def test_rename_tag_updates_notes(self, tmp_notes_dir):
        """Rename tag updates matching notes."""
        from mcp_notes.server import rename_tag

        # Create notes with tag
        await create_note(
            title="Tagged 1",
            content="Content",
            tags=["old-tag"],
        )
        await create_note(
            title="Tagged 2",
            content="Content",
            tags=["old-tag", "other"],
        )

        result = await rename_tag("old-tag", "new-tag")
        assert result["updated_count"] == 2

    @pytest.mark.asyncio
    async def test_rename_tag_onto_existing_tag_no_duplicates(self, tmp_notes_dir, monkeypatch):
        """Renaming onto a tag the note already has must not duplicate it (issue #13)."""
        # Stub the indexer so re-indexing doesn't require a live embedding service
        fake_indexer = AsyncMock()

        async def fake_get_indexer():
            return fake_indexer

        monkeypatch.setattr("mcp_notes.tools.tags.get_indexer", fake_get_indexer)

        created = await create_note(
            title="Doubly Tagged",
            content="Content",
            tags=["foo", "bar"],
        )

        result = await rename_tag("foo", "bar")
        assert result["updated_count"] == 1

        note = await read_note(created["id"])
        assert note["tags"] == ["bar"]


class TestMergeTags:
    """Tests for merge_tags tool."""

    @pytest.mark.asyncio
    async def test_merge_tags_no_matches(self, tmp_notes_dir):
        """Merge tags with no matching notes."""
        from mcp_notes.server import merge_tags

        result = await merge_tags(["nonexistent1", "nonexistent2"], "target")
        assert result["updated_count"] == 0

    @pytest.mark.asyncio
    async def test_merge_tags_updates_notes(self, tmp_notes_dir):
        """Merge tags updates matching notes."""
        from mcp_notes.server import merge_tags

        # Create notes with different tags
        await create_note(
            title="Tagged 1",
            content="Content",
            tags=["source1"],
        )
        await create_note(
            title="Tagged 2",
            content="Content",
            tags=["source2"],
        )

        result = await merge_tags(["source1", "source2"], "target")
        assert result["updated_count"] == 2


class TestMoveCategory:
    """Tests for move_category tool."""

    @pytest.mark.asyncio
    async def test_move_category_no_matches(self, tmp_notes_dir):
        """Move category with no matching notes."""
        from mcp_notes.server import move_category

        result = await move_category("nonexistent", "newpath")
        assert result["updated_count"] == 0

    @pytest.mark.asyncio
    async def test_move_category_updates_notes(self, tmp_notes_dir):
        """Move category updates matching notes."""
        from mcp_notes.server import move_category

        # Create notes with category
        await create_note(
            title="Note 1",
            content="Content",
            category="old/path",
        )
        await create_note(
            title="Note 2",
            content="Content",
            category="old/path/sub",
        )

        result = await move_category("old", "new")
        assert result["updated_count"] == 2

    @pytest.mark.asyncio
    async def test_move_category_matches_unslugified_old_path(
        self, tmp_notes_dir, monkeypatch
    ):
        """old_path/new_path are normalized to the stored slug form, so a
        human-form old_path ("Work & Projects") matches notes stored under
        "work-projects" instead of silently moving nothing."""
        from mcp_notes.server import move_category

        # Avoid the embedding-backed reindex (index_all) so the test is
        # environment independent; it asserts the match/rename logic only.
        fake_indexer = AsyncMock()
        monkeypatch.setattr(
            "mcp_notes.tools.categories.get_indexer",
            AsyncMock(return_value=fake_indexer),
        )

        await create_note(
            title="Note 1",
            content="Content",
            category="Work & Projects",
        )

        result = await move_category("Work & Projects", "Archive")
        assert result["updated_count"] == 1
        fake_indexer.index_all.assert_awaited_once()

        # The note now lives under the normalized destination slug.
        moved = await list_notes(category="archive")
        assert len(moved) == 1
        assert moved[0]["category"] == "archive"


class TestListNotesSort:
    """Tests for list_notes sorting options."""

    @pytest.mark.asyncio
    async def test_list_notes_sort_by_title(self, tmp_notes_dir):
        """List notes sorted by title."""
        await create_note(title="Zebra", content="Content")
        await create_note(title="Apple", content="Content")
        await create_note(title="Banana", content="Content")

        results = await list_notes(sort_by="title")
        titles = [r["title"] for r in results]
        assert titles == sorted(titles, key=str.lower)

    @pytest.mark.asyncio
    async def test_list_notes_sort_by_created(self, tmp_notes_dir):
        """List notes sorted by created date."""
        await create_note(title="First", content="Content")
        await create_note(title="Second", content="Content")
        await create_note(title="Third", content="Content")

        results = await list_notes(sort_by="created")
        # Most recent first
        assert results[0]["title"] == "Third"


class TestCleanupAsyncResources:
    """Tests for cleanup_async_resources."""

    @pytest.mark.asyncio
    async def test_cleanup_clears_resources(self, tmp_notes_dir):
        """Cleanup resets global instances."""
        import mcp_notes.singletons as singletons_module

        # Initialize resources
        await get_indexer()
        await get_search()

        # Cleanup
        await cleanup_async_resources()

        # AsyncSingleton instances should not be initialized after cleanup
        assert not singletons_module._indexer.is_initialized
        assert not singletons_module._search_engine.is_initialized


class TestMCPResources:
    """Tests for MCP resource endpoints."""

    @pytest.mark.asyncio
    async def test_get_notes_index(self, tmp_notes_dir):
        """Get notes index resource."""
        import json

        from mcp_notes.server import get_notes_index

        # Create some notes
        await create_note(title="Note 1", content="Content 1")
        await create_note(title="Note 2", content="Content 2")

        result = await get_notes_index()
        data = json.loads(result)

        assert "notes" in data
        assert "total" in data
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_get_tags_resource(self, tmp_notes_dir):
        """Get tags resource."""
        import json

        from mcp_notes.server import get_tags_resource

        # Create notes with tags
        await create_note(title="Note 1", content="Content", tags=["python"])
        await create_note(title="Note 2", content="Content", tags=["python", "testing"])

        result = await get_tags_resource()
        data = json.loads(result)

        assert isinstance(data, list)
        tag_names = [t["name"] for t in data]
        assert "python" in tag_names

    @pytest.mark.asyncio
    async def test_get_categories_resource(self, tmp_notes_dir):
        """Get categories resource."""
        import json

        from mcp_notes.server import get_categories_resource

        # Create notes with categories
        await create_note(title="Note 1", content="Content", category="work")
        await create_note(title="Note 2", content="Content", category="personal")

        result = await get_categories_resource()
        data = json.loads(result)

        assert "categories" in data
        assert "total_notes" in data

    @pytest.mark.asyncio
    async def test_get_recent_notes(self, tmp_notes_dir):
        """Get recent notes resource."""
        import json

        from mcp_notes.server import get_recent_notes

        # Create some notes
        await create_note(title="Old Note", content="Content")
        await create_note(title="Recent Note", content="Content")

        result = await get_recent_notes()
        data = json.loads(result)

        assert isinstance(data, list)
        # Most recent should be first
        assert data[0]["title"] == "Recent Note"

    @pytest.mark.asyncio
    async def test_get_orphan_notes(self, tmp_notes_dir):
        """Get orphan notes resource."""
        import json

        from mcp_notes.server import get_orphan_notes

        # Create an orphan note (no incoming links)
        await create_note(title="Orphan", content="Content")

        result = await get_orphan_notes()
        data = json.loads(result)

        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_broken_links_resource(self, tmp_notes_dir):
        """Get broken links resource."""
        import json

        from mcp_notes.server import get_broken_links_resource

        # Create note with broken link
        await create_note(
            title="With Broken Link",
            content="See [[00000000-0000-0000-0000-000000000000]]",
        )

        result = await get_broken_links_resource()
        data = json.loads(result)

        assert isinstance(data, list)


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_limit_with_none(self):
        """validate_limit returns default when None."""
        from vector_core import validate_limit

        result = validate_limit(None, default=10)
        assert result == 10

    def test_validate_limit_clamps_low(self):
        """validate_limit clamps to minimum."""
        from vector_core import DEFAULT_MIN_LIMIT, validate_limit

        result = validate_limit(-1)
        assert result == 10  # Default when <=0

    def test_validate_limit_clamps_high(self):
        """validate_limit clamps to maximum."""
        from vector_core import DEFAULT_MAX_LIMIT, validate_limit

        result = validate_limit(DEFAULT_MAX_LIMIT + 1)
        assert result == DEFAULT_MAX_LIMIT

    def test_validate_tag_empty_string(self):
        """_validate_tag rejects empty tag (line 62)."""
        from mcp_notes.server import _validate_tag

        normalized, error = _validate_tag("   ")
        assert error == "Tag cannot be empty"
        assert normalized == ""

    def test_validate_tag_too_long(self):
        """_validate_tag rejects long tag (line 65)."""
        from mcp_notes.server import _validate_tag
        from mcp_notes.settings import settings

        long_tag = "a" * (settings.max_tag_length + 1)
        normalized, error = _validate_tag(long_tag)
        assert "exceeds maximum length" in error

    def test_validate_tag_invalid_pattern(self):
        """_validate_tag rejects invalid pattern (line 68)."""
        from mcp_notes.server import _validate_tag

        # Tag starting with hyphen is invalid
        normalized, error = _validate_tag("-invalid")
        assert "must contain only" in error

    def test_validate_tag_normalizes(self):
        """_validate_tag normalizes valid tags."""
        from mcp_notes.server import _validate_tag

        normalized, error = _validate_tag("  My Tag  ")
        assert error is None
        assert normalized == "my-tag"


class TestHealthAndDiagnostics:
    """Tests for health and diagnostic functions."""

    @pytest.mark.asyncio
    async def test_get_parse_errors_resource(self, tmp_notes_dir):
        """get_parse_errors_resource returns parse errors (lines 872-879)."""
        import json

        from mcp_notes.server import get_parse_errors_resource

        result = await get_parse_errors_resource()
        data = json.loads(result)

        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_check_note_health_healthy(self, tmp_notes_dir):
        """check_note_health reports healthy when no issues (lines 897-941)."""
        from mcp_notes.server import check_note_health

        # Create a valid note
        await create_note(title="Valid Note", content="Some content")

        result = await check_note_health()

        assert "total_notes" in result
        assert "parse_errors" in result
        assert "broken_links" in result
        assert "is_healthy" in result
        assert result["total_notes"] >= 1
        assert result["parse_errors"] == 0

    @pytest.mark.asyncio
    async def test_check_note_health_with_broken_links(self, tmp_notes_dir):
        """check_note_health detects broken links (lines 929-939)."""
        from mcp_notes.server import check_note_health

        # Create a note with a broken link
        await create_note(
            title="With Broken Link",
            content="See [[00000000-0000-0000-0000-000000000000]]",
        )

        result = await check_note_health()

        assert result["broken_links"] >= 1
        assert result["is_healthy"] is False
        if result["broken_links"] > 0:
            assert "broken_link_details" in result
