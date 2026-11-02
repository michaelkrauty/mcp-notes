"""Integration tests for note versioning with git."""

from uuid import uuid4

import pytest

from mcp_notes.storage.parser import parse_note

from ..conftest import claim_notes_test_collection, delete_test_collection_sync

# Git history and link resolution are the subject here, not the vector index.
pytestmark = pytest.mark.usefixtures("offline_collection")


class TestNoteVersioning:
    """Tests for note version history and restore."""

    @pytest.fixture
    def versioned_notes_dir(self, tmp_path, monkeypatch):
        """Create notes directory with git for versioning."""
        import subprocess

        notes_dir = tmp_path / f"notes_{uuid4().hex[:8]}"
        notes_dir.mkdir()

        # Initialize git
        subprocess.run(["git", "init"], check=False, cwd=notes_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=False, cwd=notes_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=False, cwd=notes_dir, capture_output=True
        )

        # Patch settings
        monkeypatch.setenv("NOTES_DIR", str(notes_dir))

        # Reset GlobalVocabulary singleton from vector-core and clear stale data
        from vector_core.embeddings.global_vocab import GlobalVocabulary
        GlobalVocabulary.reset_instance()
        vocab = GlobalVocabulary.get_instance()
        try:
            vocab.unregister_codebase("notes")
        except Exception:
            pass

        # Reset global instances
        import mcp_notes.singletons as singletons_module

        # Save sync singleton instances using SyncSingleton API
        original_store = singletons_module._note_store.get_if_initialized()
        original_git = singletons_module._git_manager.get_if_initialized()
        original_links = singletons_module._link_resolver.get_if_initialized()

        # Reset sync singletons using SyncSingleton API
        singletons_module._note_store.reset()
        singletons_module._git_manager.reset()
        singletons_module._link_resolver.reset()

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        from mcp_notes.settings import settings
        original_notes_dir = settings.dir
        settings.dir = notes_dir

        # Track collection for cleanup
        collection_name = claim_notes_test_collection(notes_dir)

        yield notes_dir

        # Cleanup: delete the test collection, its alias and generations from Qdrant
        delete_test_collection_sync(collection_name)

        # Restore sync singletons using SyncSingleton API
        singletons_module._note_store.set_instance(original_store)
        singletons_module._git_manager.set_instance(original_git)
        singletons_module._link_resolver.set_instance(original_links)

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        # Reset GlobalVocabulary to clear test data
        GlobalVocabulary.reset_instance()

        settings.dir = original_notes_dir

    @pytest.mark.asyncio
    async def test_note_history_empty(self, versioned_notes_dir):
        """New note has empty history."""
        from mcp_notes.server import create_note, get_note_history

        # Create a note
        note = await create_note(
            title="New Note",
            content="Initial content.",
        )

        # Get history
        history = await get_note_history(note["id"])

        # Should have at least the creation commit
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_note_history_with_updates(self, versioned_notes_dir):
        """Updated note has history."""
        from mcp_notes.server import create_note, get_note_history, update_note

        # Create and update
        note = await create_note(
            title="History Test",
            content="Version 1.",
        )
        note_id = note["id"]

        await update_note(note_id=note_id, content="Version 2.")
        await update_note(note_id=note_id, content="Version 3.")

        # Get history
        history = await get_note_history(note_id, limit=10)

        # One creation commit and two update commits touch this note.
        assert len(history) == 3
        assert len({version["commit_sha"] for version in history}) == 3

    @pytest.mark.asyncio
    async def test_restore_version_invalid_uuid(self, versioned_notes_dir):
        """Restore with invalid UUID returns error."""
        from mcp_notes.server import restore_note_version

        result = await restore_note_version("not-a-uuid", "abc123")

        assert "error_code" in result
        assert "Invalid UUID" in result["message"]

    @pytest.mark.asyncio
    async def test_restore_version_invalid_commit(self, versioned_notes_dir):
        """Restore with invalid commit returns error."""
        from mcp_notes.server import create_note, restore_note_version

        # Create a note
        note = await create_note(
            title="Restore Test",
            content="Content.",
        )

        # Try to restore with invalid commit
        result = await restore_note_version(note["id"], "invalid_commit_sha")

        # Should fail gracefully
        assert "error_code" in result or "Failed" in str(result)

    @pytest.mark.asyncio
    async def test_restore_version_success(self, versioned_notes_dir):
        """Restore to previous version works."""
        from mcp_notes.server import (
            create_note,
            get_note_history,
            read_note,
            restore_note_version,
            update_note,
        )

        # Create note with specific content
        original_content = "Original content ABC123."
        modified_content = "Modified content XYZ789."
        note = await create_note(
            title="Restore Test",
            content=original_content,
        )
        note_id = note["id"]

        # Update to new content
        await update_note(note_id=note_id, content=modified_content)
        modified = await read_note(note_id)
        assert parse_note(modified["content"]).body == modified_content

        # Get history to find original version
        history = await get_note_history(note_id, limit=10)
        assert len(history) == 2
        old_version = history[-1]  # Oldest version is the creation commit.
        assert old_version["message"] == "Create note: Restore Test"
        result = await restore_note_version(note_id, old_version["commit_sha"])

        assert "error_code" not in result
        restored = await read_note(note_id)
        assert parse_note(restored["content"]).body == original_content


class TestNoteLinks:
    """Tests for note linking functionality."""

    @pytest.fixture
    def linked_notes_dir(self, tmp_path, monkeypatch):
        """Create notes directory for link testing."""
        notes_dir = tmp_path / f"notes_{uuid4().hex[:8]}"
        notes_dir.mkdir()

        # Patch settings
        monkeypatch.setenv("NOTES_DIR", str(notes_dir))

        # Reset GlobalVocabulary singleton from vector-core and clear stale data
        from vector_core.embeddings.global_vocab import GlobalVocabulary
        GlobalVocabulary.reset_instance()
        vocab = GlobalVocabulary.get_instance()
        try:
            vocab.unregister_codebase("notes")
        except Exception:
            pass

        import mcp_notes.singletons as singletons_module

        # Save sync singleton instances using SyncSingleton API
        original_store = singletons_module._note_store.get_if_initialized()
        original_git = singletons_module._git_manager.get_if_initialized()
        original_links = singletons_module._link_resolver.get_if_initialized()

        # Reset sync singletons using SyncSingleton API
        singletons_module._note_store.reset()
        singletons_module._git_manager.reset()
        singletons_module._link_resolver.reset()

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        from mcp_notes.settings import settings
        original_notes_dir = settings.dir
        settings.dir = notes_dir

        collection_name = claim_notes_test_collection(notes_dir)

        yield notes_dir

        # Cleanup: delete the test collection, its alias and generations from Qdrant
        delete_test_collection_sync(collection_name)

        # Restore sync singletons using SyncSingleton API
        singletons_module._note_store.set_instance(original_store)
        singletons_module._git_manager.set_instance(original_git)
        singletons_module._link_resolver.set_instance(original_links)

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        # Reset GlobalVocabulary to clear test data
        GlobalVocabulary.reset_instance()

        settings.dir = original_notes_dir

    @pytest.mark.asyncio
    async def test_get_links_no_links(self, linked_notes_dir):
        """Get links for note with no links."""
        from mcp_notes.server import create_note, get_note_links

        note = await create_note(
            title="No Links",
            content="Just plain text.",
        )

        result = await get_note_links(note["id"])

        assert "outgoing" in result
        assert "incoming" in result
        assert "broken" in result

    @pytest.mark.asyncio
    async def test_get_links_with_outgoing(self, linked_notes_dir):
        """Get links for note with outgoing links."""
        from mcp_notes.server import create_note, get_note_links

        # Create target note
        target = await create_note(
            title="Target Note",
            content="This is the target.",
        )
        target_id = target["id"]

        # Create note linking to target
        source = await create_note(
            title="Source Note",
            content=f"See [[{target_id}]] for more.",
        )

        result = await get_note_links(source["id"])

        assert {note["id"] for note in result["outgoing"]} == {target_id}
        assert {note["id"] for note in result["incoming"]} == set()
        assert set(result["broken"]) == set()

    @pytest.mark.asyncio
    async def test_get_links_with_broken(self, linked_notes_dir):
        """Get links for note with broken links."""
        from mcp_notes.server import create_note, get_note_links

        # Create note with link to non-existent note
        note = await create_note(
            title="Broken Link Note",
            content="See [[00000000-0000-0000-0000-000000000000]] for nothing.",
        )

        result = await get_note_links(note["id"])

        assert {note["id"] for note in result["outgoing"]} == set()
        assert {note["id"] for note in result["incoming"]} == set()
        assert set(result["broken"]) == {"00000000-0000-0000-0000-000000000000"}

    @pytest.mark.asyncio
    async def test_get_links_with_backlinks(self, linked_notes_dir):
        """Get incoming links (backlinks)."""
        from mcp_notes.server import create_note, get_note_links

        # Create target note
        target = await create_note(
            title="Target with Backlinks",
            content="I'm the target.",
        )
        target_id = target["id"]

        # Create multiple notes linking to target
        linker1 = await create_note(
            title="Linker 1",
            content=f"Link to [[{target_id}]].",
        )
        linker2 = await create_note(
            title="Linker 2",
            content=f"Also links to [[{target_id}]].",
        )

        result = await get_note_links(target_id)

        assert {note["id"] for note in result["outgoing"]} == set()
        assert {note["id"] for note in result["incoming"]} == {linker1["id"], linker2["id"]}
        assert set(result["broken"]) == set()


class TestOrphanNotes:
    """Tests for finding orphan notes."""

    @pytest.fixture
    def orphan_notes_dir(self, tmp_path, monkeypatch):
        """Create notes directory for orphan testing."""
        notes_dir = tmp_path / f"notes_{uuid4().hex[:8]}"
        notes_dir.mkdir()

        monkeypatch.setenv("NOTES_DIR", str(notes_dir))

        # Reset GlobalVocabulary singleton from vector-core and clear stale data
        from vector_core.embeddings.global_vocab import GlobalVocabulary
        GlobalVocabulary.reset_instance()
        vocab = GlobalVocabulary.get_instance()
        try:
            vocab.unregister_codebase("notes")
        except Exception:
            pass

        import mcp_notes.singletons as singletons_module

        # Save sync singleton instances using SyncSingleton API
        original_store = singletons_module._note_store.get_if_initialized()
        original_git = singletons_module._git_manager.get_if_initialized()
        original_links = singletons_module._link_resolver.get_if_initialized()

        # Reset sync singletons using SyncSingleton API
        singletons_module._note_store.reset()
        singletons_module._git_manager.reset()
        singletons_module._link_resolver.reset()

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        from mcp_notes.settings import settings
        original_notes_dir = settings.dir
        settings.dir = notes_dir

        collection_name = claim_notes_test_collection(notes_dir)

        yield notes_dir

        # Cleanup: delete the test collection, its alias and generations from Qdrant
        delete_test_collection_sync(collection_name)

        # Restore sync singletons using SyncSingleton API
        singletons_module._note_store.set_instance(original_store)
        singletons_module._git_manager.set_instance(original_git)
        singletons_module._link_resolver.set_instance(original_links)

        # Reset async singletons
        singletons_module._indexer.reset()
        singletons_module._search_engine.reset()
        singletons_module._glossary_indexer.reset()
        singletons_module._fact_indexer.reset()
        singletons_module._note_service.reset()

        # Reset GlobalVocabulary to clear test data
        GlobalVocabulary.reset_instance()

        settings.dir = original_notes_dir

    @pytest.mark.asyncio
    async def test_get_orphan_notes_all_orphans(self, orphan_notes_dir):
        """All notes without incoming links are orphans."""
        import json

        from mcp_notes.server import create_note, get_orphan_notes

        # Create unlinked notes
        await create_note(title="Orphan 1", content="No links to me.")
        await create_note(title="Orphan 2", content="Also no links.")

        result = await get_orphan_notes()
        data = json.loads(result)

        assert isinstance(data, list)
        assert len(data) >= 2

    @pytest.mark.asyncio
    async def test_get_orphan_notes_with_linked(self, orphan_notes_dir):
        """Linked notes are not orphans."""
        import json

        from mcp_notes.server import create_note, get_orphan_notes

        # Create a note
        target = await create_note(
            title="Not Orphan",
            content="I'm linked.",
        )
        target_id = target["id"]

        # Create orphan
        await create_note(
            title="Orphan",
            content="No one links to me.",
        )

        # Link to target
        await create_note(
            title="Linker",
            content=f"See [[{target_id}]].",
        )

        result = await get_orphan_notes()
        data = json.loads(result)

        # Target should not be in orphans (it has a backlink)
        orphan_ids = [o.get("id") for o in data]
        assert target_id not in orphan_ids
