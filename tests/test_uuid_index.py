"""Tests for UUID-to-path index."""

import json
import logging
import multiprocessing
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.uuid_index import INDEX_FILENAME, UUIDIndex

# Spawned rather than forked: the test process may already run threads, and a
# fork copies whatever locks they hold.
_SPAWN = multiprocessing.get_context("spawn")


def _on_disk(base_dir: Path) -> dict[str, str]:
    """The map currently stored in the index file."""
    return json.loads((base_dir / ".index" / INDEX_FILENAME).read_text())["uuid_paths"]


class TestUUIDIndexInit:
    """Tests for UUIDIndex initialization."""

    def test_init_sets_paths(self, tmp_path):
        """Initialization sets correct paths."""
        index = UUIDIndex(tmp_path)
        assert index.base_dir == tmp_path
        assert index.notes_dir == tmp_path / "notes"
        assert index.index_dir == tmp_path / ".index"
        assert index.index_path == tmp_path / ".index" / INDEX_FILENAME

    def test_init_empty_index(self, tmp_path):
        """New index is empty."""
        index = UUIDIndex(tmp_path)
        assert index.count() == 0
        assert index._pending == []


class TestUUIDIndexLoad:
    """Tests for loading index from disk."""

    def test_load_nonexistent_file(self, tmp_path):
        """Loading nonexistent file results in empty index."""
        index = UUIDIndex(tmp_path)
        index.load()
        assert index.count() == 0

    def test_load_existing_file(self, tmp_path):
        """Loading existing file restores index."""
        # Create index file
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        note_id = uuid4()
        data = {"uuid_paths": {str(note_id): "test-note.md"}}
        with open(index_dir / INDEX_FILENAME, "w") as f:
            json.dump(data, f)

        # Load index
        index = UUIDIndex(tmp_path)
        index.load()

        assert index.count() == 1
        assert index.exists(note_id)

    def test_load_invalid_json(self, tmp_path):
        """Loading invalid JSON results in empty index."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        with open(index_dir / INDEX_FILENAME, "w") as f:
            f.write("not valid json")

        index = UUIDIndex(tmp_path)
        index.load()
        assert index.count() == 0

    def test_load_unexpected_structure(self, tmp_path):
        """Loading valid JSON of the wrong shape results in empty index."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        (index_dir / INDEX_FILENAME).write_text('[{"uuid_paths": {}}]')

        index = UUIDIndex(tmp_path)
        index.load()
        assert index.count() == 0

    def test_load_empty_file(self, tmp_path):
        """Loading empty file results in empty index."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        (index_dir / INDEX_FILENAME).touch()

        index = UUIDIndex(tmp_path)
        index.load()
        assert index.count() == 0


class TestUUIDIndexSave:
    """Tests for saving index to disk."""

    def test_save_creates_directory(self, tmp_path):
        """Save creates index directory if needed."""
        index = UUIDIndex(tmp_path)
        note_id = uuid4()
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        note_path = notes_dir / "test.md"
        note_path.touch()

        index.add(note_id, note_path)
        index.save()

        assert (tmp_path / ".index").exists()
        assert (tmp_path / ".index" / INDEX_FILENAME).exists()

    def test_save_writes_json(self, tmp_path):
        """Save writes valid JSON."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / "test.md"
        note_path.touch()
        index.add(note_id, note_path)
        index.save()

        # Verify JSON content
        with open(tmp_path / ".index" / INDEX_FILENAME) as f:
            data = json.load(f)

        assert "uuid_paths" in data
        assert str(note_id) in data["uuid_paths"]

    def test_save_not_dirty(self, tmp_path):
        """Save does nothing if not dirty."""
        index = UUIDIndex(tmp_path)
        index.save()  # Should not create any files
        assert not (tmp_path / ".index").exists()

    def test_add_persists_without_save(self, tmp_path):
        """Add writes the index itself; save has nothing left to do."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / "test.md"
        note_path.touch()
        index.add(note_id, note_path)

        assert index._pending == []
        assert _on_disk(tmp_path) == {str(note_id): "test.md"}
        index.save()
        assert _on_disk(tmp_path) == {str(note_id): "test.md"}


class TestUUIDIndexRebuild:
    """Tests for rebuilding index from filesystem."""

    def test_rebuild_empty_dir(self, tmp_path):
        """Rebuild on empty directory returns 0."""
        index = UUIDIndex(tmp_path)
        count = index.rebuild()
        assert count == 0
        assert index.count() == 0

    def test_rebuild_no_notes_dir(self, tmp_path):
        """Rebuild without notes directory returns 0."""
        index = UUIDIndex(tmp_path)
        # Don't create notes dir
        count = index.rebuild()
        assert count == 0

    def test_rebuild_finds_notes(self, tmp_path):
        """Rebuild finds all note files."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Create note files
        note1_id = uuid4()
        note2_id = uuid4()
        (notes_dir / f"test-{note1_id}.md").touch()
        (notes_dir / f"other-{note2_id}.md").touch()

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        assert count == 2
        assert index.exists(note1_id)
        assert index.exists(note2_id)

    def test_rebuild_finds_nested_notes(self, tmp_path):
        """Rebuild finds notes in subdirectories."""
        notes_dir = tmp_path / "notes"
        subdir = notes_dir / "projects" / "client-x"
        subdir.mkdir(parents=True)

        note_id = uuid4()
        (subdir / f"test-{note_id}.md").touch()

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        assert count == 1
        assert index.exists(note_id)

    def test_rebuild_ignores_non_md(self, tmp_path):
        """Rebuild ignores non-.md files."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        (notes_dir / f"test-{note_id}.md").touch()
        (notes_dir / "readme.txt").touch()
        (notes_dir / "data.json").touch()

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        assert count == 1

    def test_rebuild_ignores_files_without_uuid(self, tmp_path):
        """Rebuild ignores files without UUID in name."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        (notes_dir / f"test-{note_id}.md").touch()
        (notes_dir / "readme.md").touch()  # No UUID

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        assert count == 1

    def test_rebuild_clears_old_index(self, tmp_path):
        """Rebuild clears old index entries."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Add entry manually
        old_id = uuid4()
        note_path = notes_dir / f"old-{old_id}.md"
        note_path.touch()
        index.add(old_id, note_path)

        # Delete file and rebuild
        note_path.unlink()
        count = index.rebuild()

        assert count == 0
        assert not index.exists(old_id)

    def test_rebuild_saves_index(self, tmp_path):
        """Rebuild saves index to disk."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        (notes_dir / f"test-{note_id}.md").touch()

        index = UUIDIndex(tmp_path)
        index.rebuild()

        # Verify file was saved
        assert (tmp_path / ".index" / INDEX_FILENAME).exists()


class TestUUIDIndexGetPath:
    """Tests for get_path method."""

    def test_get_path_existing(self, tmp_path):
        """Get path for existing note."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_path(note_id)
        assert result == note_path

    def test_get_path_nonexistent(self, tmp_path):
        """Get path for nonexistent note returns None."""
        index = UUIDIndex(tmp_path)
        result = index.get_path(uuid4())
        assert result is None

    def test_get_path_nested(self, tmp_path):
        """Get path for nested note."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        subdir = notes_dir / "projects"
        subdir.mkdir(parents=True)

        note_id = uuid4()
        note_path = subdir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_path(note_id)
        assert result == note_path


class TestUUIDIndexGetRelativePath:
    """Tests for get_relative_path method."""

    def test_get_relative_path_existing(self, tmp_path):
        """Get relative path for existing note."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_relative_path(note_id)
        assert result == f"test-{note_id}.md"

    def test_get_relative_path_nested(self, tmp_path):
        """Get relative path for nested note."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        subdir = notes_dir / "projects" / "client"
        subdir.mkdir(parents=True)

        note_id = uuid4()
        note_path = subdir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_relative_path(note_id)
        assert result == f"projects/client/test-{note_id}.md"

    def test_get_relative_path_nonexistent(self, tmp_path):
        """Get relative path for nonexistent note returns None."""
        index = UUIDIndex(tmp_path)
        result = index.get_relative_path(uuid4())
        assert result is None


class TestUUIDIndexGetUuid:
    """Tests for get_uuid (reverse lookup) method."""

    def test_get_uuid_absolute_path(self, tmp_path):
        """Get UUID from absolute path."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_uuid(note_path)
        assert result == note_id

    def test_get_uuid_relative_path(self, tmp_path):
        """Get UUID from relative path."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        result = index.get_uuid(Path(f"test-{note_id}.md"))
        assert result == note_id

    def test_get_uuid_nonexistent(self, tmp_path):
        """Get UUID for nonexistent path returns None."""
        index = UUIDIndex(tmp_path)
        result = index.get_uuid(Path("nonexistent.md"))
        assert result is None

    def test_get_uuid_outside_notes_dir(self, tmp_path):
        """Get UUID for path outside notes_dir returns None."""
        index = UUIDIndex(tmp_path)
        result = index.get_uuid(Path("/some/other/path.md"))
        assert result is None


class TestUUIDIndexAdd:
    """Tests for add method."""

    def test_add_new_entry(self, tmp_path):
        """Add new entry to index."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()

        index.add(note_id, note_path)

        assert index.exists(note_id)
        assert _on_disk(tmp_path) == {str(note_id): note_path.name}

    def test_add_updates_existing(self, tmp_path):
        """Add updates existing entry."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        old_path = notes_dir / f"old-{note_id}.md"
        new_path = notes_dir / f"new-{note_id}.md"
        old_path.touch()
        new_path.touch()

        index.add(note_id, old_path)
        index.add(note_id, new_path)

        assert index.get_path(note_id) == new_path

    def test_add_path_outside_notes_dir(self, tmp_path, caplog):
        """Add path outside notes_dir logs error."""
        import logging

        index = UUIDIndex(tmp_path)
        (tmp_path / "notes").mkdir(parents=True)

        note_id = uuid4()
        outside_path = tmp_path / "outside" / "test.md"
        outside_path.parent.mkdir(parents=True)
        outside_path.touch()

        with caplog.at_level(logging.ERROR):
            index.add(note_id, outside_path)

        assert not index.exists(note_id)


class TestUUIDIndexRemove:
    """Tests for remove method."""

    def test_remove_existing(self, tmp_path):
        """Remove existing entry."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        index.remove(note_id)

        assert not index.exists(note_id)
        assert _on_disk(tmp_path) == {}

    def test_remove_nonexistent(self, tmp_path):
        """Remove nonexistent entry is safe."""
        index = UUIDIndex(tmp_path)
        note_id = uuid4()

        # Should not raise
        index.remove(note_id)
        assert not index.exists(note_id)
        assert index._pending == []


class TestUUIDIndexExists:
    """Tests for exists method."""

    def test_exists_true(self, tmp_path):
        """Exists returns True for indexed note."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        assert index.exists(note_id) is True

    def test_exists_false(self, tmp_path):
        """Exists returns False for non-indexed note."""
        index = UUIDIndex(tmp_path)
        assert index.exists(uuid4()) is False


class TestUUIDIndexCount:
    """Tests for count method."""

    def test_count_empty(self, tmp_path):
        """Count returns 0 for empty index."""
        index = UUIDIndex(tmp_path)
        assert index.count() == 0

    def test_count_with_entries(self, tmp_path):
        """Count returns correct number."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        for i in range(5):
            note_id = uuid4()
            note_path = notes_dir / f"test-{note_id}.md"
            note_path.touch()
            index.add(note_id, note_path)

        assert index.count() == 5


class TestUUIDIndexAllUuids:
    """Tests for all_uuids method."""

    def test_all_uuids_empty(self, tmp_path):
        """All UUIDs returns empty list for empty index."""
        index = UUIDIndex(tmp_path)
        assert index.all_uuids() == []

    def test_all_uuids_returns_all(self, tmp_path):
        """All UUIDs returns all indexed UUIDs."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_ids = []
        for i in range(3):
            note_id = uuid4()
            note_ids.append(note_id)
            note_path = notes_dir / f"test-{note_id}.md"
            note_path.touch()
            index.add(note_id, note_path)

        result = index.all_uuids()
        assert len(result) == 3
        for note_id in note_ids:
            assert note_id in result


class TestUUIDIndexEnsureLoaded:
    """Tests for ensure_loaded method."""

    def test_ensure_loaded_loads_existing(self, tmp_path):
        """Ensure loaded loads existing index file."""
        # Create index file
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        note_id = uuid4()
        data = {"uuid_paths": {str(note_id): "test.md"}}
        with open(index_dir / INDEX_FILENAME, "w") as f:
            json.dump(data, f)

        index = UUIDIndex(tmp_path)
        index.ensure_loaded()

        assert index.exists(note_id)

    def test_ensure_loaded_rebuilds_if_missing(self, tmp_path):
        """Ensure loaded rebuilds if index missing but notes exist."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        (notes_dir / f"test-{note_id}.md").touch()

        index = UUIDIndex(tmp_path)
        index.ensure_loaded()

        assert index.exists(note_id)

    def test_ensure_loaded_idempotent(self, tmp_path):
        """Ensure loaded is idempotent."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index.add(note_id, note_path)

        # Multiple calls should not change state
        index.ensure_loaded()
        index.ensure_loaded()

        assert index.count() == 1


class TestUUIDIndexPersistence:
    """Tests for index persistence across instances."""

    def test_round_trip(self, tmp_path):
        """Index survives save/load cycle."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Create and save
        index1 = UUIDIndex(tmp_path)
        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()
        index1.add(note_id, note_path)
        index1.save()

        # Load in new instance
        index2 = UUIDIndex(tmp_path)
        index2.load()

        assert index2.exists(note_id)
        assert index2.get_path(note_id) == note_path

    def test_multiple_entries_persist(self, tmp_path):
        """Multiple entries persist correctly."""
        notes_dir = tmp_path / "notes"
        subdir = notes_dir / "projects"
        subdir.mkdir(parents=True)

        # Create entries
        index1 = UUIDIndex(tmp_path)
        entries = []
        for i in range(5):
            note_id = uuid4()
            if i % 2 == 0:
                note_path = notes_dir / f"note{i}-{note_id}.md"
            else:
                note_path = subdir / f"note{i}-{note_id}.md"
            note_path.touch()
            index1.add(note_id, note_path)
            entries.append((note_id, note_path))
        index1.save()

        # Verify in new instance
        index2 = UUIDIndex(tmp_path)
        index2.load()

        for note_id, note_path in entries:
            assert index2.exists(note_id)
            assert index2.get_path(note_id) == note_path


class TestUUIDIndexConcurrency:
    """Tests for thread-safe index operations."""

    def test_concurrent_add_operations(self, tmp_path):
        """Concurrent add operations don't corrupt index."""
        import threading

        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        index = UUIDIndex(tmp_path)
        errors = []
        note_ids = []

        def add_note(i):
            try:
                note_id = uuid4()
                note_path = notes_dir / f"note{i}-{note_id}.md"
                note_path.touch()
                index.add(note_id, note_path)
                note_ids.append(note_id)
            except Exception as e:
                errors.append(e)

        # Run concurrent adds
        threads = [threading.Thread(target=add_note, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert index.count() == 20
        for note_id in note_ids:
            assert index.exists(note_id)

    def test_concurrent_rebuild_and_add(self, tmp_path):
        """Concurrent rebuild and add operations are safe."""
        import threading

        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        index = UUIDIndex(tmp_path)
        errors = []

        # Pre-create some notes
        for i in range(10):
            note_id = uuid4()
            (notes_dir / f"existing{i}-{note_id}.md").touch()

        def do_rebuild():
            try:
                index.rebuild()
            except Exception as e:
                errors.append(e)

        def do_add():
            try:
                for i in range(5):
                    note_id = uuid4()
                    note_path = notes_dir / f"new{i}-{note_id}.md"
                    note_path.touch()
                    index.add(note_id, note_path)
            except Exception as e:
                errors.append(e)

        # Run rebuild and add concurrently
        t1 = threading.Thread(target=do_rebuild)
        t2 = threading.Thread(target=do_add)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        # Index should be consistent
        assert index.count() >= 10  # At least the pre-existing notes

    def test_rebuild_if_path_missing_no_double_rebuild(self, tmp_path):
        """rebuild_if_path_missing doesn't cause double rebuilds."""
        import threading

        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Create a note and add to index
        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()

        index = UUIDIndex(tmp_path)
        index.add(note_id, note_path)
        index.save()

        # Delete the file to trigger rebuild
        note_path.unlink()

        rebuild_count = []

        # Patch rebuild to track calls
        original_rebuild = index.rebuild

        def tracking_rebuild():
            rebuild_count.append(1)
            return original_rebuild()

        index.rebuild = tracking_rebuild

        # Concurrent calls to rebuild_if_path_missing
        def check_rebuild():
            index.rebuild_if_path_missing(note_id)

        threads = [threading.Thread(target=check_rebuild) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Due to locking, only one rebuild should have happened
        # (once file is gone, subsequent checks after first rebuild won't trigger)
        assert len(rebuild_count) == 1

    def test_rebuild_if_path_missing_file_exists(self, tmp_path):
        """rebuild_if_path_missing returns False if file exists."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / f"test-{note_id}.md"
        note_path.touch()

        index = UUIDIndex(tmp_path)
        index.add(note_id, note_path)

        # File exists, should not rebuild
        result = index.rebuild_if_path_missing(note_id)
        assert result is False

    def test_rebuild_if_path_missing_not_in_index(self, tmp_path):
        """rebuild_if_path_missing returns False if note not in index."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        index = UUIDIndex(tmp_path)

        # Note not in index, should not rebuild
        result = index.rebuild_if_path_missing(uuid4())
        assert result is False


class TestUUIDIndexSymlinkSecurity:
    """Tests for symlink security in UUID index rebuild."""

    def test_rebuild_skips_symlinks(self, tmp_path):
        """Rebuild skips symlink files."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Create real note
        real_id = uuid4()
        real_path = notes_dir / f"real-{real_id}.md"
        real_path.touch()

        # Create external file
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_file = external_dir / "secret.md"
        external_file.touch()

        # Create symlink in notes dir
        symlink_id = uuid4()
        symlink_path = notes_dir / f"link-{symlink_id}.md"
        symlink_path.symlink_to(external_file)

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        # Should only count the real note
        assert count == 1
        assert index.exists(real_id)
        assert not index.exists(symlink_id)

    def test_rebuild_skips_path_traversal(self, tmp_path):
        """Rebuild skips paths that escape notes_dir via symlink resolution."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        # Create real note
        real_id = uuid4()
        real_path = notes_dir / f"real-{real_id}.md"
        real_path.touch()

        # Create external directory
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinked subdirectory pointing outside
        subdir = notes_dir / "subdir"
        subdir.symlink_to(external_dir)

        # Create note in external dir
        evil_id = uuid4()
        evil_path = external_dir / f"evil-{evil_id}.md"
        evil_path.touch()

        index = UUIDIndex(tmp_path)
        count = index.rebuild()

        # Should only count the real note in notes_dir
        assert count == 1
        assert index.exists(real_id)
        assert not index.exists(evil_id)


def _create_notes_in_child(base_dir: str, out_file: str, barrier, count: int) -> None:
    """Child process: load the shared index, wait for the peer, then create notes."""
    store = NoteStore(notes_dir=Path(base_dir))
    store.ensure_directories()
    barrier.wait(timeout=60)
    created = [str(store.create(title=f"Note {i}", content="body").id) for i in range(count)]
    Path(out_file).write_text(json.dumps(created))


def _churn_index_in_child(base_dir: str, started, rounds: int) -> None:
    """Child process: commit many changes to an index another process is reading."""
    index = UUIDIndex(Path(base_dir))
    index.load()
    started.set()
    for i in range(rounds):
        note_id = uuid4()
        index.add(note_id, index.notes_dir / f"churn-{i}-{note_id}.md")
        if i % 2:
            index.remove(note_id)


def _fail(*args, **kwargs):
    raise AssertionError("must not be called")


def _failing_replace(src, dst):
    raise OSError("disk full")


@contextmanager
def _unavailable_lock(path, timeout=10.0):
    raise TimeoutError(f"Could not acquire lock on {path} within {timeout}s")
    yield


class TestUUIDIndexSharedFile:
    """Separate instances over one index file, as separate processes have."""

    def _seeded(self, tmp_path: Path) -> tuple[Path, UUID]:
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        seed_id = uuid4()
        UUIDIndex(tmp_path).add(seed_id, notes_dir / f"seed-{seed_id}.md")
        return notes_dir, seed_id

    def test_adds_from_stale_instances_both_survive(self, tmp_path):
        """Regression (#68): the second save replaced the file with its own stale map."""
        notes_dir, seed_id = self._seeded(tmp_path)
        a, b = UUIDIndex(tmp_path), UUIDIndex(tmp_path)
        a.load()
        b.load()

        x, y = uuid4(), uuid4()
        a.add(x, notes_dir / f"x-{x}.md")
        a.save()
        b.add(y, notes_dir / f"y-{y}.md")
        b.save()

        assert set(_on_disk(tmp_path)) == {str(seed_id), str(x), str(y)}
        fresh = UUIDIndex(tmp_path)
        fresh.load()
        assert fresh.get_path(x) == notes_dir / f"x-{x}.md"
        assert fresh.get_path(y) == notes_dir / f"y-{y}.md"

    def test_stale_instance_does_not_resurrect_removed_note(self, tmp_path):
        """A peer's add must not write back an entry this instance removed."""
        notes_dir, seed_id = self._seeded(tmp_path)
        a, b = UUIDIndex(tmp_path), UUIDIndex(tmp_path)
        a.load()
        b.load()
        assert a.exists(seed_id) and b.exists(seed_id)

        y = uuid4()
        a.remove(seed_id)
        b.add(y, notes_dir / f"y-{y}.md")

        assert _on_disk(tmp_path) == {str(y): f"y-{y}.md"}
        fresh = UUIDIndex(tmp_path)
        fresh.load()
        assert not fresh.exists(seed_id)
        assert fresh.get_path(y) == notes_dir / f"y-{y}.md"

    def test_remove_reaches_entry_this_instance_never_saw(self, tmp_path):
        """remove() works on the map on disk, not only on the in-memory view."""
        notes_dir, seed_id = self._seeded(tmp_path)
        stale = UUIDIndex(tmp_path)
        stale.load()
        x = uuid4()
        UUIDIndex(tmp_path).add(x, notes_dir / f"x-{x}.md")

        stale.remove(x)

        assert _on_disk(tmp_path) == {str(seed_id): f"seed-{seed_id}.md"}


class TestUUIDIndexFreshness:
    """Lookups that miss pick up what other processes wrote, without a rebuild."""

    @pytest.mark.parametrize("lookup", ["get_path", "get_relative_path", "get_uuid", "exists"])
    def test_miss_loads_note_another_instance_added(self, tmp_path, monkeypatch, lookup):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        seed_id = uuid4()
        writer = UUIDIndex(tmp_path)
        writer.add(seed_id, notes_dir / f"seed-{seed_id}.md")
        reader = UUIDIndex(tmp_path)
        reader.load()
        monkeypatch.setattr(reader, "rebuild", _fail)

        note_id = uuid4()
        path = notes_dir / f"later-{note_id}.md"
        path.touch()
        writer.add(note_id, path)

        if lookup == "get_path":
            assert reader.get_path(note_id) == path
        elif lookup == "get_relative_path":
            assert reader.get_relative_path(note_id) == path.name
        elif lookup == "get_uuid":
            assert reader.get_uuid(path) == note_id
        else:
            assert reader.exists(note_id)
        # The reload kept the entries the reader already had
        assert reader.exists(seed_id)

    def test_store_reads_note_created_by_another_store(self, tmp_path, monkeypatch):
        """Two NoteStores (two server processes) over one notes directory."""
        store_a = NoteStore(notes_dir=tmp_path)
        store_b = NoteStore(notes_dir=tmp_path)
        store_a.create(title="Seed", content="seed")
        store_b.ensure_directories()
        monkeypatch.setattr(store_b.uuid_index, "rebuild", _fail)

        note = store_a.create(title="Later", content="written by a")

        assert store_b.read(note.id).title == "Later"

    def test_hit_does_not_check_the_file(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        note_id = uuid4()
        path = notes_dir / f"note-{note_id}.md"
        index = UUIDIndex(tmp_path)
        index.add(note_id, path)
        monkeypatch.setattr(index, "_reload_if_changed", _fail)

        assert index.get_path(note_id) == path
        assert index.get_relative_path(note_id) == path.name
        assert index.get_uuid(path) == note_id
        assert index.exists(note_id)

    def test_miss_on_unchanged_file_does_not_reread(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        note_id = uuid4()
        index.add(note_id, notes_dir / f"note-{note_id}.md")
        monkeypatch.setattr(index, "_read_file", _fail)

        assert index.get_path(uuid4()) is None
        assert index.get_uuid(notes_dir / "unknown.md") is None

    def test_unreadable_version_is_not_reread_on_every_miss(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        note_id = uuid4()
        index.add(note_id, notes_dir / f"note-{note_id}.md")
        index.index_path.write_text('{"uuid_paths": {')
        real_read_file = index._read_file
        reads = []

        def counting_read_file():
            reads.append(1)
            return real_read_file()

        monkeypatch.setattr(index, "_read_file", counting_read_file)

        assert index.get_path(uuid4()) is None
        assert index.get_path(uuid4()) is None
        assert len(reads) == 1
        # The in-memory map is kept
        assert index.exists(note_id)

    def test_move_by_another_instance_is_followed_without_rebuild(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        note_id = uuid4()
        old_path = notes_dir / f"old-{note_id}.md"
        old_path.touch()
        mover = UUIDIndex(tmp_path)
        mover.add(note_id, old_path)
        follower = UUIDIndex(tmp_path)
        follower.load()

        new_path = notes_dir / "moved" / f"new-{note_id}.md"
        new_path.parent.mkdir()
        old_path.rename(new_path)
        mover.add(note_id, new_path)
        monkeypatch.setattr(follower, "rebuild", _fail)

        assert follower.rebuild_if_path_missing(note_id) is False
        assert follower.get_path(note_id) == new_path


class TestUUIDIndexAtomicWrite:
    """The index file is only ever replaced whole."""

    def test_writes_leave_only_the_index_and_its_lock(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        ids = [uuid4() for _ in range(3)]
        for note_id in ids:
            (notes_dir / f"note-{note_id}.md").touch()
            index.add(note_id, notes_dir / f"note-{note_id}.md")
        index.remove(ids[0])
        index.rebuild()

        assert sorted(p.name for p in (tmp_path / ".index").iterdir()) == [
            INDEX_FILENAME,
            f"{INDEX_FILENAME}.lock",
        ]

    def test_failed_write_cleans_up_and_is_retried_by_save(self, tmp_path, monkeypatch, caplog):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        seed_id = uuid4()
        index.add(seed_id, notes_dir / f"seed-{seed_id}.md")
        note_id = uuid4()
        path = notes_dir / f"note-{note_id}.md"

        def failing_replace(src, dst):
            raise OSError("disk full")

        with monkeypatch.context() as m, caplog.at_level(logging.ERROR):
            m.setattr(os, "replace", failing_replace)
            index.add(note_id, path)

        assert "Failed to save UUID index" in caplog.text
        assert list((tmp_path / ".index").glob("*.tmp")) == []
        assert set(_on_disk(tmp_path)) == {str(seed_id)}
        # Still served from memory
        assert index.get_path(note_id) == path

        index.save()

        assert set(_on_disk(tmp_path)) == {str(seed_id), str(note_id)}
        assert index._pending == []

    def test_failed_change_is_written_by_the_next_change(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        x, y = uuid4(), uuid4()

        def failing_fsync(fd):
            raise OSError("I/O error")

        with monkeypatch.context() as m:
            m.setattr(os, "fsync", failing_fsync)
            index.add(x, notes_dir / f"x-{x}.md")
        assert list((tmp_path / ".index").glob("*.tmp")) == []

        index.add(y, notes_dir / f"y-{y}.md")

        assert set(_on_disk(tmp_path)) == {str(x), str(y)}

    def test_change_rewrites_a_corrupt_file(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        index = UUIDIndex(tmp_path)
        x, y = uuid4(), uuid4()
        index.add(x, notes_dir / f"x-{x}.md")
        index.index_path.write_text('{"uuid_paths": {')

        index.add(y, notes_dir / f"y-{y}.md")

        assert set(_on_disk(tmp_path)) == {str(x), str(y)}


class TestUUIDIndexOwedRebuild:
    """A rebuild whose lock or write failed is retried, never silently dropped."""

    def _with_unindexed_note(self, tmp_path: Path) -> tuple[UUIDIndex, Path, UUID, UUID, Path]:
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)
        seed_id = uuid4()
        seed_path = notes_dir / f"seed-{seed_id}.md"
        seed_path.touch()
        index = UUIDIndex(tmp_path)
        index.add(seed_id, seed_path)
        # A note file the index does not know about, e.g. restored by git
        stray_id = uuid4()
        stray_path = notes_dir / f"stray-{stray_id}.md"
        stray_path.touch()
        return index, notes_dir, seed_id, stray_id, stray_path

    def test_failed_rebuild_write_survives_an_unrelated_add(self, tmp_path, monkeypatch):
        """Regression: the next add re-read the stale file and dropped the rebuild's repair."""
        index, notes_dir, seed_id, stray_id, stray_path = self._with_unindexed_note(tmp_path)
        with monkeypatch.context() as m:
            m.setattr(os, "replace", _failing_replace)
            index.rebuild()
        assert index.get_path(stray_id) == stray_path
        assert set(_on_disk(tmp_path)) == {str(seed_id)}

        other_id = uuid4()
        other_path = notes_dir / f"other-{other_id}.md"
        other_path.touch()
        index.add(other_id, other_path)

        assert index.get_path(stray_id) == stray_path
        fresh = UUIDIndex(tmp_path)
        fresh.load()
        assert fresh.get_path(stray_id) == stray_path
        assert fresh.get_path(other_id) == other_path
        assert index._rebuild_owed is False

    def test_failed_rebuild_write_is_retried_by_save_with_a_fresh_scan(self, tmp_path, monkeypatch):
        index, notes_dir, seed_id, stray_id, stray_path = self._with_unindexed_note(tmp_path)
        with monkeypatch.context() as m:
            m.setattr(os, "replace", _failing_replace)
            index.rebuild()
        # The filesystem moves on before the retry, which must scan again
        # rather than write the earlier scan
        stray_path.unlink()
        late_id = uuid4()
        late_path = notes_dir / f"late-{late_id}.md"
        late_path.touch()

        index.save()

        assert set(_on_disk(tmp_path)) == {str(seed_id), str(late_id)}
        assert index.get_path(late_id) == late_path
        assert index._rebuild_owed is False

    def test_failed_rebuild_lock_is_retried_when_another_process_writes(
        self, tmp_path, monkeypatch
    ):
        index, notes_dir, seed_id, stray_id, stray_path = self._with_unindexed_note(tmp_path)
        with monkeypatch.context() as m:
            m.setattr("mcp_notes.storage.uuid_index.file_lock", _unavailable_lock)
            index.rebuild()
        assert index._rebuild_owed is True
        assert index.get_path(stray_id) == stray_path
        # Another process, which never saw the stray note, writes the stale map
        peer = UUIDIndex(tmp_path)
        peer.load()
        peer_id = uuid4()
        peer_path = notes_dir / f"peer-{peer_id}.md"
        peer_path.touch()
        peer.add(peer_id, peer_path)
        assert str(stray_id) not in _on_disk(tmp_path)

        # The miss retries the owed rebuild instead of adopting the stale file
        assert index.get_path(peer_id) == peer_path
        assert index.get_path(stray_id) == stray_path
        assert index._rebuild_owed is False
        assert set(_on_disk(tmp_path)) == {str(seed_id), str(stray_id), str(peer_id)}

    def test_stale_file_never_replaces_an_owed_rebuild(self, tmp_path, monkeypatch):
        """While retries keep failing, a miss and load() both keep the in-memory scan."""
        index, notes_dir, seed_id, stray_id, stray_path = self._with_unindexed_note(tmp_path)
        with monkeypatch.context() as m:
            m.setattr(os, "replace", _failing_replace)
            index.rebuild()
        peer = UUIDIndex(tmp_path)
        peer.load()
        peer_id = uuid4()
        peer_path = notes_dir / f"peer-{peer_id}.md"
        peer_path.touch()
        peer.add(peer_id, peer_path)
        real_commit = index._commit
        retries = []

        def counting_commit(changes):
            retries.append(changes)
            real_commit(changes)

        monkeypatch.setattr(index, "_commit", counting_commit)

        with monkeypatch.context() as m:
            m.setattr(os, "replace", _failing_replace)
            assert index.get_path(uuid4()) is None
            assert index.get_path(stray_id) == stray_path
            # One retry per version of the file, not one per miss
            assert index.get_path(uuid4()) is None
            assert len(retries) == 1
            index.load()
            assert index.get_path(stray_id) == stray_path
            assert index._rebuild_owed is True

        index.save()

        assert set(_on_disk(tmp_path)) == {str(seed_id), str(stray_id), str(peer_id)}
        assert index._rebuild_owed is False


class TestUUIDIndexMultiProcess:
    """Real separate processes sharing one notes directory."""

    def test_concurrent_note_creation_loses_nothing(self, tmp_path):
        seed = NoteStore(notes_dir=tmp_path).create(title="Seed", content="seed")
        barrier = _SPAWN.Barrier(2)
        outputs = [tmp_path / f"created-{n}.json" for n in range(2)]
        workers = [
            _SPAWN.Process(
                target=_create_notes_in_child, args=(str(tmp_path), str(out), barrier, 25)
            )
            for out in outputs
        ]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=120)
            assert [worker.exitcode for worker in workers] == [0, 0]
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()

        created = [UUID(i) for out in outputs for i in json.loads(out.read_text())]
        assert len(set(created)) == 50
        fresh = UUIDIndex(tmp_path)
        fresh.load()
        assert fresh.count() == 51
        for note_id in [seed.id, *created]:
            path = fresh.get_path(note_id)
            assert path is not None and path.exists()

    def test_reader_never_sees_a_partial_file(self, tmp_path):
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        # Large enough that a non-atomic rewrite takes several write() calls
        entries = {str(uuid4()): f"category/note-{i}.md" for i in range(500)}
        (index_dir / INDEX_FILENAME).write_text(json.dumps({"uuid_paths": entries}))
        started = _SPAWN.Event()
        writer = _SPAWN.Process(target=_churn_index_in_child, args=(str(tmp_path), started, 200))
        failures = []
        try:
            writer.start()
            assert started.wait(timeout=60)
            while True:
                alive = writer.is_alive()
                try:
                    data = json.loads((index_dir / INDEX_FILENAME).read_text(encoding="utf-8"))
                    assert set(entries) <= set(data["uuid_paths"])
                except ValueError as e:
                    failures.append(e)
                if not alive:
                    break
            writer.join(timeout=60)
            assert writer.exitcode == 0
        finally:
            if writer.is_alive():
                writer.kill()

        assert failures == []
