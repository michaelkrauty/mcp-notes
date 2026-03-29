"""Tests for UUID-to-path index."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mcp_notes.storage.uuid_index import INDEX_FILENAME, UUIDIndex


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
        assert index._dirty is False


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

    def test_save_clears_dirty(self, tmp_path):
        """Save clears dirty flag."""
        index = UUIDIndex(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir(parents=True)

        note_id = uuid4()
        note_path = notes_dir / "test.md"
        note_path.touch()
        index.add(note_id, note_path)

        assert index._dirty is True
        index.save()
        assert index._dirty is False


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
        assert index._dirty is True

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

        index._dirty = False  # Reset for test
        index.remove(note_id)

        assert not index.exists(note_id)
        assert index._dirty is True

    def test_remove_nonexistent(self, tmp_path):
        """Remove nonexistent entry is safe."""
        index = UUIDIndex(tmp_path)
        note_id = uuid4()

        # Should not raise
        index.remove(note_id)
        assert index._dirty is False


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
