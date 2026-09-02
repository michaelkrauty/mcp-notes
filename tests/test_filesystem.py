"""Tests for filesystem note storage."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from mcp_notes.storage.filesystem import (
    NoteNotFoundError,
    NoteStore,
    _atomic_write,
)
from mcp_notes.storage.parser import parse_note, serialize_note


class TestAtomicWrite:
    """Tests for atomic file write."""

    def test_writes_content(self, tmp_path):
        """Writes content to file."""
        path = tmp_path / "test.txt"
        _atomic_write(path, "Hello, world!")

        assert path.exists()
        assert path.read_text() == "Hello, world!"

    def test_atomic_replaces_existing(self, tmp_path):
        """Atomically replaces existing file."""
        path = tmp_path / "test.txt"
        path.write_text("old content")

        _atomic_write(path, "new content")

        assert path.read_text() == "new content"

    def test_cleans_up_on_error(self, tmp_path, monkeypatch):
        """Cleans up temp file on write error."""
        path = tmp_path / "test.txt"

        # Make os.replace fail
        def mock_replace(src, dst):
            raise OSError("Mock error")

        monkeypatch.setattr("os.replace", mock_replace)

        with pytest.raises(IOError):
            _atomic_write(path, "content")

        # No leftover temp files
        assert len(list(tmp_path.glob("*.tmp"))) == 0


class TestNoteStoreInit:
    """Tests for NoteStore initialization."""

    def test_default_dir(self, monkeypatch, tmp_path):
        """Uses settings dir by default."""
        from mcp_notes.settings import settings

        monkeypatch.setattr(settings, "dir", tmp_path)

        store = NoteStore()

        assert store.base_dir == tmp_path
        assert store.notes_dir == tmp_path / "notes"

    def test_custom_dir(self, tmp_path):
        """Accepts custom directory."""
        store = NoteStore(notes_dir=tmp_path)

        assert store.base_dir == tmp_path


class TestNoteStoreEnsureDirectories:
    """Tests for directory creation."""

    def test_creates_directories(self, tmp_path):
        """Creates notes and .index directories."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        assert (tmp_path / "notes").exists()
        assert (tmp_path / ".index").exists()

    def test_idempotent(self, tmp_path):
        """Can be called multiple times."""
        store = NoteStore(notes_dir=tmp_path)

        store.ensure_directories()
        store.ensure_directories()

        assert (tmp_path / "notes").exists()


class TestNoteStoreExists:
    """Tests for note existence check."""

    def test_exists_true(self, tmp_path):
        """Returns True for existing note."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Use store.create() to properly register in UUID index
        note = store.create(title="Test", content="Content")

        assert store.exists(note.id) is True

    def test_exists_false(self, tmp_path):
        """Returns False for non-existing note."""
        store = NoteStore(notes_dir=tmp_path)

        assert store.exists(uuid4()) is False


class TestNoteStoreCreate:
    """Tests for note creation."""

    def test_create_basic(self, tmp_path):
        """Creates a basic note."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Test Note",
            content="This is the content.",
        )

        assert note.title == "Test Note"
        assert note.id is not None
        assert note.category is None
        assert store.exists(note.id)

    def test_create_with_tags(self, tmp_path):
        """Creates note with tags."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Tagged Note",
            content="Content",
            tags=["important", "work"],
        )

        assert "important" in note.tags
        assert "work" in note.tags

    def test_create_normalizes_tags(self, tmp_path):
        """Tags are normalized (lowercase, trim, spaces to hyphens)."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Tagged Note",
            content="Content",
            tags=["  Work Project  ", "IMPORTANT"],
        )

        assert "work-project" in note.tags
        assert "important" in note.tags

    def test_create_dedups_tags_that_collapse(self, tmp_path):
        """Distinct raw tags that canonicalize to the same stored form are
        deduplicated on write (mirroring the read path), so the returned/stored
        tags do not contain duplicates that disagree with a later read."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Dup Tags",
            content="Content",
            tags=["My Tag", "my-tag", "FOO", "foo"],
        )

        assert note.tags == ["my-tag", "foo"]

    def test_create_returns_canonical_path_category(self, tmp_path):
        """Create returns the same canonical category as its path and later reads."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Categorized Note",
            content="Content",
            category="Work & Projects/Client (Main)",
        )

        assert note.category == "work-projects/client-main"
        assert store.read(note.id).category == note.category
        note_path = store.get_note_path(note.id)
        assert note_path is not None
        assert note_path.parent.relative_to(store.notes_dir).as_posix() == note.category

    def test_create_with_custom_id(self, tmp_path):
        """Creates note with custom UUID."""
        store = NoteStore(notes_dir=tmp_path)
        custom_id = uuid4()

        note = store.create(
            title="Custom ID Note",
            content="Content",
            note_id=custom_id,
        )

        assert note.id == custom_id

    def test_create_extracts_links(self, tmp_path):
        """Extracts inline links from content."""
        store = NoteStore(notes_dir=tmp_path)
        target_id = uuid4()

        note = store.create(
            title="Linked Note",
            content=f"See [[{target_id}]] for details.",
        )

        assert target_id in note.links

    def test_create_timestamps(self, tmp_path):
        """Sets created and modified timestamps."""
        store = NoteStore(notes_dir=tmp_path)

        before = datetime.now(UTC)
        note = store.create(title="Timestamped", content="Content")
        after = datetime.now(UTC)

        assert before <= note.created <= after
        assert note.created == note.modified


class TestNoteStoreRead:
    """Tests for note reading."""

    def test_read_existing(self, tmp_path):
        """Reads an existing note."""
        store = NoteStore(notes_dir=tmp_path)

        created = store.create(
            title="Test Note",
            content="Content here",
            tags=["test"],
        )

        read = store.read(created.id)

        assert read.id == created.id
        assert read.title == "Test Note"
        assert "test" in read.tags

    def test_read_nonexistent(self, tmp_path):
        """Raises error for non-existent note."""
        store = NoteStore(notes_dir=tmp_path)

        with pytest.raises(NoteNotFoundError):
            store.read(uuid4())


class TestNoteStoreUpdate:
    """Tests for note updates."""

    def test_update_title(self, tmp_path):
        """Updates note title."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Original", content="Content")
        updated = store.update(note.id, title="Updated Title")

        assert updated.title == "Updated Title"

    def test_update_content(self, tmp_path):
        """Updates note content."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Original content")
        updated = store.update(note.id, content="New content")

        assert "New content" in updated.content

    def test_update_preserves_frontmatter_only_links(self, tmp_path):
        """A content-less update must not drop frontmatter `links` that have no
        inline [[uuid]] counterpart. The read side treats frontmatter links as
        live edges, so a hand-edited/imported note's links must survive a
        tags/title/category update (which rebuilds links from the body only)."""
        store = NoteStore(notes_dir=tmp_path)
        note = store.create(title="B", content="a body with no inline link")
        target = uuid4()

        # Hand-edit B's file: add a frontmatter link to `target`, body unchanged
        # (no inline [[target]]), as an imported/hand-edited note would have.
        path = store.get_note_path(note.id)
        parsed = parse_note(path.read_text(encoding="utf-8"))
        path.write_text(
            serialize_note(
                note_id=note.id,
                title=parsed.title,
                body=parsed.body,
                tags=parsed.tags or None,
                category=None,
                links=[target],
                created=parsed.created,
                modified=parsed.modified,
            ),
            encoding="utf-8",
        )

        # A content-less update (tags only) must preserve the frontmatter link.
        store.update(note.id, tags=["x"])

        reparsed = parse_note(store.get_note_path(note.id).read_text(encoding="utf-8"))
        assert target in reparsed.links

    def test_update_tags(self, tmp_path):
        """Updates note tags."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Content", tags=["old"])
        updated = store.update(note.id, tags=["new", "tags"])

        assert "new" in updated.tags
        assert "tags" in updated.tags
        assert "old" not in updated.tags

    def test_update_clears_tags(self, tmp_path):
        """Clears tags when empty list provided."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Content", tags=["tag"])
        updated = store.update(note.id, tags=[])

        assert updated.tags == []

    def test_update_returns_canonical_path_category(self, tmp_path):
        """A category-changing update returns the category used by reads and indexing."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Content", category="old")
        updated = store.update(note.id, category="Work & Projects/Client (Main)")

        assert updated.category == "work-projects/client-main"
        assert store.read(note.id).category == updated.category
        note_path = store.get_note_path(note.id)
        assert note_path is not None
        assert note_path.parent.relative_to(store.notes_dir).as_posix() == updated.category

    def test_update_clears_category(self, tmp_path):
        """Clears category when empty string provided."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Content", category="cat")
        updated = store.update(note.id, category="")

        assert updated.category is None

    def test_update_modified_timestamp(self, tmp_path):
        """Updates modified timestamp."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="Title", content="Content")
        original_modified = note.modified

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        updated = store.update(note.id, content="New content")

        assert updated.modified > original_modified
        assert updated.created == note.created  # Created unchanged

    def test_update_nonexistent(self, tmp_path):
        """Raises error for non-existent note."""
        store = NoteStore(notes_dir=tmp_path)

        with pytest.raises(NoteNotFoundError):
            store.update(uuid4(), title="New")

    def test_update_preserves_unmodified(self, tmp_path):
        """Preserves fields not being updated."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Original",
            content="Content",
            tags=["tag"],
            category="cat",
        )
        updated = store.update(note.id, title="New Title")

        assert updated.title == "New Title"
        assert "tag" in updated.tags
        assert updated.category == "cat"


class TestNoteStoreDelete:
    """Tests for note deletion."""

    def test_delete_existing(self, tmp_path):
        """Deletes an existing note."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(title="To Delete", content="Content")
        result = store.delete(note.id)

        assert result is True
        assert not store.exists(note.id)

    def test_delete_nonexistent(self, tmp_path):
        """Raises error for non-existent note."""
        store = NoteStore(notes_dir=tmp_path)

        with pytest.raises(NoteNotFoundError):
            store.delete(uuid4())


class TestNoteStoreListAll:
    """Tests for listing all notes."""

    def test_list_empty(self, tmp_path):
        """Returns empty list when no notes."""
        store = NoteStore(notes_dir=tmp_path)

        assert store.list_all() == []

    def test_list_all_notes(self, tmp_path):
        """Lists all notes as summaries."""
        store = NoteStore(notes_dir=tmp_path)

        store.create(title="Note 1", content="Content 1")
        store.create(title="Note 2", content="Content 2")
        store.create(title="Note 3", content="Content 3")

        summaries = store.list_all()

        assert len(summaries) == 3
        titles = [s.title for s in summaries]
        assert "Note 1" in titles
        assert "Note 2" in titles
        assert "Note 3" in titles

    def test_list_includes_metadata(self, tmp_path):
        """Summaries include metadata."""
        store = NoteStore(notes_dir=tmp_path)

        store.create(
            title="Detailed Note",
            content="Some content here",
            tags=["test"],
            category="work",
        )

        summaries = store.list_all()

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.title == "Detailed Note"
        assert "test" in summary.tags
        assert summary.category == "work"
        assert summary.excerpt != ""


class TestNoteStoreIterAll:
    """Tests for iterating all notes."""

    def test_iter_empty(self, tmp_path):
        """Returns empty iterator when no notes."""
        store = NoteStore(notes_dir=tmp_path)

        assert list(store.iter_all()) == []

    def test_iter_all_notes(self, tmp_path):
        """Iterates over all parsed notes."""
        store = NoteStore(notes_dir=tmp_path)

        store.create(title="Note 1", content="Content 1")
        store.create(title="Note 2", content="Content 2")

        parsed_notes = list(store.iter_all())

        assert len(parsed_notes) == 2


class TestNoteStoreGetSummary:
    """Tests for getting single note summary."""

    def test_get_summary(self, tmp_path):
        """Gets summary for a single note."""
        store = NoteStore(notes_dir=tmp_path)

        note = store.create(
            title="Summary Test",
            content="This is the content for the summary.",
            tags=["summary"],
        )

        summary = store.get_summary(note.id)

        assert summary.id == note.id
        assert summary.title == "Summary Test"
        assert "summary" in summary.tags

    def test_get_summary_nonexistent(self, tmp_path):
        """Raises error for non-existent note."""
        store = NoteStore(notes_dir=tmp_path)

        with pytest.raises(NoteNotFoundError):
            store.get_summary(uuid4())

    def test_get_summary_stale_index_raises_notnotfound(self, tmp_path):
        """A UUID still in the in-memory index whose file was removed
        out-of-band raises NoteNotFoundError (not a bare FileNotFoundError),
        mirroring read(). Otherwise a dangling outgoing link crashes the whole
        get_note_links view instead of being reported as broken."""
        store = NoteStore(notes_dir=tmp_path)
        note = store.create(title="Gone", content="x")

        # Remove the file directly, leaving the UUID in the index (a git
        # pull/checkout or external delete with a lagging index).
        store._note_path(note.id).unlink()

        with pytest.raises(NoteNotFoundError):
            store.get_summary(note.id)

    def test_get_summary_moved_file_repoints_via_rebuild(self, tmp_path):
        """A file moved out-of-band (UUID unchanged) is re-resolved via the
        index rebuild and read normally, not falsely reported missing."""
        store = NoteStore(notes_dir=tmp_path)
        note = store.create(title="Moved", content="x")
        old_path = store._note_path(note.id)

        # Move the file out-of-band; the in-memory index still points at the
        # old path until rebuild (a git checkout/pull moving a file).
        new_dir = store.notes_dir / "moved-here"
        new_dir.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_dir / old_path.name)

        summary = store.get_summary(note.id)
        assert summary.id == note.id
        assert summary.title == "Moved"


class TestNoteStoreCount:
    """Tests for counting notes."""

    def test_count_empty(self, tmp_path):
        """Returns 0 when no notes."""
        store = NoteStore(notes_dir=tmp_path)

        assert store.count() == 0

    def test_count_notes(self, tmp_path):
        """Counts all notes."""
        store = NoteStore(notes_dir=tmp_path)

        store.create(title="Note 1", content="Content 1")
        store.create(title="Note 2", content="Content 2")
        store.create(title="Note 3", content="Content 3")

        assert store.count() == 3

    def test_count_after_delete(self, tmp_path):
        """Count updates after deletion."""
        store = NoteStore(notes_dir=tmp_path)

        note1 = store.create(title="Note 1", content="Content 1")
        store.create(title="Note 2", content="Content 2")

        assert store.count() == 2

        store.delete(note1.id)

        assert store.count() == 1


class TestAtomicWriteCleanupError:
    """Tests for atomic write cleanup on error (lines 42-43)."""

    def test_unlink_oserror_suppressed(self, tmp_path, monkeypatch):
        """OSError during temp file cleanup is suppressed (lines 42-43)."""
        import os as os_module

        path = tmp_path / "test.txt"

        def mock_replace(src, dst):
            raise OSError("Replace failed")

        def mock_unlink(p):
            raise OSError("Unlink failed")

        monkeypatch.setattr(os_module, "replace", mock_replace)
        monkeypatch.setattr(os_module, "unlink", mock_unlink)

        # Should raise the original IOError, not the OSError from unlink
        with pytest.raises(IOError, match="Replace failed"):
            _atomic_write(path, "content")


class TestListAllParseError:
    """Tests for parse error handling in list_all (lines 292-293)."""

    def test_list_all_skips_malformed_notes(self, tmp_path):
        """list_all logs warning and skips malformed notes (lines 292-293)."""
        store = NoteStore(notes_dir=tmp_path)

        # Create a valid note
        store.create(title="Valid Note", content="Valid content")

        # Create a malformed note file (no valid YAML frontmatter)
        notes_dir = tmp_path / "notes"
        malformed_path = notes_dir / "malformed.md"
        malformed_path.write_text("not valid yaml frontmatter")

        # list_all should skip the malformed note
        summaries = store.list_all()

        # Should have 1 valid note, malformed one skipped
        assert len(summaries) == 1
        assert summaries[0].title == "Valid Note"


class TestIterAllParseError:
    """Tests for parse error handling in iter_all (lines 311-312)."""

    def test_iter_all_skips_malformed_notes(self, tmp_path):
        """iter_all logs warning and skips malformed notes (lines 311-312)."""
        store = NoteStore(notes_dir=tmp_path)

        # Create a valid note
        store.create(title="Valid Note", content="Valid content")

        # Create a malformed note file
        notes_dir = tmp_path / "notes"
        malformed_path = notes_dir / "broken.md"
        malformed_path.write_text("this is not a valid note format")

        # iter_all should skip the malformed note
        # iter_all now yields (ParsedNote, category) tuples
        parsed_notes = [(parsed, cat) for parsed, cat in store.iter_all()]

        # Should have 1 valid note
        assert len(parsed_notes) == 1
        assert parsed_notes[0][0].title == "Valid Note"


class TestToSummaryExcerptEdge:
    """Tests for excerpt word boundary logic (lines 319-323)."""

    def test_excerpt_no_space_in_first_70_percent(self, tmp_path, monkeypatch):
        """Excerpt truncation when no space in first 70% (lines 319-323)."""
        from mcp_notes.settings import settings

        # Set short excerpt length for testing
        monkeypatch.setattr(settings, "excerpt_length", 20)

        store = NoteStore(notes_dir=tmp_path)

        # Create note with content that has no space in first 70% of excerpt
        # 70% of 20 = 14 chars. Space must be after position 14
        # "aaaaaaaaaaaaaaa bbb" = 19 chars, space at position 15
        note = store.create(
            title="Edge Case",
            content="aaaaaaaaaaaaaaa bbb continued text beyond excerpt",
        )

        summary = store.get_summary(note.id)

        # Last space at position 15 > 14 (70% of 20), so should use word boundary
        assert summary.excerpt.endswith("...")

    def test_excerpt_no_space_falls_back(self, tmp_path, monkeypatch):
        """Excerpt falls back to hard cut when no suitable space (lines 319-323)."""
        from mcp_notes.settings import settings

        # Set short excerpt length for testing
        monkeypatch.setattr(settings, "excerpt_length", 20)

        store = NoteStore(notes_dir=tmp_path)

        # Create note with content that has space very early (before 70%)
        # "aa bbbbbbbbbbbbbbbbbbbbbbb" - space at position 2, which is < 14 (70% of 20)
        note = store.create(
            title="Hard Cut",
            content="aa bbbbbbbbbbbbbbbbbbbbbbb continued text beyond excerpt",
        )

        summary = store.get_summary(note.id)

        # Space at position 2 < 14 (70%), so should hard cut at excerpt_length
        assert summary.excerpt.endswith("...")
        # Should be 20 chars + "..."
        assert len(summary.excerpt) == 23  # 20 + 3 for "..."


class TestSymlinkSecurity:
    """Tests for symlink traversal protection."""

    def test_symlink_to_external_file_blocked(self, tmp_path):
        """Symlink pointing outside notes_dir is blocked."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create external file
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_file = external_dir / "secret.md"
        external_file.write_text("""---
id: 12345678-1234-1234-1234-123456789012
title: Secret
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---
Secret content""")

        # Create symlink in notes dir pointing to external
        notes_dir = tmp_path / "notes"
        symlink_path = notes_dir / "evil-link-12345678-1234-1234-1234-123456789012.md"
        symlink_path.symlink_to(external_file)

        # list_all should skip the symlink
        summaries = store.list_all()
        assert len(summaries) == 0

        # iter_all should skip the symlink
        parsed = list(store.iter_all())
        assert len(parsed) == 0

        # count should skip the symlink
        assert store.count() == 0

    def test_symlink_within_notes_dir_blocked(self, tmp_path):
        """Symlink within notes_dir is still blocked (all symlinks blocked)."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create a real note
        note = store.create(title="Real Note", content="Real content")

        # Get path to real note
        from mcp_notes.storage.slugify import build_filename

        real_path = tmp_path / "notes" / build_filename("Real Note", note.id)

        # Create symlink pointing to the real note
        notes_dir = tmp_path / "notes"
        symlink_path = notes_dir / "alias-12345678-1234-1234-1234-123456789012.md"
        symlink_path.symlink_to(real_path)

        # Should only count the real note, not the symlink
        assert store.count() == 1

        # list_all should only return the real note
        summaries = store.list_all()
        assert len(summaries) == 1
        assert summaries[0].id == note.id

    def test_broken_symlink_skipped(self, tmp_path):
        """Broken symlinks are skipped gracefully."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create a real note first
        note = store.create(title="Valid Note", content="Valid content")

        # Create a broken symlink (target doesn't exist)
        notes_dir = tmp_path / "notes"
        broken_symlink = notes_dir / "broken-12345678-1234-1234-1234-123456789012.md"
        broken_symlink.symlink_to("/nonexistent/path/to/file.md")

        # Operations should succeed, skipping the broken symlink
        summaries = store.list_all()
        assert len(summaries) == 1
        assert summaries[0].title == "Valid Note"

        assert store.count() == 1

    def test_directory_symlink_blocked(self, tmp_path):
        """Symlinked directories are blocked."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create external directory with notes
        external_dir = tmp_path.parent / "external_notes"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_note = external_dir / "external-12345678-1234-1234-1234-123456789012.md"
        external_note.write_text("""---
id: 12345678-1234-1234-1234-123456789012
title: External Note
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---
External content""")

        # Create symlink directory inside notes_dir
        notes_dir = tmp_path / "notes"
        symlink_dir = notes_dir / "linked_folder"
        symlink_dir.symlink_to(external_dir)

        # Notes in symlinked directory should not be found
        summaries = store.list_all()
        assert len(summaries) == 0

    def test_symlink_chain_blocked(self, tmp_path):
        """Chain of symlinks is blocked."""
        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create external file
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_file = external_dir / "target.md"
        external_file.write_text("""---
id: 12345678-1234-1234-1234-123456789012
title: Target
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---
Content""")

        # Create intermediate symlink
        intermediate_link = tmp_path / "intermediate"
        intermediate_link.symlink_to(external_file)

        # Create symlink in notes pointing to intermediate
        notes_dir = tmp_path / "notes"
        chain_link = notes_dir / "chain-12345678-1234-1234-1234-123456789012.md"
        chain_link.symlink_to(intermediate_link)

        # Should block the symlink chain
        summaries = store.list_all()
        assert len(summaries) == 0

    def test_note_read_rejects_symlink(self, tmp_path):
        """Reading a note by ID rejects symlinks with PathTraversalError."""
        from mcp_notes.storage.filesystem import PathTraversalError
        from uuid import UUID

        store = NoteStore(notes_dir=tmp_path)
        store.ensure_directories()

        # Create external file
        external_dir = tmp_path.parent / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_file = external_dir / "secret.md"
        note_id = UUID("12345678-1234-1234-1234-123456789012")
        external_file.write_text(f"""---
id: {note_id}
title: Secret
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---
Secret content""")

        # Create symlink
        notes_dir = tmp_path / "notes"
        symlink_path = notes_dir / f"evil-{note_id}.md"
        symlink_path.symlink_to(external_file)

        # Manually add to UUID index to simulate corruption
        store.uuid_index.add(note_id, symlink_path)
        store.uuid_index.save()

        # Reading should fail - symlink is blocked with PathTraversalError
        with pytest.raises(PathTraversalError, match="Symlinks not allowed"):
            store.read(note_id)
