"""max_note_size_kb bounds every read of a note file, not just read()."""

import logging
from uuid import uuid4

import pytest

from mcp_notes.indexing import migration as migration_module
from mcp_notes.indexing.migration import NotesMigration
from mcp_notes.settings import settings
from mcp_notes.storage import filesystem
from mcp_notes.storage.filesystem import NoteStore, NoteTooLargeError, read_note_text

LIMIT_KB = 1
LIMIT = LIMIT_KB * 1024


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_note_size_kb", LIMIT_KB)


@pytest.fixture
def store(tmp_path):
    return NoteStore(tmp_path / "source")


def oversize(store, title="Huge", extra=10):
    """Create a note, then grow its file past the limit as an outside edit would."""
    note = store.create(title, "small body")
    path = store.get_note_path(note.id)
    with path.open("a", encoding="utf-8") as f:
        f.write("x" * (LIMIT + extra))
    return note, path


class TestReadNoteText:
    def test_accepts_a_file_at_the_limit(self, tmp_path, small_limit):
        path = tmp_path / "at-limit.md"
        path.write_bytes(b"a" * LIMIT)
        assert read_note_text(path) == "a" * LIMIT

    def test_refuses_one_byte_over_the_limit(self, tmp_path, small_limit):
        path = tmp_path / "over.md"
        path.write_bytes(b"a" * (LIMIT + 1))
        with pytest.raises(NoteTooLargeError) as caught:
            read_note_text(path)
        assert caught.value.size_kb == LIMIT_KB + 1
        assert caught.value.max_kb == LIMIT_KB

    def test_never_reads_past_the_limit_when_the_file_grows(
        self, tmp_path, small_limit, monkeypatch
    ):
        # The size is checked first; a file that grows after that check must
        # still be refused, and no more than limit + 1 bytes may be read.
        path = tmp_path / "growing.md"
        path.write_bytes(b"a" * (LIMIT * 4))
        real_fstat = filesystem.os.fstat
        reads = []

        class Stat:
            def __init__(self, stat):
                self._stat = stat
                self.st_size = 10

            def __getattr__(self, name):
                return getattr(self._stat, name)

        monkeypatch.setattr(filesystem.os, "fstat", lambda fd: Stat(real_fstat(fd)))
        real_open = type(path).open

        def tracking_open(self, *args, **kwargs):
            handle = real_open(self, *args, **kwargs)
            real_read = handle.read

            def read(size=-1):
                data = real_read(size)
                reads.append(len(data))
                return data

            handle.read = read
            return handle

        monkeypatch.setattr(type(path), "open", tracking_open)
        with pytest.raises(NoteTooLargeError):
            read_note_text(path)
        assert reads == [LIMIT + 1]

    def test_translates_newlines_like_text_mode(self, tmp_path):
        path = tmp_path / "newlines.md"
        path.write_bytes(b"one\r\ntwo\rthree\n")
        assert read_note_text(path) == "one\ntwo\nthree\n"


class TestBulkScans:
    def test_list_all_records_an_oversized_note_and_lists_the_rest(self, store, small_limit):
        kept = store.create("Kept", "fine")
        _, path = oversize(store)

        summaries = store.list_all()

        assert [summary.id for summary in summaries] == [kept.id]
        errors = store.get_parse_errors()
        assert [error.path for error in errors] == [path]
        assert isinstance(errors[0].error, NoteTooLargeError)

    def test_iter_all_records_an_oversized_note_and_yields_the_rest(self, store, small_limit):
        kept = store.create("Kept", "fine")
        _, path = oversize(store)

        notes = list(store.iter_all())

        assert [parsed.id for parsed, _ in notes] == [kept.id]
        assert [error.path for error in store.get_parse_errors()] == [path]

    def test_get_summary_refuses_an_oversized_note(self, store, small_limit):
        note, _ = oversize(store)
        with pytest.raises(NoteTooLargeError):
            store.get_summary(note.id)

    def test_update_refuses_an_oversized_note_without_changing_it(self, store, small_limit):
        note, path = oversize(store)
        before = path.read_bytes()
        with pytest.raises(NoteTooLargeError):
            store.update(note.id, title="Renamed")
        assert path.read_bytes() == before

    def test_note_ids_on_disk_still_identifies_an_oversized_hand_named_note(
        self, store, small_limit
    ):
        # A file without a UUID in its name is identified from its frontmatter,
        # which a bounded prefix still contains.
        note, path = oversize(store)
        renamed = path.with_name("hand-named.md")
        path.rename(renamed)
        assert store.note_ids_on_disk() == {note.id}


def bury_id(path, note_id):
    """Rewrite a note so its id comes after a title longer than the size limit."""
    path.write_text(
        "---\n"
        f"title: {'T' * (LIMIT * 2)}\n"
        f"id: {note_id}\n"
        "created: '2026-01-01T00:00:00+00:00'\n"
        "modified: '2026-01-01T00:00:00+00:00'\n"
        "---\n\nbody\n",
        encoding="utf-8",
    )


class TestDeletedNoteCheck:
    def test_an_oversized_note_that_cannot_be_identified_fails_the_scan(self, store, small_limit):
        # YAML does not require id first. If the id lies past the bounded
        # prefix, the scan must not report the note as absent.
        note = store.create("Buried", "body")
        path = store.get_note_path(note.id).rename(store.notes_dir / "hand-named.md")
        bury_id(path, note.id)
        with pytest.raises(OSError, match="Cannot identify"):
            store.note_ids_on_disk()


class TestMigrationSourceScan:
    def test_snapshot_skips_an_oversized_note(self, store, small_limit, monkeypatch, caplog):
        kept = store.create("Kept", "fine")
        note, _ = oversize(store)
        indexer = type("Indexer", (), {"note_store": store})()
        caplog.set_level(logging.WARNING, logger=migration_module.__name__)

        snapshot = NotesMigration(indexer)._snapshot()

        assert set(snapshot) == {kept.id}
        assert note.id not in snapshot
        assert "Skipping oversized note source" in caplog.text


def test_unrelated_uuid_named_files_are_unaffected(store, small_limit):
    # Sanity: a normal note under the limit reads and lists as before.
    note = store.create(f"Note {uuid4().hex[:6]}", "body")
    assert store.read(note.id).id == note.id
    assert [summary.id for summary in store.list_all()] == [note.id]
