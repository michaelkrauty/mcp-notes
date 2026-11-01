"""index_all removes notes whose markdown file disappeared outside the service.

A git checkout, restore or manual delete removes a note file without telling
the index. A note is gone only when no file under the notes directory is its
own, so a file that exists but no longer parses keeps its points, and a scan
that cannot see the whole directory removes nothing. Pass 1 already leaves the
gone notes out of the vocabulary registration, so removing their points makes
the contribution describe what is stored again.
"""

import os
from collections import Counter
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_core import EmbeddingClient, QdrantStorage
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing import indexer as indexer_module
from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID, NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import NoteStore

MARKER = "notes_root"
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def isolated_embedder(capacity, namespace="vanished"):
    client = EmbeddingClient(
        model="isolated-model",
        dim=128,
        profile="raw",
        max_text_chars=capacity,
        cache_namespace=namespace,
    )
    client.resolve_identity = AsyncMock(return_value=client.configured_identity())
    client._embed_prepared_batch = AsyncMock(
        side_effect=lambda texts: [[1.0] + [0.0] * 127 for _ in texts]
    )
    return client


def paragraphs(prefix, count):
    """Distinct paragraphs, each holding terms no other paragraph uses."""
    return "\n\n".join(f"{prefix}{word} filler " * 14 for word in WORDS[:count])


@pytest.fixture
async def notes(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_settings, "collection_name", None)
    monkeypatch.setattr(vector_settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "max_chunk_chars", 100000)
    monkeypatch.setattr(settings, "section_overlap_chars", 0)
    store = NoteStore(tmp_path / "source")
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=128)
    storage._client = AsyncQdrantClient(location=":memory:")
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    embedder = isolated_embedder(256)
    indexer = NoteIndexer(store, storage, embedder, vocab)
    yield indexer, vocab
    await storage.close()
    await embedder.close()
    vocab.close()


def contribution(vocab):
    """The notes document count and every non-zero document frequency."""
    doc_freq = {token: freq for token, freq in vocab._get_doc_freq().items() if freq}
    return vocab.get_codebase_doc_count(NOTES_CODEBASE_ID), doc_freq


def registration(token_sets):
    """What register_codebase records for these documents on a fresh database."""
    return len(token_sets), dict(Counter(token for tokens in token_sets for token in tokens))


def source_token_sets(indexer):
    """One set per summary and chunk of every source note, as index_all builds them."""
    vocab = indexer.global_vocab
    sets = []
    for parsed, _ in indexer.note_store.iter_all():
        sets.append(set(vocab.tokenize(generate_note_summary(parsed))))
        sets.extend(set(vocab.tokenize(c.content)) for c in chunk_note(parsed, indexer.embedder))
    return sets


async def all_points(indexer, note_id=None, collection=None):
    """Every note and chunk point, derived fragments included, optionally of one note."""
    if collection is None:
        collection = (await ensure_notes_collection(indexer)).physical_name
    return [
        payload
        for payload in await indexer.storage.scroll_points(collection, max_results=0)
        if payload.get("type") in {"note", "chunk"}
        and (note_id is None or payload["note_id"] == str(note_id))
    ]


async def canonical(indexer, note_id=None, collection=None):
    return [
        payload
        for payload in await all_points(indexer, note_id, collection)
        if not is_derived_fragment(payload)
    ]


async def stored_token_sets(indexer):
    return [
        set(indexer.global_vocab.tokenize(stored_embedding_text(payload)))
        for payload in await canonical(indexer)
    ]


async def searchable(indexer, query):
    """Notes a hybrid search returns; every stored note matches the uniform dense vector."""
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    return {result.note.id for result in await engine.search(query)}


async def similar(indexer, note_id):
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    return {result.note.id for result in await engine.find_similar(note_id)}


def summary_of(note_id):
    return Filter(
        must=[
            FieldCondition(key="type", match=MatchValue(value="note")),
            FieldCondition(key="note_id", match=MatchValue(value=str(note_id))),
        ]
    )


def delete_externally(store, *notes):
    """Remove note files as a git checkout would, then let the UUID index forget them.

    Any lookup of a missing path rebuilds the UUID index, after which nothing
    in the store remembers the deleted notes.
    """
    for note in notes:
        store.get_note_path(note.id).unlink()
    store.uuid_index.rebuild()


@pytest.mark.parametrize("force", [False, True])
async def test_index_all_removes_externally_deleted_note(notes, force):
    indexer, vocab = notes
    store = indexer.note_store
    keep = store.create("Keeper", paragraphs("kept", 2))
    gone = store.create("Departing", paragraphs("gone", 3))
    assert (await indexer.index_all()).index_healthy
    assert gone.id in await searchable(indexer, "gonealpha")
    assert gone.id in await similar(indexer, keep.id)

    delete_externally(store, gone)
    # Incrementally nothing else changed, so pass 2 has no work.
    status = await indexer.index_all(force=force)

    assert status.index_healthy
    assert (status.total_notes, status.indexed_notes) == (1, 1)
    assert await all_points(indexer, gone.id) == []
    assert {payload["note_id"] for payload in await all_points(indexer)} == {str(keep.id)}
    assert await searchable(indexer, "gonealpha") == {keep.id}
    assert gone.id not in await similar(indexer, keep.id)
    # The contribution is a fresh registration of the remaining corpus.
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert contribution(vocab) == registration(await stored_token_sets(indexer))
    assert vocab._get_doc_freq().get("gonealpha", 0) == 0


@pytest.mark.parametrize("force", [False, True])
async def test_index_all_removes_chunk_only_group_of_deleted_note(notes, force):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    gone = store.create("Departing", paragraphs("gone", 4))
    await indexer.index_all()
    # An interrupted write or partial delete can leave chunks without a summary,
    # which the note-hash listing never sees.
    client = await indexer.storage.get_client()
    await client.delete(
        (await ensure_notes_collection(indexer)).physical_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value="note")),
                FieldCondition(key="note_id", match=MatchValue(value=str(gone.id))),
            ]
        ),
        wait=True,
    )
    assert {payload["type"] for payload in await all_points(indexer, gone.id)} == {"chunk"}

    delete_externally(store, gone)
    status = await indexer.index_all(force=force)

    assert status.index_healthy
    assert await all_points(indexer, gone.id) == []
    assert contribution(vocab) == registration(source_token_sets(indexer))


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("force", [False, True])
async def test_note_whose_file_no_longer_parses_is_kept(notes, force, named):
    indexer, _ = notes
    store = indexer.note_store
    keep = store.create("Keeper", paragraphs("kept", 2))
    broken = store.create("Broken", paragraphs("broken", 2))
    path = store.get_note_path(broken.id)
    if not named:
        # A hand-written note need not carry its UUID in the file name.
        path = path.rename(path.with_name("handwritten.md"))
        store.uuid_index.rebuild()
    await indexer.index_all()
    before = await all_points(indexer, broken.id)
    assert before

    # An unresolved merge conflict, say: the file is there but does not parse.
    path.write_text("<<<<<<< ours\n" + path.read_text())
    status = await indexer.index_all(force=force)

    assert status.total_notes == 1
    assert await all_points(indexer, broken.id) == before
    assert {payload["note_id"] for payload in await canonical(indexer)} == {
        str(keep.id),
        str(broken.id),
    }


@pytest.mark.parametrize("force", [False, True])
async def test_oversized_note_that_cannot_be_identified_is_kept(notes, monkeypatch, force):
    indexer, _ = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    buried = store.create("Buried", paragraphs("buried", 2))
    path = store.get_note_path(buried.id).rename(store.notes_dir / "handwritten.md")
    store.uuid_index.rebuild()
    await indexer.index_all()
    before = await all_points(indexer, buried.id)
    assert before

    # Over the size limit with its id behind a long title: the bounded
    # deleted-note check cannot see the id, so nothing may be removed.
    monkeypatch.setattr(settings, "max_note_size_kb", 1)
    path.write_text(
        f"---\ntitle: {'T' * 4096}\nid: {buried.id}\n"
        "created: '2026-01-01T00:00:00+00:00'\nmodified: '2026-01-01T00:00:00+00:00'\n"
        "---\n\nbody\n",
        encoding="utf-8",
    )
    status = await indexer.index_all(force=force)

    assert await all_points(indexer, buried.id) == before
    assert status.index_healthy is False


async def test_empty_notes_directory_removes_nothing_unless_forced(notes, caplog):
    indexer, vocab = notes
    store = indexer.note_store
    first = store.create("First", paragraphs("first", 2))
    second = store.create("Second", paragraphs("second", 2))
    await indexer.index_all()
    before = await all_points(indexer)

    # An empty mount point or a misconfigured directory looks like this.
    delete_externally(store, first, second)
    assert store.notes_dir.is_dir()
    status = await indexer.index_all()

    assert status.index_healthy is False
    assert await all_points(indexer) == before
    assert "No notes found" in caplog.text
    # Kept, so still registered as stored.
    assert contribution(vocab) == registration(await stored_token_sets(indexer))
    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) > 0

    # A forced reindex is explicit intent.
    status = await indexer.index_all(force=True)

    assert status.index_healthy
    assert await all_points(indexer) == []
    assert contribution(vocab) == registration([])


@pytest.mark.parametrize("force", [False, True])
async def test_missing_notes_directory_removes_nothing(notes, caplog, force):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("First", paragraphs("first", 2))
    store.create("Second", paragraphs("second", 2))
    await indexer.index_all()
    before = await all_points(indexer)

    store.notes_dir.rename(store.base_dir / "unmounted")
    status = await indexer.index_all(force=force)

    assert status.index_healthy is False
    assert await all_points(indexer) == before
    assert "Could not scan" in caplog.text
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list an unreadable directory")
@pytest.mark.parametrize("force", [False, True])
async def test_unreadable_subdirectory_removes_nothing(notes, caplog, force):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    hidden = store.create("Hidden", paragraphs("hidden", 2), category="private")
    await indexer.index_all()
    before = await all_points(indexer, hidden.id)
    assert before

    # The source scan skips a directory it cannot list without an error.
    private = store.get_note_path(hidden.id).parent
    private.chmod(0)
    try:
        status = await indexer.index_all(force=force)
    finally:
        private.chmod(0o755)

    assert status.index_healthy is False
    assert await all_points(indexer, hidden.id) == before
    assert "Could not scan" in caplog.text
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.parametrize("pass_two_work", [False, True])
async def test_failed_removal_is_unhealthy_and_retried(notes, monkeypatch, caplog, pass_two_work):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    stuck = store.create("Stuck", paragraphs("stuck", 2))
    gone = store.create("Departing", paragraphs("gone", 2))
    await indexer.index_all()
    delete_externally(store, stuck, gone)
    newcomer = store.create("Newcomer", paragraphs("new", 2)) if pass_two_work else None

    real = indexer._delete_note_points

    async def flaky(note_id):
        if note_id == stuck.id:
            raise RuntimeError("injected delete failure")
        await real(note_id)

    monkeypatch.setattr(indexer, "_delete_note_points", flaky)
    status = await indexer.index_all()

    assert status.index_healthy is False
    assert "injected delete failure" in caplog.text
    assert await all_points(indexer, gone.id) == []
    assert await all_points(indexer, stuck.id)
    # Still stored, so still registered.
    assert contribution(vocab) == registration(await stored_token_sets(indexer))
    if newcomer is not None:
        # Indexing the other notes went ahead.
        assert (status.total_notes, status.indexed_notes) == (2, 2)
        assert await canonical(indexer, newcomer.id)

    monkeypatch.setattr(indexer, "_delete_note_points", real)
    status = await indexer.index_all()

    assert status.index_healthy
    assert await all_points(indexer, stuck.id) == []
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.parametrize("force", [False, True])
async def test_shared_collection_removes_only_this_directorys_deleted_notes(
    notes, tmp_path, monkeypatch, caplog, force
):
    monkeypatch.setattr(vector_settings, "collection_name", "shared-knowledge")
    indexer, _ = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    mine = store.create("Mine", paragraphs("mine", 2))
    legacy = store.create("Legacy", paragraphs("legacy", 2))
    await indexer.index_all()
    collection = (await ensure_notes_collection(indexer)).physical_name
    # Another notes directory writes to the same collection.
    other_vocab = GlobalVocabulary(db_path=tmp_path / "other-vocabulary.db")
    other_store = NoteStore(tmp_path / "other-source")
    foreign = other_store.create("Foreign", paragraphs("foreign", 2))
    other = NoteIndexer(other_store, indexer.storage, indexer.embedder, other_vocab)
    try:
        await other.index_note(foreign.id)
    finally:
        other_vocab.close()
    assert other.notes_root != indexer.notes_root
    foreign_before = await all_points(indexer, foreign.id)
    assert {payload[MARKER] for payload in await canonical(indexer, foreign.id)} == {
        other.notes_root
    }
    # A note indexed before points named their directory could be anyone's.
    client = await indexer.storage.get_client()
    await client.delete_payload(
        collection,
        keys=[MARKER],
        points=Filter(must=[FieldCondition(key="note_id", match=MatchValue(value=str(legacy.id)))]),
        wait=True,
    )
    legacy_before = await all_points(indexer, legacy.id)

    # None of the three has a file in this directory.
    delete_externally(store, mine, legacy)
    status = await indexer.index_all(force=force)

    assert status.index_healthy
    assert await all_points(indexer, mine.id) == []
    assert await all_points(indexer, foreign.id) == foreign_before
    assert await all_points(indexer, legacy.id) == legacy_before
    assert caplog.text.count("Keeping 1 deleted note(s) indexed before") == 1


@pytest.mark.parametrize("pass_two_work", [False, True])
async def test_removed_pending_note_leaves_the_index_healthy(notes, pass_two_work):
    indexer, _ = notes
    store = indexer.note_store
    keep = store.create("Keeper", paragraphs("kept", 2))
    gone = store.create("Departing", paragraphs("gone", 2))
    await indexer.index_all()
    # A write that did not finish, or an index policy change, left it pending.
    client = await indexer.storage.get_client()
    await client.set_payload(
        (await ensure_notes_collection(indexer)).physical_name,
        payload={"note_hash": "", "source_reindex_pending": True},
        points=summary_of(gone.id),
        wait=True,
    )
    if pass_two_work:
        store.update(keep.id, content=paragraphs("changed", 2))

    delete_externally(store, gone)
    status = await indexer.index_all()

    assert await all_points(indexer, gone.id) == []
    assert status.index_healthy
    assert (await indexer.get_status()).index_healthy


@pytest.mark.parametrize("then", ["delete", "repair"])
@pytest.mark.parametrize("force", [False, True])
async def test_unparseable_note_stays_registered_as_stored(notes, force, then):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    broken = store.create("Broken", paragraphs("broken", 3))
    await indexer.index_all()
    path = store.get_note_path(broken.id)
    source = path.read_text()
    path.write_text("<<<<<<< ours\n" + source)

    await indexer.index_all(force=force)

    retained = [
        set(vocab.tokenize(stored_embedding_text(payload)))
        for payload in await canonical(indexer, broken.id)
    ]
    assert retained
    # The parsed source notes, plus what is stored for the one that does not parse.
    assert contribution(vocab) == registration(source_token_sets(indexer) + retained)
    assert contribution(vocab) == registration(await stored_token_sets(indexer))

    # Retiring or rewriting it later swaps out exactly what was registered.
    if then == "delete":
        await indexer.delete_note_index(broken.id)
    else:
        path.write_text(source.replace("brokenalpha", "fixedalpha"))
        await indexer.index_note(broken.id)
        assert vocab._get_doc_freq()["fixedalpha"] > 0
    assert vocab._get_doc_freq().get("brokenalpha", 0) == 0
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.parametrize("failure", ["read", "write", "ownership"])
async def test_retained_registration_failure_is_logged_not_raised(
    notes, monkeypatch, caplog, failure
):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    broken = store.create("Broken", paragraphs("broken", 3))
    await indexer.index_all()
    path = store.get_note_path(broken.id)
    path.write_text("<<<<<<< ours\n" + path.read_text())
    injected = AsyncMock(side_effect=RuntimeError("injected read failure"))
    if failure == "read":
        monkeypatch.setattr(indexer, "_stored_token_sets", injected)
    elif failure == "write":

        def locked(*args, **kwargs):
            raise RuntimeError("injected vocabulary failure")

        monkeypatch.setattr(vocab, "update_codebase_incremental", locked)
    else:
        # Without a full scan, ownership is read from the points themselves.
        store.notes_dir.rename(store.base_dir / "unmounted")
        monkeypatch.setattr(indexer, "_owned_note_ids", injected)

    status = await indexer.index_all()

    assert status.index_healthy is (failure != "ownership")
    assert "injected" in caplog.text
    assert await canonical(indexer, broken.id)
    # Nothing worse than registering the parsed source notes alone.
    assert contribution(vocab) == registration(source_token_sets(indexer))


@pytest.mark.parametrize("force", [False, True])
async def test_retained_and_unfinished_notes_are_each_registered_once(notes, monkeypatch, force):
    indexer, vocab = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    broken = store.create("Broken", paragraphs("broken", 3))
    failing = store.create("Failing", paragraphs("old", 3))
    await indexer.index_all()
    path = store.get_note_path(broken.id)
    path.write_text("<<<<<<< ours\n" + path.read_text())
    store.update(failing.id, content=paragraphs("new", 3))

    original = indexer_module.upsert_fragment_group

    async def fail_second_chunk(storage, collection, points):
        payload = points[0].payload
        if payload.get("note_id") == str(failing.id) and payload.get("chunk_index") == 1:
            raise RuntimeError("injected write failure")
        await original(storage, collection, points)

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", fail_second_chunk)
    retained = AsyncMock(wraps=indexer._register_retained_notes)
    unfinished = AsyncMock(wraps=indexer._reconcile_unfinished_notes)
    monkeypatch.setattr(indexer, "_register_retained_notes", retained)
    monkeypatch.setattr(indexer, "_reconcile_unfinished_notes", unfinished)

    assert (await indexer.index_all(force=force)).index_healthy is False

    # A retained note has no parsed source, so pass 2 never writes it.
    assert set().union(*(call.args[0] for call in retained.await_args_list)) == {str(broken.id)}
    assert {
        str(note_id) for call in unfinished.await_args_list for note_id, _, _ in call.args[0]
    } == {str(failing.id)}
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.parametrize("force", [False, True])
async def test_shared_collection_registers_only_this_directorys_retained_notes(
    notes, tmp_path, monkeypatch, force
):
    monkeypatch.setattr(vector_settings, "collection_name", "shared-knowledge")
    indexer, vocab = notes
    store = indexer.note_store
    keep = store.create("Keeper", paragraphs("kept", 2))
    broken = store.create("Broken", paragraphs("broken", 3))
    legacy = store.create("Legacy", paragraphs("legacy", 2))
    await indexer.index_all()
    collection = (await ensure_notes_collection(indexer)).physical_name
    other_vocab = GlobalVocabulary(db_path=tmp_path / "other-vocabulary.db")
    other_store = NoteStore(tmp_path / "other-source")
    foreign = other_store.create("Foreign", paragraphs("foreign", 2))
    other = NoteIndexer(other_store, indexer.storage, indexer.embedder, other_vocab)
    try:
        await other.index_note(foreign.id)
    finally:
        other_vocab.close()
    client = await indexer.storage.get_client()
    await client.delete_payload(
        collection,
        keys=[MARKER],
        points=Filter(must=[FieldCondition(key="note_id", match=MatchValue(value=str(legacy.id)))]),
        wait=True,
    )
    path = store.get_note_path(broken.id)
    path.write_text("<<<<<<< ours\n" + path.read_text())
    delete_externally(store, legacy)

    await indexer.index_all(force=force)

    # The unparseable note is this directory's; the other root's note and the
    # unattributable legacy one are kept but not counted here.
    assert await canonical(indexer, foreign.id)
    assert await canonical(indexer, legacy.id)
    own = [
        set(vocab.tokenize(stored_embedding_text(payload)))
        for payload in await canonical(indexer)
        if payload["note_id"] in {str(keep.id), str(broken.id)}
    ]
    assert contribution(vocab) == registration(own)
    assert vocab._get_doc_freq()["brokenalpha"] > 0
    assert vocab._get_doc_freq().get("legacyalpha", 0) == 0


async def test_written_points_name_their_notes_directory(notes, tmp_path):
    indexer, vocab = notes
    store = indexer.note_store
    first = store.create("First", paragraphs("first", 3))
    second = store.create("Second", paragraphs("second", 3))
    await indexer.index_note(first.id)
    await indexer.index_all()

    points = await canonical(indexer)
    assert {payload["type"] for payload in points} == {"note", "chunk"}
    assert {payload["note_id"] for payload in points} == {str(first.id), str(second.id)}
    assert {payload[MARKER] for payload in points} == {indexer.notes_root}
    # Stable for the directory, distinct from another, and not the path itself.
    same = NoteIndexer(NoteStore(store.base_dir), indexer.storage, indexer.embedder, vocab)
    elsewhere = NoteIndexer(
        NoteStore(tmp_path / "elsewhere"), indexer.storage, indexer.embedder, vocab
    )
    assert same.notes_root == indexer.notes_root
    assert elsewhere.notes_root != indexer.notes_root
    assert str(store.base_dir) not in indexer.notes_root


async def test_migration_keeps_and_rebuilds_with_the_marker(notes):
    indexer, vocab = notes
    store = indexer.note_store
    unchanged = store.create("Unchanged", paragraphs("steady", 2))
    changed = store.create("Changed", paragraphs("first", 2))
    await indexer.index_all()
    original = (await ensure_notes_collection(indexer)).physical_name
    # The changed source makes the migration rebuild this note in the candidate.
    store.update(changed.id, content=paragraphs("rebuilt", 2))

    embedder = isolated_embedder(256, namespace="migrated")
    try:
        replacement = NoteIndexer(store, indexer.storage, embedder, vocab)
        generation = await ensure_notes_collection(replacement)
        assert generation.migrated
        assert generation.physical_name != original
        points = await canonical(replacement, collection=generation.physical_name)
    finally:
        await embedder.close()

    assert {payload["note_id"] for payload in points} == {str(unchanged.id), str(changed.id)}
    assert any(
        "rebuiltalpha" in stored_embedding_text(payload)
        for payload in points
        if payload["note_id"] == str(changed.id)
    )
    assert {payload[MARKER] for payload in points} == {indexer.notes_root}


async def test_note_indexed_before_the_marker_is_marked_without_reindexing(notes, monkeypatch):
    monkeypatch.setattr(vector_settings, "collection_name", "shared-knowledge")
    indexer, _ = notes
    store = indexer.note_store
    store.create("Keeper", paragraphs("kept", 2))
    note = store.create("Subject", paragraphs("first", 3))
    await indexer.index_all()
    collection = (await ensure_notes_collection(indexer)).physical_name
    client = await indexer.storage.get_client()
    # As written by a release without the marker; hash and policy are current.
    await client.delete_payload(
        collection,
        keys=[MARKER],
        points=Filter(must=[FieldCondition(key="note_id", match=MatchValue(value=str(note.id)))]),
        wait=True,
    )
    before = await all_points(indexer, note.id)
    assert all(MARKER not in payload for payload in before)

    rewrite = AsyncMock(wraps=indexer._index_note)
    monkeypatch.setattr(indexer, "_index_note", rewrite)
    status = await indexer.index_all()

    assert status.index_healthy
    rewrite.assert_not_awaited()
    after = await all_points(indexer, note.id)
    assert {payload[MARKER] for payload in after if not is_derived_fragment(payload)} == {
        indexer.notes_root
    }
    assert [{k: v for k, v in payload.items() if k != MARKER} for payload in after] == before

    # Marked, it is removed once deleted, even from the shared collection.
    delete_externally(store, note)
    assert (await indexer.index_all()).index_healthy
    assert await all_points(indexer, note.id) == []


def test_note_ids_on_disk_counts_every_note_file(tmp_path):
    store = NoteStore(tmp_path / "source")
    with pytest.raises(FileNotFoundError):
        store.note_ids_on_disk()

    named = store.create("Named", "Body", category="work")
    handwritten = store.create("Handwritten", "Body")
    path = store.get_note_path(handwritten.id)
    path.rename(path.with_name("handwritten.md"))
    broken, mentioned = uuid4(), uuid4()
    (store.notes_dir / f"broken-{broken}.md").write_text("no frontmatter")
    (store.notes_dir / "conflict.md").write_text(f"<<<<<<< ours\nid: {mentioned}\n")
    (store.notes_dir / f"{uuid4()}.txt").write_text("not markdown")

    assert store.note_ids_on_disk() == {named.id, handwritten.id, broken, mentioned}
