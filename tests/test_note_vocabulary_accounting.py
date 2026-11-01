"""Note writes keep the shared sparse vocabulary equal to the indexed notes.

index_all registers one vocabulary document per canonical point: the note
summary plus each chunk. Edits, deletes and failed writes, including notes an
index_all pass does not finish, must leave the contribution exactly where a
fresh registration of the stored corpus would put it.
"""

import asyncio
import sqlite3
from collections import Counter
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_core import EmbeddingClient, QdrantStorage, generate_point_id
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing import indexer as indexer_module
from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID, NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.parser import parse_note

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def isolated_embedder(capacity):
    client = EmbeddingClient(
        model="isolated-model",
        dim=128,
        profile="raw",
        max_text_chars=capacity,
        cache_namespace="vocabulary",
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
async def notes_vocab(tmp_path, monkeypatch):
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


async def stored_payloads(indexer):
    generation = await ensure_notes_collection(indexer)
    return [
        payload
        for payload in await indexer.storage.scroll_points(generation.physical_name, max_results=0)
        if payload.get("type") in {"note", "chunk"}
    ]


async def stored_token_sets(indexer):
    """One set per canonical point actually stored, read independently of the indexer."""
    return [
        set(indexer.global_vocab.tokenize(stored_embedding_text(payload)))
        for payload in await stored_payloads(indexer)
        if not is_derived_fragment(payload)
    ]


def point_count(indexer, note_id):
    parsed = parse_note(indexer.note_store.read(note_id).content)
    return 1 + len(chunk_note(parsed, indexer.embedder))


def fail_first_vocabulary_write(monkeypatch, vocab):
    """Fail only the first delta; later calls reach the real database."""
    real = vocab.update_codebase_incremental
    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(vocab, "update_codebase_incremental", flaky)
    return calls


async def test_repeated_edits_keep_one_document_per_point(notes_vocab):
    indexer, vocab = notes_vocab
    store = indexer.note_store
    neighbour = store.create("Neighbour", paragraphs("steady", 2))
    note = store.create("Subject", paragraphs("first", 3))
    await indexer.index_note(neighbour.id)
    await indexer.index_note(note.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))

    previous = "firstalpha"
    for version, count in [("second", 5), ("third", 1), ("fourth", 4), ("fifth", 2)]:
        store.update(note.id, content=paragraphs(version, count))
        await indexer.index_note(note.id)
        expected_documents = point_count(indexer, neighbour.id) + point_count(indexer, note.id)
        assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == expected_documents
        assert contribution(vocab) == registration(source_token_sets(indexer))
        # The vocabulary index is append-only: a retired term keeps its row at zero.
        assert vocab._get_doc_freq().get(previous, 0) == 0
        assert vocab._get_doc_freq()["steadyalpha"] > 0
        previous = f"{version}alpha"

    # A full registration of the same corpus finds nothing to correct.
    before = contribution(vocab)
    await indexer.index_all()
    assert contribution(vocab) == before


async def test_new_term_reaches_sparse_vectors_of_edited_note(notes_vocab):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("plain", 2))
    await indexer.index_note(note.id)
    assert "zanzibarquokka" not in vocab._get_vocab()

    indexer.note_store.update(note.id, content="zanzibarquokka " + paragraphs("plain", 2))
    await indexer.index_note(note.id)

    generation = await ensure_notes_collection(indexer)
    client = await indexer.storage.get_client()
    records, _ = await client.scroll(
        generation.physical_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="note_id", match=MatchValue(value=str(note.id)))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=True,
    )
    holders = [
        record
        for record in records
        if not is_derived_fragment(record.payload)
        and "zanzibarquokka" in stored_embedding_text(record.payload)
    ]
    assert {record.payload["type"] for record in holders} == {"note", "chunk"}
    index = vocab._get_vocab()["zanzibarquokka"]
    assert all(index in record.vector["sparse"].indices for record in holders)


async def test_shrinking_note_retires_dropped_chunk_documents(notes_vocab):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("long", 8) + "\n\ntailonlyterm")
    await indexer.index_note(note.id)
    many = point_count(indexer, note.id)
    assert many > 5
    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == many
    assert vocab._get_doc_freq()["tailonlyterm"] == 1

    indexer.note_store.update(note.id, content="Short replacement")
    await indexer.index_note(note.id)
    assert point_count(indexer, note.id) == 2
    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == 2
    assert vocab._get_doc_freq().get("tailonlyterm", 0) == 0
    assert vocab._get_doc_freq().get("longalpha", 0) == 0
    assert contribution(vocab) == registration(source_token_sets(indexer))


async def test_fragment_children_are_not_vocabulary_documents(notes_vocab):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("first", 4))
    await indexer.index_note(note.id)
    # The summary is longer than the embedder capacity, so it carries children.
    children = [p for p in await stored_payloads(indexer) if is_derived_fragment(p)]
    assert any(p["type"] == "note" for p in children)

    for version in ("second", "third"):
        indexer.note_store.update(note.id, content=paragraphs(version, 4))
        await indexer.index_note(note.id)
        assert any(is_derived_fragment(p) for p in await stored_payloads(indexer))
        assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == point_count(indexer, note.id)
        assert contribution(vocab) == registration(source_token_sets(indexer))


async def test_textless_legacy_point_still_counts_as_a_document(notes_vocab):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("first", 3))
    await indexer.index_note(note.id)
    generation = await ensure_notes_collection(indexer)
    client = await indexer.storage.get_client()
    # A summary from before retained embedding text: nothing the read-back
    # projects but note_id survives.
    await client.delete_payload(
        generation.physical_name,
        keys=["embedding_text", "embedding_fragment"],
        points=[generate_point_id(f"note:{note.id}")],
        wait=True,
    )

    indexer.note_store.update(note.id, content=paragraphs("second", 2))
    await indexer.index_note(note.id)
    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == point_count(indexer, note.id)


async def test_delete_retires_the_notes_documents(notes_vocab):
    indexer, vocab = notes_vocab
    keep = indexer.note_store.create("Keeper", paragraphs("kept", 2))
    gone = indexer.note_store.create("Departing", paragraphs("gone", 3))
    await indexer.index_note(keep.id)
    await indexer.index_note(gone.id)
    kept = contribution(vocab)

    indexer.note_store.delete(gone.id)
    await indexer.delete_note_index(gone.id)

    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == point_count(indexer, keep.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert vocab._get_doc_freq().get("gonealpha", 0) == 0
    assert vocab._get_doc_freq()["keptalpha"] == kept[1]["keptalpha"]


async def test_failed_delete_retires_nothing(notes_vocab, monkeypatch):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("first", 3))
    await indexer.index_note(note.id)
    before = contribution(vocab)
    stored = await stored_payloads(indexer)

    monkeypatch.setattr(
        indexer.storage,
        "delete_by_filter",
        AsyncMock(side_effect=RuntimeError("injected delete failure")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        await indexer.delete_note_index(note.id)

    assert contribution(vocab) == before
    assert await stored_payloads(indexer) == stored


async def test_deleting_an_unindexed_note_changes_nothing(notes_vocab, monkeypatch):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("first", 2))
    await indexer.index_note(note.id)
    before = contribution(vocab)
    calls = []
    monkeypatch.setattr(vocab, "update_codebase_incremental", lambda *a, **k: calls.append(k))

    await indexer.delete_note_index(uuid4())

    assert calls == []
    assert contribution(vocab) == before


@pytest.mark.parametrize("stage", ["embed", "summary"])
async def test_write_failure_before_any_commit_reverts_the_delta_exactly(
    notes_vocab, monkeypatch, stage
):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("old", 5))
    await indexer.index_note(note.id)
    before = contribution(vocab)
    indexer.note_store.update(note.id, content="revertedterm " + paragraphs("new", 2))

    original_write = indexer_module.upsert_fragment_group

    async def fail_summary(storage, collection, points):
        if points[0].payload.get("type") == "note":
            raise RuntimeError("injected summary write failure")
        await original_write(storage, collection, points)

    if stage == "embed":
        monkeypatch.setattr(
            indexer.embedder,
            "embed_all",
            AsyncMock(side_effect=RuntimeError("injected embed failure")),
        )
    else:
        monkeypatch.setattr(indexer_module, "upsert_fragment_group", fail_summary)
    with pytest.raises(RuntimeError, match="injected"):
        await indexer.index_note(note.id)

    # Nothing was replaced, so the previous contribution is restored exactly.
    assert contribution(vocab) == before
    assert vocab._get_doc_freq()["revertedterm"] == 0
    assert contribution(vocab) == registration(await stored_token_sets(indexer))


@pytest.mark.parametrize("stage", ["chunk", "cleanup", "complete"])
async def test_partial_write_leaves_contribution_matching_stored_points(
    notes_vocab, monkeypatch, stage
):
    indexer, vocab = notes_vocab
    neighbour = indexer.note_store.create("Neighbour", paragraphs("steady", 2))
    note = indexer.note_store.create("Subject", paragraphs("old", 6))
    await indexer.index_note(neighbour.id)
    await indexer.index_note(note.id)
    old_points = point_count(indexer, note.id)
    indexer.note_store.update(note.id, content=paragraphs("new", 3))
    assert old_points > point_count(indexer, note.id) + 1

    original_write = indexer_module.upsert_fragment_group
    original_cleanup = indexer._delete_orphan_chunks
    original_complete = indexer._complete_note_index

    async def fail_second_chunk(storage, collection, points):
        payload = points[0].payload
        if payload.get("type") == "chunk" and payload.get("chunk_index") == 1:
            raise RuntimeError("injected chunk write failure")
        await original_write(storage, collection, points)

    async def fail(*args):
        raise RuntimeError(f"injected {stage} failure")

    if stage == "chunk":
        monkeypatch.setattr(indexer_module, "upsert_fragment_group", fail_second_chunk)
    elif stage == "cleanup":
        monkeypatch.setattr(indexer, "_delete_orphan_chunks", fail)
    else:
        monkeypatch.setattr(indexer, "_complete_note_index", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await indexer.index_note(note.id)

    # The stored note is part old, part new, or all new; the contribution
    # describes exactly what is stored, which is what the retry retires.
    assert contribution(vocab) == registration(await stored_token_sets(indexer))

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", original_write)
    monkeypatch.setattr(indexer, "_delete_orphan_chunks", original_cleanup)
    monkeypatch.setattr(indexer, "_complete_note_index", original_complete)
    await indexer.index_note(note.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert vocab._get_doc_freq().get("oldalpha", 0) == 0


async def test_vocabulary_write_failure_does_not_lose_the_note(notes_vocab, monkeypatch, caplog):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("old", 2))
    await indexer.index_note(note.id)
    before = contribution(vocab)
    calls = fail_first_vocabulary_write(monkeypatch, vocab)

    indexer.note_store.update(note.id, content="lockedoutterm " + paragraphs("new", 2))
    await indexer.index_note(note.id)

    assert len(calls) == 1
    assert "Failed to update the notes vocabulary contribution" in caplog.text
    assert contribution(vocab) == before
    summary = next(
        p
        for p in await stored_payloads(indexer)
        if p["type"] == "note" and not is_derived_fragment(p)
    )
    assert "lockedoutterm" in summary["embedding_text"]
    assert summary["source_reindex_pending"] is False


async def test_failed_vocabulary_write_is_never_compensated(notes_vocab, monkeypatch):
    indexer, vocab = notes_vocab
    note = indexer.note_store.create("Subject", paragraphs("old", 5))
    await indexer.index_note(note.id)
    before = contribution(vocab)
    calls = fail_first_vocabulary_write(monkeypatch, vocab)
    monkeypatch.setattr(
        indexer.embedder,
        "embed_all",
        AsyncMock(side_effect=RuntimeError("injected embed failure")),
    )

    indexer.note_store.update(note.id, content=paragraphs("new", 2))
    with pytest.raises(RuntimeError, match="injected embed failure"):
        await indexer.index_note(note.id)

    # Inverting the delta that never landed would reach the real database.
    assert len(calls) == 1
    assert contribution(vocab) == before


# index_all registers every source note before pass 2 writes it. A note that
# pass 2 does not finish must end up registered as what is actually stored for
# it, so that a later index_note retry swaps exactly that.


def fail_note_writes(monkeypatch, note_id, stage, error=None):
    """Fail one note's summary group ("nothing" new stored) or second chunk ("partial")."""
    original = indexer_module.upsert_fragment_group

    async def failing(storage, collection, points):
        payload = points[0].payload
        target = (
            payload.get("type") == "note"
            if stage == "nothing"
            else (payload.get("chunk_index") == 1)
        )
        if payload.get("note_id") == str(note_id) and target:
            raise error or RuntimeError("injected write failure")
        await original(storage, collection, points)

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", failing)
    return original


@pytest.mark.parametrize("stage", ["nothing", "partial"])
@pytest.mark.parametrize("force", [False, True])
async def test_index_all_registers_failed_new_note_as_stored(
    notes_vocab, monkeypatch, force, stage
):
    indexer, vocab = notes_vocab
    store = indexer.note_store
    good = store.create("Good", paragraphs("good", 2))
    bad = store.create("Bad", paragraphs("bad", 3))
    if force:
        # A force pass clears both groups before rewriting them.
        await indexer.index_note(good.id)
        await indexer.index_note(bad.id)
    original = fail_note_writes(monkeypatch, bad.id, stage)

    status = await indexer.index_all(force=force)

    assert status.index_healthy is False
    stored = await stored_token_sets(indexer)
    assert contribution(vocab) == registration(stored)
    if stage == "nothing":
        assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == point_count(indexer, good.id)

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", original)
    await indexer.index_note(bad.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))


@pytest.mark.parametrize("stage", ["nothing", "partial"])
async def test_index_all_registers_failed_existing_note_as_stored(notes_vocab, monkeypatch, stage):
    indexer, vocab = notes_vocab
    store = indexer.note_store
    good = store.create("Good", paragraphs("good", 2))
    bad = store.create("Bad", paragraphs("old", 6))
    await indexer.index_note(good.id)
    await indexer.index_note(bad.id)
    store.update(bad.id, content=paragraphs("new", 3))
    original = fail_note_writes(monkeypatch, bad.id, stage)

    assert (await indexer.index_all()).index_healthy is False

    # The old points, or a mix of the new summary and leading chunk with the
    # old trailing chunks, are what the retry will read back and retire.
    assert contribution(vocab) == registration(await stored_token_sets(indexer))
    # The last old paragraph survives either way, in an old trailing chunk.
    assert vocab._get_doc_freq()["oldfoxtrot"] > 0

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", original)
    await indexer.index_note(bad.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))
    assert vocab._get_doc_freq().get("oldfoxtrot", 0) == 0


@pytest.mark.parametrize("force", [False, True])
async def test_index_all_unreadable_note_without_points_is_assumed_to_store_nothing(
    notes_vocab, monkeypatch, caplog, force
):
    indexer, vocab = notes_vocab
    good = indexer.note_store.create("Good", paragraphs("good", 2))
    bad = indexer.note_store.create("Bad", paragraphs("bad", 3))
    if force:
        # New, or cleared by the force pass: either way it had no points.
        await indexer.index_note(good.id)
        await indexer.index_note(bad.id)
    fail_note_writes(monkeypatch, bad.id, "nothing")
    monkeypatch.setattr(
        indexer, "_indexed_token_sets", AsyncMock(side_effect=RuntimeError("injected read"))
    )

    await indexer.index_all(force=force)

    assert "assuming it stored nothing" in caplog.text
    assert vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) == point_count(indexer, good.id)
    assert vocab._get_doc_freq().get("badalpha", 0) == 0


async def test_index_all_unreadable_existing_note_keeps_its_registration(
    notes_vocab, monkeypatch, caplog
):
    indexer, vocab = notes_vocab
    good = indexer.note_store.create("Good", paragraphs("good", 2))
    bad = indexer.note_store.create("Bad", paragraphs("old", 3))
    await indexer.index_note(good.id)
    await indexer.index_note(bad.id)
    indexer.note_store.update(bad.id, content=paragraphs("new", 2))
    fail_note_writes(monkeypatch, bad.id, "nothing")
    monkeypatch.setattr(
        indexer, "_indexed_token_sets", AsyncMock(side_effect=RuntimeError("injected read"))
    )

    await indexer.index_all()

    # Its old sets are unknown, so nothing is retired in their place.
    assert "stays as registered" in caplog.text
    assert contribution(vocab) == registration(source_token_sets(indexer))


async def test_index_all_reconciliation_failure_is_logged_not_raised(
    notes_vocab, monkeypatch, caplog
):
    indexer, vocab = notes_vocab
    indexer.note_store.create("Good", paragraphs("good", 2))
    bad = indexer.note_store.create("Bad", paragraphs("bad", 3))
    fail_note_writes(monkeypatch, bad.id, "nothing")
    calls = []

    def locked(*args, **kwargs):
        calls.append(kwargs)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(vocab, "update_codebase_incremental", locked)

    status = await indexer.index_all()

    assert status.index_healthy is False
    assert len(calls) == 1
    assert "Failed to update the notes vocabulary contribution" in caplog.text


@pytest.mark.parametrize("force", [False, True])
async def test_cancelled_index_all_registers_unfinished_notes_as_stored(
    notes_vocab, monkeypatch, force
):
    indexer, vocab = notes_vocab
    notes = [
        indexer.note_store.create(title, paragraphs(f"old{title.lower()}", 4))
        for title in ("First", "Second", "Third")
    ]
    for note in notes:
        await indexer.index_note(note.id)
    for note in notes:
        indexer.note_store.update(note.id, content=paragraphs("new", 3))

    original = indexer_module.upsert_fragment_group
    order: list[str] = []

    async def cancelling(storage, collection, points):
        payload = points[0].payload
        if payload.get("type") == "note" and payload["note_id"] not in order:
            order.append(payload["note_id"])
        # Cancel partway through the second note: the first is finished, the
        # second part written and the third never reached.
        if order[1:] == [payload["note_id"]] and payload.get("chunk_index") == 1:
            raise asyncio.CancelledError
        await original(storage, collection, points)

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", cancelling)
    with pytest.raises(asyncio.CancelledError):
        await indexer.index_all(force=force)
    assert len(order) == 2

    assert contribution(vocab) == registration(await stored_token_sets(indexer))

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", original)
    for note in notes:
        await indexer.index_note(note.id)
    assert contribution(vocab) == registration(source_token_sets(indexer))
