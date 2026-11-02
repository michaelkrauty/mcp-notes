"""A note deleted while it is being updated stays deleted (issue #66).

#66 described an update reading a note between a delete's index removal and
its file removal, then writing the deleted note back to the index. Every note
tool holds vector-core's collection lock, a flock in the shared lock
directory, from before its service call until it returns: ``collection_mutation``
opens the scope and ``prepare_collection_mutation`` enters
``indexer.collection_operation()``. A delete's two removals therefore happen
under one hold of that lock, so no update in this or another process can read
the note in between. These tests pin that scope.

Processes are modelled as services over one notes directory, Qdrant database,
vocabulary and lock directory. Interleavings are driven with events: an
embedding, or a delete just after its index removal, parks until released.
"""

import asyncio
import inspect
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from qdrant_client import AsyncQdrantClient
from vector_core import EmbeddingClient, QdrantStorage
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID, NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.server import (
    create_note,
    delete_note,
    get_git,
    get_indexer,
    get_note_history,
    get_store,
    merge_tags,
    move_category,
    rename_tag,
    restore_note_version,
    update_note,
)
from mcp_notes.services.note_service import NoteService
from mcp_notes.singletons import get_note_service
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.git import GitManager

DIM = 128


@pytest.fixture
async def spawn(tmp_path, monkeypatch):
    """Start a "process": its own store, indexer, git handle and embedder.

    All share the notes directory, Qdrant database, vocabulary database, lock
    directory and embedding identity.
    """
    monkeypatch.setattr(vector_settings, "collection_name", None)
    monkeypatch.setattr(vector_settings, "cache_dir", tmp_path / "cache")
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=DIM)
    storage._client = AsyncQdrantClient(location=":memory:")
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    embedders = []

    def start():
        embedder = EmbeddingClient(model="m", dim=DIM, profile="raw", cache_namespace="race")
        embedder.resolve_identity = AsyncMock(return_value=embedder.configured_identity())
        embedder._embed_prepared_batch = AsyncMock(
            side_effect=lambda texts: [[1.0] + [0.0] * (DIM - 1) for _ in texts]
        )
        embedders.append(embedder)
        store = NoteStore(tmp_path / "notes")
        indexer = NoteIndexer(store, storage, embedder, vocab)
        integrity = MagicMock(**{"mark_note_modified.return_value": 0})
        integrity.mark_note_deleted.return_value = 0
        service = NoteService(store, GitManager(notes_dir=store.base_dir), indexer, integrity)
        return SimpleNamespace(store=store, indexer=indexer, service=service, vocab=vocab)

    yield start
    await storage.close()
    for embedder in embedders:
        await embedder.close()
    vocab.close()


@asynccontextmanager
async def tool_scope(process, scoped=True):
    """The lock scope a note tool holds around its service call (tools/mutation.py)."""
    if not scoped:
        yield
        return
    async with process.indexer.collection_operation():
        yield


async def mutate(process, call, scoped=True):
    async with tool_scope(process, scoped):
        return await call


class Gate:
    def __init__(self):
        self.parked = asyncio.Event()
        self.release = asyncio.Event()

    async def park(self):
        self.parked.set()
        await self.release.wait()


def park_next_embedding(monkeypatch, process):
    """Park the next embedding: its note has been read and not yet written."""
    gate, embedder = Gate(), process.indexer.embedder
    embed_all, armed = embedder.embed_all, [True]

    async def embed(texts, **kwargs):
        if armed:
            armed.clear()
            await gate.park()
        return await embed_all(texts, **kwargs)

    monkeypatch.setattr(embedder, "embed_all", embed)
    return gate


def park_after(monkeypatch, target, name):
    """Park every caller of an async method once it returns."""
    gate, method = Gate(), getattr(target, name)

    async def parked(*args, **kwargs):
        result = await method(*args, **kwargs)
        await gate.park()
        return result

    monkeypatch.setattr(target, name, parked)
    return gate


def text(prefix):
    return "\n\n".join(f"{prefix}{word} filler " * 14 for word in ("alpha", "bravo", "charlie"))


async def points(process, note_id):
    """Every point of a note, derived fragments included."""
    collection = (await ensure_notes_collection(process.indexer)).physical_name
    return [
        payload
        for payload in await process.indexer.storage.scroll_points(collection, max_results=0)
        if payload.get("note_id") == str(note_id)
    ]


async def searchable(process, query):
    indexer = process.indexer
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    return {result.note.id for result in await engine.search(query)}


def registration(token_sets):
    """The notes document count and document frequencies these documents register."""
    return len(token_sets), dict(Counter(token for tokens in token_sets for token in tokens))


async def assert_only_keeper_indexed(process, keeper, doomed):
    """The deleted note left no points, no search hit and no vocabulary document."""
    vocab = process.vocab
    assert await points(process, doomed.id) == []
    assert await searchable(process, "editedalpha") <= {keeper.id}
    assert keeper.id in await searchable(process, "keptalpha")
    on_disk = []
    for parsed, _ in process.store.iter_all():
        on_disk.append(set(vocab.tokenize(generate_note_summary(parsed))))
        on_disk.extend(
            set(vocab.tokenize(c.content)) for c in chunk_note(parsed, process.indexer.embedder)
        )
    collection = (await ensure_notes_collection(process.indexer)).physical_name
    stored = [
        set(vocab.tokenize(stored_embedding_text(payload)))
        for payload in await process.indexer.storage.scroll_points(collection, max_results=0)
        if payload.get("type") in {"note", "chunk"} and not is_derived_fragment(payload)
    ]
    doc_freq = {token: freq for token, freq in vocab._get_doc_freq().items() if freq}
    contribution = (vocab.get_codebase_doc_count(NOTES_CODEBASE_ID), doc_freq)
    assert contribution == registration(on_disk) == registration(stored)


async def notes(process):
    keeper = (await process.service.create("Keeper", text("kept"))).note
    doomed = (await process.service.create("Doomed", text("doomed"))).note
    return keeper, doomed


@pytest.mark.parametrize("across", [False, True], ids=["one-process", "two-processes"])
async def test_delete_waits_for_an_update_parked_in_its_embedding(spawn, monkeypatch, across):
    """The issue's interleaving cannot start: the update holds the lock while it embeds.

    The update holds the collection lock from reading the note to writing it,
    in its own index work and, across processes, in the tool scope, so the
    delete waits for it and then removes what it wrote.
    """
    updater = spawn()
    deleter = spawn() if across else updater
    keeper, doomed = await notes(updater)
    embedding = park_next_embedding(monkeypatch, updater)

    edit = updater.service.update(doomed.id, content=text("edited"))
    update = asyncio.create_task(mutate(updater, edit, scoped=across))
    await embedding.parked.wait()
    delete = asyncio.create_task(mutate(deleter, deleter.service.delete(doomed.id), across))
    await asyncio.sleep(0)
    assert not delete.done()
    assert updater.store.get_note_path(doomed.id).exists()
    embedding.release.set()
    updated, deleted = await asyncio.gather(update, delete)

    assert updated.success and deleted.success
    assert not deleter.store.exists(doomed.id)
    await assert_only_keeper_indexed(updater, keeper, doomed)


@pytest.mark.parametrize("scoped", [True, False], ids=["tool-scope", "no-scope"])
async def test_tool_scope_keeps_an_update_out_of_a_delete_in_progress(spawn, monkeypatch, scoped):
    """Another process's update arrives between the delete's index and file removals.

    Inside the tool scope the delete still holds the lock there, so the update
    waits and then finds the note gone. Without it the update reads the note,
    indexes it and outlives the delete's index removal: #66's resurrection.
    """
    deleter, updater = spawn(), spawn()
    keeper, doomed = await notes(deleter)
    removed = park_after(monkeypatch, deleter.indexer, "delete_note_index")

    delete = asyncio.create_task(mutate(deleter, deleter.service.delete(doomed.id), scoped))
    await removed.parked.wait()
    edit = updater.service.update(doomed.id, content=text("edited"))
    update = asyncio.create_task(mutate(updater, edit, scoped))
    if scoped:
        await asyncio.sleep(0)
        assert not update.done()
    else:
        assert (await update).success
    removed.release.set()
    deleted, updated = await asyncio.gather(delete, update)

    assert deleted.success
    assert doomed.id not in deleter.store.note_ids_on_disk()
    if scoped:
        assert updated.error_code == "note_not_found"
        await assert_only_keeper_indexed(deleter, keeper, doomed)
    else:
        assert await points(deleter, doomed.id)
        assert doomed.id in await searchable(deleter, "editedalpha")


@pytest.fixture
async def trace(tmp_notes_dir, offline_collection, monkeypatch):
    """Record, in order, the tools' outermost collection lock scope and the watched calls."""
    events: list[str] = []
    indexer = await get_indexer()
    operation, depth = indexer.collection_operation, 0

    @asynccontextmanager
    async def traced_operation():
        nonlocal depth
        depth += 1
        try:
            async with operation() as generation:
                if depth == 1:
                    events.append("lock")
                yield generation
        finally:
            depth -= 1
            if not depth:
                events.append("unlock")

    monkeypatch.setattr(indexer, "collection_operation", traced_operation)

    def watch(target, name):
        method = getattr(target, name)

        async def watched_async(*args, **kwargs):
            events.append(name)
            result = await method(*args, **kwargs)
            events.append(f"{name} returned")
            return result

        def watched(*args, **kwargs):
            events.append(name)
            return method(*args, **kwargs)

        monkeypatch.setattr(
            target, name, watched_async if inspect.iscoroutinefunction(method) else watched
        )

    return SimpleNamespace(events=events, watch=watch)


async def test_note_tools_hold_the_lock_across_the_service_call(trace):
    service = await get_note_service()
    for name in ("create", "update", "delete"):
        trace.watch(service, name)

    note_id = (await create_note(title="N", content="Body"))["id"]
    assert trace.events == ["lock", "create", "create returned", "unlock"]
    trace.events.clear()
    assert (await update_note(note_id, content="Edited"))["id"] == note_id
    assert trace.events == ["lock", "update", "update returned", "unlock"]
    trace.events.clear()
    assert (await delete_note(note_id))["success"]
    assert trace.events == ["lock", "delete", "delete returned", "unlock"]


async def test_restore_holds_the_lock_across_rewrite_and_reindex(trace):
    note_id = (await create_note(title="N", content="First"))["id"]
    await update_note(note_id, content="Second")
    first = (await get_note_history(note_id))[-1]["commit_sha"]
    trace.watch(get_git(), "restore_version")
    trace.watch(await get_indexer(), "index_note")
    trace.events.clear()

    assert (await restore_note_version(note_id, first))["content"].endswith("First")
    assert trace.events == [
        "lock",
        "restore_version",
        "index_note",
        "index_note returned",
        "unlock",
    ]


@pytest.mark.parametrize(
    ("tool", "args"),
    [(move_category, ("old", "new")), (rename_tag, ("old", "new")), (merge_tags, (["old"], "new"))],
)
async def test_bulk_rewrites_hold_the_lock_across_rewrite_and_reindex(trace, tool, args):
    note_id = UUID(
        (await create_note(title="N", content="Body", tags=["old"], category="old"))["id"]
    )
    trace.watch(get_store(), "update")
    trace.watch(await get_indexer(), "index_all")
    trace.events.clear()

    assert (await tool(*args))["updated_count"] == 1
    assert get_store().get_note_path(note_id).exists()
    assert trace.events == ["lock", "update", "index_all", "index_all returned", "unlock"]
