"""Searches by a bound engine do not wait for a note mutation in progress.

Every note tool holds vector-core's collection lock for its whole operation,
embedding included. vector-core 1.12.0's current_embedding_binding() confirms
without that lock that a client's earlier binding is still the active,
verified generation, and NoteIndexer.read_operation() tries it before falling
back to collection_operation(). The search engine and get_status() resolve
their targets that way.

The offline stack here resolves the embedding identity for real, as a server
process does; the lock check needs it. A mutation is parked inside
collection_operation() on an event, so it holds the lock while searches run.
The guard only stops a regression from hanging the suite: the lock itself
waits up to an hour.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from vector_core import EmbeddingClient
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_migration import (
    EmbeddingMigrationError,
    current_embedding_binding,
)

import mcp_notes.indexing.indexer as indexer_module
import mcp_notes.indexing.migration as migration_module
from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.server import create_note, get_indexer, get_search, get_store, search_notes

DEADLOCK_GUARD = 30.0

pytestmark = pytest.mark.usefixtures("tmp_notes_dir", "offline_collection")


@pytest.fixture
def offline_identity_resolved():
    return True


@pytest.fixture
def log(monkeypatch):
    """Record, in order, each bind made under the collection lock and test events."""
    entries: list[object] = []
    real = migration_module.ensure_notes_collection

    async def recording(indexer, *, lock_held=False):
        entries.append(("bind", indexer.embedder))
        return await real(indexer, lock_held=lock_held)

    # collection_operation() imports it at call time, so this sees every locked bind.
    monkeypatch.setattr(migration_module, "ensure_notes_collection", recording)
    return entries


def binds_by(log: list[object], client: EmbeddingClient) -> list[int]:
    """Positions in the log of locked binds made with this client."""
    return [
        position
        for position, entry in enumerate(log)
        if isinstance(entry, tuple) and entry[1] is client
    ]


@asynccontextmanager
async def mutation_in_progress(log: list[object]) -> AsyncIterator[asyncio.Task[None]]:
    """A mutation parked inside the indexer's collection_operation(), holding the lock."""
    indexer = await get_indexer()
    parked, release = asyncio.Event(), asyncio.Event()

    async def hold() -> None:
        async with indexer.collection_operation():
            parked.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    try:
        await asyncio.wait_for(parked.wait(), DEADLOCK_GUARD)
        yield holder
    finally:
        log.append("released")
        release.set()
        await asyncio.wait_for(holder, DEADLOCK_GUARD)


def offline_client(model: str, tmp_path) -> EmbeddingClient:
    """Another client of the offline backend, configured like offline_collection's."""
    dimension = vector_settings.embedding_dim
    client = EmbeddingClient(
        base_url="http://offline.invalid",
        model=model,
        dim=dimension,
        profile="raw",
        cache_namespace="offline",
        cache_path=tmp_path / f"{model}-embeddings.db",
    )
    client._embed_prepared_batch = AsyncMock(
        side_effect=lambda texts: [[1.0] + [0.0] * (dimension - 1) for _ in texts]
    )
    return client


async def publish_replacement(engine, model: str, tmp_path) -> str:
    """Another process's client binds the notes collection, migrating it if needed."""
    client = offline_client(model, tmp_path)
    try:
        other = NoteIndexer(get_store(), engine.storage, client, engine.global_vocab)
        return (await ensure_notes_collection(other)).physical_name
    finally:
        await client.close()


async def bound_note(log):
    """Index a note, then bind the search engine with its first search."""
    created = await create_note(title="Lantern", content="The lighthouse lantern burns kerosene.")
    engine = await get_search()
    results = await search_notes(query="lighthouse")
    assert {r["note"]["id"] for r in results} == {created["id"]}
    assert binds_by(log, engine.embedder)
    assert engine.embedder.resolved_identity() is not None
    bound = await current_embedding_binding(engine.storage, engine.collection_name, engine.embedder)
    assert bound is not None
    return SimpleNamespace(id=created["id"], engine=engine, generation=bound)


async def test_bound_search_does_not_wait_for_a_mutation_in_progress(log):
    note = await bound_note(log)
    log.clear()

    async with mutation_in_progress(log) as mutation:
        results = await asyncio.wait_for(search_notes(query="lighthouse"), DEADLOCK_GUARD)
        assert not mutation.done()

    assert {r["note"]["id"] for r in results} == {note.id}
    assert not any(r["degraded"] for r in results)
    assert not binds_by(log, note.engine.embedder)


async def test_bound_filter_search_and_similar_notes_do_not_wait(log):
    note = await bound_note(log)
    other = await create_note(title="Harbor", content="The harbor pilot reads the tide tables.")
    log.clear()

    async with mutation_in_progress(log) as mutation:
        listed = await asyncio.wait_for(note.engine.search("title:Lantern"), DEADLOCK_GUARD)
        similar = await asyncio.wait_for(note.engine.find_similar(UUID(note.id)), DEADLOCK_GUARD)
        assert not mutation.done()

    assert [str(result.note.id) for result in listed] == [note.id]
    assert [str(result.note.id) for result in similar] == [other["id"]]
    assert not binds_by(log, note.engine.embedder)


async def test_bound_status_read_does_not_wait_for_a_mutation_in_progress(log):
    # Creating the note bound the indexer under the lock.
    await create_note(title="Lantern", content="The lighthouse lantern burns kerosene.")
    indexer = await get_indexer()
    log.clear()

    async with mutation_in_progress(log) as mutation:
        status = await asyncio.wait_for(indexer.get_status(), DEADLOCK_GUARD)
        assert not mutation.done()

    # Counted from the pinned physical generation, not the logical name.
    assert (status.total_notes, status.indexed_notes) == (1, 1)
    assert status.index_healthy
    # Only the parked mutation bound under the lock.
    assert binds_by(log, indexer.embedder) == [0]


async def test_first_search_in_a_process_waits_for_the_lock_then_succeeds(log, monkeypatch):
    created = await create_note(title="Lantern", content="The lighthouse lantern burns kerosene.")
    engine = await get_search()
    assert (
        await current_embedding_binding(engine.storage, engine.collection_name, engine.embedder)
        is None
    )
    requested = asyncio.Event()
    real_lock = indexer_module.embedding_collection_lock

    @asynccontextmanager
    async def watched_lock(storage, logical_name):
        requested.set()
        async with real_lock(storage, logical_name):
            yield

    async with mutation_in_progress(log) as mutation:
        # Only the search asks for the lock from here on.
        monkeypatch.setattr(indexer_module, "embedding_collection_lock", watched_lock)
        search = asyncio.create_task(search_notes(query="lighthouse"))
        await asyncio.wait_for(requested.wait(), DEADLOCK_GUARD)
        assert not search.done()
        assert not mutation.done()

    results = await asyncio.wait_for(search, DEADLOCK_GUARD)
    assert {r["note"]["id"] for r in results} == {created["id"]}
    # The engine bound only once the parked mutation had released the lock.
    assert binds_by(log, engine.embedder) == [log.index("released") + 1]


async def test_search_after_a_replacement_with_another_identity_fails_closed(log, tmp_path):
    note = await bound_note(log)
    replacement = await publish_replacement(note.engine, "another-model", tmp_path)
    assert replacement != note.generation.physical_name
    log.clear()

    for _ in range(2):
        with pytest.raises(EmbeddingMigrationError, match="superseded"):
            await search_notes(query="lighthouse")
    # Both fell back to the locked path, which refused the replacement.
    assert len(binds_by(log, note.engine.embedder)) == 2


async def test_search_after_a_same_identity_replacement_rebinds_under_the_lock(log, tmp_path):
    note = await bound_note(log)
    await publish_replacement(note.engine, "another-model", tmp_path)
    # A client with the engine's identity migrates back, publishing a third generation.
    probe = offline_client("offline-model", tmp_path)
    assert probe.configured_identity() == note.engine.embedder.resolved_identity()
    await probe.close()
    current = await publish_replacement(note.engine, "offline-model", tmp_path)
    assert current != note.generation.physical_name
    log.clear()

    results = await search_notes(query="lighthouse")
    assert {r["note"]["id"] for r in results} == {note.id}
    assert len(binds_by(log, note.engine.embedder)) == 1
    rebound = await current_embedding_binding(
        note.engine.storage, note.engine.collection_name, note.engine.embedder
    )
    assert rebound is not None
    assert rebound.physical_name == current

    # Rebound, the engine reads without the lock again.
    log.clear()
    async with mutation_in_progress(log) as mutation:
        results = await asyncio.wait_for(search_notes(query="lighthouse"), DEADLOCK_GUARD)
        assert not mutation.done()
    assert {r["note"]["id"] for r in results} == {note.id}
    assert not binds_by(log, note.engine.embedder)
