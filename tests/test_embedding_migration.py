"""Mixed notes migration against isolated in-memory Qdrant and temporary sources."""

from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector
from vector_core import EmbeddingClient, QdrantStorage, generate_point_id
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.facts import FactStore
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_migration import (
    EmbeddingMigrationError,
    active_embedding_collection,
    ensure_embedding_collection,
    resolve_shared_embedding_text,
)

from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.parser import parse_note
from mcp_notes.tools import categories, facts, mutation, notes, tags, versioning


def model(name="new-model", dimension=128, namespace="deployment-1"):
    client = EmbeddingClient(model=name, dim=dimension, cache_namespace=namespace)
    client.resolve_identity = AsyncMock(
        return_value=EmbeddingIdentity(
            model=name, namespace=namespace, endpoint="http://isolated.invalid", dimension=dimension
        )
    )
    vector = [1.0] + [0.0] * (dimension - 1)
    client.embed_all = AsyncMock(side_effect=lambda texts, **kwargs: [vector for _ in texts])
    client.embed_single_cached = AsyncMock(return_value=vector)
    return client


@pytest.fixture
async def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(vector_settings, "collection_name", None)
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=128)
    storage._client = AsyncQdrantClient(location=":memory:")
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    vocab.register_codebase("notes", [{"original", "content"}])
    store = NoteStore(tmp_path / "source")
    note = store.create("Original", "Original content")
    indexer = NoteIndexer(store, storage, model(), vocab)
    logical = indexer.logical_collection_name
    await storage.create_collection(logical)
    parsed = parse_note(note.content)
    payloads = [
        (
            f"note:{note.id}",
            {
                "type": "note",
                "note_id": str(note.id),
                "title": note.title,
                "tags": [],
                "category": None,
                "note_hash": indexer._hash_note(parsed, None),
            },
        ),
        (
            f"chunk:{note.id}:0",
            {
                "type": "chunk",
                "note_id": str(note.id),
                "chunk_index": 0,
                "title": note.title,
                "content": "Original content",
                "tags": [],
            },
        ),
        ("fact:retained", {"type": "fact", "fact_id": "retained", "content": "A known fact"}),
        (
            "glossary:retained",
            {
                "type": "glossary",
                "glossary_id": "retained",
                "embedding_text": "A definition",
            },
        ),
    ]
    raw = await storage.get_client()
    await raw.upsert(
        logical,
        [
            PointStruct(
                id=generate_point_id(key),
                payload=payload,
                vector={
                    "dense": [0.0, 1.0] + [0.0] * 126,
                    "sparse": SparseVector(indices=[1], values=[1.0]),
                },
            )
            for key, payload in payloads
        ],
        wait=True,
    )
    yield indexer, note
    await storage.close()
    await indexer.embedder.close()
    vocab.close()


async def points(indexer, collection):
    client = await indexer.storage.get_client()
    records, _ = await client.scroll(collection, with_payload=True, with_vectors=True)
    return {point.id: point for point in records if point.id != 0}


async def test_same_dimension_model_change_preserves_mixed_sparse_and_sources(corpus):
    indexer, note = corpus
    original = await points(indexer, indexer.logical_collection_name)
    source = indexer.note_store.read(note.id).content
    generation = await ensure_notes_collection(indexer)
    migrated = await points(indexer, generation.physical_name)
    assert migrated.keys() == original.keys()
    for point_id, record in original.items():
        assert migrated[point_id].vector["sparse"] == record.vector["sparse"]
        assert migrated[point_id].vector["dense"] != record.vector["dense"]
    assert await points(indexer, indexer.logical_collection_name) == original
    assert indexer.note_store.read(note.id).content == source


async def test_external_edit_rebuilds_whole_group_without_stale_chunks(corpus):
    indexer, note = corpus
    client = await indexer.storage.get_client()
    stale = generate_point_id(f"chunk:{note.id}:9")
    await client.upsert(
        indexer.logical_collection_name,
        [
            PointStruct(
                id=stale,
                payload={
                    "type": "chunk",
                    "note_id": str(note.id),
                    "chunk_index": 9,
                    "content": "Stale section",
                },
                vector={"dense": [1.0] * 128, "sparse": SparseVector(indices=[1], values=[1.0])},
            )
        ],
        wait=True,
    )
    indexer.note_store.update(note.id, content="Changed source with new words")
    before_stats = indexer.global_vocab.get_codebase_stats("notes")
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    assert stale not in result
    summary = result[generate_point_id(f"note:{note.id}")].payload
    assert "Changed source" in summary["embedding_text"]
    assert summary["note_hash"] == ""
    assert summary["source_reindex_pending"] is True
    assert indexer.global_vocab.get_codebase_stats("notes") == before_stats
    assert len(result) == 4
    status = await indexer.index_all()
    assert status.index_healthy
    result = await points(indexer, generation.physical_name)
    assert result[generate_point_id(f"note:{note.id}")].payload["note_hash"]
    assert not result[generate_point_id(f"note:{note.id}")].payload["source_reindex_pending"]


async def test_confirmed_external_delete_omits_note_group_only(corpus):
    indexer, note = corpus
    indexer.note_store.get_note_path(note.id).unlink()
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    assert {point.payload["type"] for point in result.values()} == {"fact", "glossary"}
    assert len(await points(indexer, indexer.logical_collection_name)) == 4


async def test_failed_source_rebuild_does_not_activate_or_change_vocabulary(corpus):
    indexer, note = corpus
    indexer.note_store.update(note.id, content="Edited")
    original = await points(indexer, indexer.logical_collection_name)
    before_stats = indexer.global_vocab.get_codebase_stats("notes")
    calls = 0

    async def fail_finalizer(texts, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("rebuild interrupted")
        return [[1.0] + [0.0] * 127 for _ in texts]

    indexer.embedder.embed_all.side_effect = fail_finalizer
    with pytest.raises(EmbeddingMigrationError, match="rebuild interrupted"):
        await ensure_notes_collection(indexer)
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == indexer.logical_collection_name
    )
    assert await points(indexer, indexer.logical_collection_name) == original
    assert indexer.global_vocab.get_codebase_stats("notes") == before_stats


async def test_unavailable_source_root_is_not_treated_as_deletion(corpus):
    indexer, _ = corpus
    indexer.note_store.notes_dir.rename(indexer.note_store.base_dir / "unavailable")
    with pytest.raises(EmbeddingMigrationError, match="unavailable"):
        await ensure_notes_collection(indexer)
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == indexer.logical_collection_name
    )


async def test_search_first_migrates_and_embeds_query_with_query_role(corpus):
    indexer, _ = corpus
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    results = await engine.search("Original")
    assert results
    indexer.embedder.embed_single_cached.assert_awaited_once_with("Original", role="query")
    active = await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
    assert active != indexer.logical_collection_name


async def test_stale_client_cannot_write_after_model_cutover(corpus):
    indexer, note = corpus
    first = await ensure_notes_collection(indexer)
    newer = NoteIndexer(
        indexer.note_store, indexer.storage, model("another-model"), indexer.global_vocab
    )
    second = await ensure_notes_collection(newer)
    assert second.physical_name != first.physical_name
    with pytest.raises(EmbeddingMigrationError, match="superseded"):
        await indexer.index_note(note.id)
    assert len(await points(indexer, second.physical_name)) == 4


async def test_shared_notes_without_local_source_are_retained(corpus, tmp_path):
    indexer, _ = corpus
    other_store = NoteStore(tmp_path / "other-source")
    other_store.ensure_directories()
    foreign = NoteIndexer(other_store, indexer.storage, indexer.embedder, indexer.global_vocab)
    foreign._collection_name = indexer.logical_collection_name
    generation = await ensure_notes_collection(foreign)
    result = await points(indexer, generation.physical_name)
    assert len(result) == 4
    assert {point.payload["type"] for point in result.values()} == {
        "note",
        "chunk",
        "fact",
        "glossary",
    }


async def test_docs_first_generation_can_be_searched_without_note_sources(corpus, monkeypatch):
    indexer, _ = corpus
    generation = await ensure_embedding_collection(
        indexer.storage,
        indexer.logical_collection_name,
        indexer.embedder,
        resolve_shared_embedding_text,
    )
    indexer.note_store.notes_dir.rename(indexer.note_store.base_dir / "unavailable")
    engine = NoteSearchEngine(indexer.note_store, indexer.storage, model(), indexer.global_vocab)
    monkeypatch.setattr(engine, "_collection_name", indexer.logical_collection_name)
    assert await engine.search("Original")
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == generation.physical_name
    )


@pytest.mark.parametrize("dimension,namespace", [(256, "deployment-1"), (128, "deployment-2")])
async def test_single_write_first_handles_dimension_or_deployment_change(
    corpus, dimension, namespace
):
    indexer, note = corpus
    original = await ensure_notes_collection(indexer)
    changed = NoteIndexer(
        indexer.note_store,
        indexer.storage,
        model(dimension=dimension, namespace=namespace),
        indexer.global_vocab,
    )
    await changed.index_note(note.id)
    active = await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
    assert active != original.physical_name
    result = await points(indexer, active)
    assert len(result) == 4
    assert all(len(point.vector["dense"]) == dimension for point in result.values())


@pytest.mark.parametrize(
    "tool,kwargs",
    [
        (notes.delete_note, {"note_id": "invalid"}),
        (notes.update_note, {"note_id": "invalid", "title": "New"}),
        (versioning.restore_note_version, {"note_id": "invalid", "version_id": "abc"}),
        (
            facts.add_fact,
            {
                "subject": "A",
                "predicate": "knows",
                "object": "B",
                "valid_from": "2026-02-01",
                "valid_to": "2026-01-01",
            },
        ),
    ],
)
async def test_invalid_input_never_initializes_migration(tool, kwargs, monkeypatch):
    getter = AsyncMock(side_effect=AssertionError("must validate before backend access"))
    monkeypatch.setattr(mutation, "get_indexer", getter)
    result = await tool(**kwargs)
    assert result["error_code"] in {"invalid_uuid", "invalid_input"}
    getter.assert_not_awaited()


async def test_invalid_effective_fact_range_never_initializes_migration(tmp_path, monkeypatch):
    store = FactStore(db_path=tmp_path / "facts.db")
    fact = store.create("A", "knows", "B", valid_from=date(2026, 2, 1))
    getter = AsyncMock(side_effect=AssertionError("must validate before backend access"))
    monkeypatch.setattr(mutation, "get_indexer", getter)
    monkeypatch.setattr(facts, "get_fact_store", lambda: store)
    result = await facts.update_fact(str(fact.id), valid_to="2026-01-01")
    assert result["error_code"] == "invalid_input"
    getter.assert_not_awaited()
    assert store.read(fact.id).valid_to is None
    store.close()


@pytest.mark.parametrize(
    "module,tool,args",
    [
        (tags, tags.rename_tag, ("old", "new")),
        (tags, tags.merge_tags, (["old"], "new")),
        (categories, categories.move_category, ("old", "new")),
    ],
)
async def test_bulk_source_snapshot_is_taken_under_writer_lock(module, tool, args, monkeypatch):
    locked = False

    @asynccontextmanager
    async def operation():
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def snapshot():
        assert locked, "source snapshot must not race a competing source mutation"
        return []

    indexer = SimpleNamespace(collection_operation=operation)
    store = MagicMock()
    store.list_all.side_effect = snapshot
    getter = AsyncMock(return_value=indexer)
    monkeypatch.setattr(mutation, "get_indexer", getter)
    monkeypatch.setattr(module, "get_indexer", getter)
    monkeypatch.setattr(module, "get_store", lambda: store)
    monkeypatch.setattr(module, "get_git", MagicMock())
    assert await tool(*args) == {"updated_count": 0}
    assert not locked
