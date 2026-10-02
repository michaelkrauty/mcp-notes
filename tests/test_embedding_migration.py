"""Mixed notes migration against isolated in-memory Qdrant and temporary sources."""

from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector
from vector_core import EmbeddingClient, QdrantStorage, generate_point_id
from vector_core.embeddings.client import CircuitBreakerOpenError
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
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing.chunker import chunk_note, generate_note_summary
from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.indexing.migration import NotesMigration, ensure_notes_collection
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.settings import settings
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
                "index_policy": indexer._index_policy(),
            },
        ),
        (
            f"chunk:{note.id}:0",
            {
                "type": "chunk",
                "note_id": str(note.id),
                "chunk_index": 0,
                "title": note.title,
                "content": chunk_note(parsed)[0].content,
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
    records, _ = await client.scroll(collection, with_payload=True, with_vectors=True, limit=1000)
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
    assert migrated[generate_point_id(f"note:{note.id}")].payload[
        "embedding_text"
    ] == generate_note_summary(parse_note(source))


@pytest.mark.parametrize("deleted", [False, True])
async def test_symlinked_source_root_reconciles_owned_notes(corpus, deleted):
    indexer, note = corpus
    root = indexer.note_store.notes_dir
    target = root.with_name("real-notes")
    root.rename(target)
    root.symlink_to(target, target_is_directory=True)
    if deleted:
        indexer.note_store.get_note_path(note.id).unlink()
        with pytest.raises(
            EmbeddingMigrationError, match="complete retained embedding input is missing"
        ):
            await ensure_notes_collection(indexer)
        assert (
            await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
            == indexer.logical_collection_name
        )
        return
    else:
        indexer.note_store.update(note.id, content="Changed under symlinked root")
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    summary_id = generate_point_id(f"note:{note.id}")
    if deleted:
        assert summary_id not in result
    else:
        assert "Changed under symlinked root" in result[summary_id].payload["embedding_text"]


async def test_symlink_entry_below_root_keeps_retained_fallback(corpus):
    indexer, note = corpus
    path = indexer.note_store.get_note_path(note.id)
    target = indexer.note_store.base_dir / "external.md"
    path.rename(target)
    path.symlink_to(target)
    original = await points(indexer, indexer.logical_collection_name)
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    assert result.keys() == original.keys()
    assert (
        result[generate_point_id(f"note:{note.id}")].payload["embedding_text_source"]
        == "legacy-note-metadata"
    )


@pytest.mark.parametrize("change", ["edit", "delete", "unchanged"])
async def test_modern_orphan_chunks_reconcile_owned_source(corpus, change):
    indexer, note = corpus
    await indexer._index_note(parse_note(note.content), None)
    client = await indexer.storage.get_client()
    await client.delete(
        indexer.logical_collection_name,
        points_selector=[generate_point_id(f"note:{note.id}")],
        wait=True,
    )
    if change == "delete":
        indexer.note_store.get_note_path(note.id).unlink()
    elif change == "edit":
        indexer.note_store.update(note.id, content="Edited orphan source")
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    owned = [
        point.payload for point in result.values() if point.payload.get("note_id") == str(note.id)
    ]
    if change == "delete":
        assert {payload["type"] for payload in owned} == {"chunk"}
        assert all("Original content" in stored_embedding_text(payload) for payload in owned)
    else:
        assert {payload["type"] for payload in owned} == {"note", "chunk"}
        if change == "edit":
            assert all(
                "Edited orphan source" in stored_embedding_text(payload) for payload in owned
            )


async def test_restore_reads_current_path_only_after_mutation_lock(monkeypatch):
    locked = False

    @asynccontextmanager
    async def operation():
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def read_note(*args):
        assert locked
        return SimpleNamespace(title="Current")

    def current_path(*args):
        assert locked

    indexer = SimpleNamespace(collection_operation=operation)
    getter = AsyncMock(return_value=indexer)
    store = SimpleNamespace(read=read_note, get_note_path=current_path)
    git = MagicMock()
    git.restore_version.return_value = None
    git.is_note_deleted_at.return_value = False
    monkeypatch.setattr(mutation, "get_indexer", getter)
    monkeypatch.setattr(versioning, "get_indexer", getter)
    monkeypatch.setattr(versioning, "get_store", lambda: store)
    monkeypatch.setattr(versioning, "get_git", lambda: git)
    await versioning.restore_note_version(str(uuid4()), "version")
    assert not locked


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


async def test_forced_reindex_preserves_foreign_note_root(corpus, tmp_path):
    indexer, _ = corpus
    other_store = NoteStore(tmp_path / "other-notes")
    foreign_note = other_store.create("Foreign", "Foreign source body")
    other = NoteIndexer(other_store, indexer.storage, indexer.embedder, indexer.global_vocab)
    other._collection_name = indexer.logical_collection_name
    await other._index_note(parse_note(foreign_note.content), None)
    generation = await ensure_notes_collection(indexer)
    before = await points(indexer, generation.physical_name)
    foreign_ids = {
        point_id
        for point_id, point in before.items()
        if point.payload.get("note_id") == str(foreign_note.id)
    }
    assert len(foreign_ids) == 2
    assert (await indexer.index_all(force=True)).index_healthy
    after = await points(indexer, generation.physical_name)
    assert all(after[point_id] == before[point_id] for point_id in foreign_ids)


@pytest.mark.parametrize("query", ["Original", ""])
async def test_identity_outage_uses_retained_generation_without_dense_query(corpus, query):
    indexer, _ = corpus
    generation = await ensure_notes_collection(indexer)
    indexer.embedder.resolve_identity.side_effect = CircuitBreakerOpenError("isolated", 60)
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    client = await indexer.storage.get_client()
    client.query_points = AsyncMock(wraps=client.query_points)
    await engine.search(query)
    indexer.embedder.embed_single_cached.assert_not_awaited()
    if query:
        assert client.query_points.await_args.args[0] == generation.physical_name
        assert client.query_points.await_args.kwargs["using"] == "sparse"
        assert "prefetch" not in client.query_points.await_args.kwargs
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == generation.physical_name
    )


async def test_migration_failure_does_not_fall_back_to_retained_reads(corpus):
    indexer, _ = corpus
    indexer.embedder.resolve_identity.side_effect = EmbeddingMigrationError("invalid identity")
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    with pytest.raises(EmbeddingMigrationError, match="invalid identity"):
        await engine.search("Original")
    indexer.embedder.embed_single_cached.assert_not_awaited()


async def test_owner_reindexes_metadata_fallback_after_foreign_first_migration(corpus, tmp_path):
    indexer, note = corpus
    foreign_store = NoteStore(tmp_path / "foreign-first")
    foreign_store.ensure_directories()
    foreign = NoteIndexer(foreign_store, indexer.storage, indexer.embedder, indexer.global_vocab)
    foreign._collection_name = indexer.logical_collection_name
    generation = await ensure_notes_collection(foreign)
    summary_id = generate_point_id(f"note:{note.id}")
    before = (await points(indexer, generation.physical_name))[summary_id].payload
    assert before["embedding_text_source"] == "legacy-note-metadata"
    assert "Original content" not in before["embedding_text"]
    assert (await indexer.index_all()).index_healthy
    after = (await points(indexer, generation.physical_name))[summary_id].payload
    assert "Original content" in after["embedding_text"]
    assert "embedding_text_source" not in after


async def test_similar_lookup_uses_stored_vector_during_identity_outage(corpus):
    indexer, note = corpus
    generation = await ensure_notes_collection(indexer)
    indexer.embedder.resolve_identity.side_effect = CircuitBreakerOpenError("isolated", 60)
    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    client = await indexer.storage.get_client()
    client.scroll = AsyncMock(wraps=client.scroll)
    client.query_points_groups = AsyncMock(wraps=client.query_points_groups)
    await engine.find_similar(note.id)
    indexer.embedder.embed_single_cached.assert_not_awaited()
    assert client.scroll.await_args.args[0] == generation.physical_name
    assert client.query_points_groups.await_args.args[0] == generation.physical_name
    assert client.query_points_groups.await_args.kwargs["using"] == "dense"
    stored = (await points(indexer, generation.physical_name))[
        generate_point_id(f"chunk:{note.id}:0")
    ].vector["dense"]
    assert client.query_points_groups.await_args.kwargs["query"] == stored


async def test_unchanged_legacy_chunk_recovers_full_source_input(corpus):
    indexer, note = corpus
    parsed = parse_note(indexer.note_store.read(note.id).content)
    full_text = chunk_note(parsed)[0].content
    chunk_id = generate_point_id(f"chunk:{note.id}:0")
    client = await indexer.storage.get_client()
    await client.set_payload(
        indexer.logical_collection_name, {"content": full_text[:8]}, points=[chunk_id]
    )
    original = await points(indexer, indexer.logical_collection_name)
    generation = await ensure_notes_collection(indexer)
    migrated = await points(indexer, generation.physical_name)
    assert stored_embedding_text(migrated[chunk_id].payload) == full_text
    assert migrated[generate_point_id(f"note:{note.id}")].payload["source_reindex_pending"]
    assert any(full_text in call.args[0] for call in indexer.embedder.embed_all.await_args_list)
    await indexer.index_all()
    assert (
        stored_embedding_text((await points(indexer, generation.physical_name))[chunk_id].payload)
        == full_text
    )
    assert await points(indexer, indexer.logical_collection_name) == original


async def test_missing_legacy_source_fails_closed_instead_of_inferred_deletion(corpus):
    indexer, note = corpus
    before = await points(indexer, indexer.logical_collection_name)
    indexer.note_store.get_note_path(note.id).unlink()
    with pytest.raises(
        EmbeddingMigrationError, match="complete retained embedding input is missing"
    ):
        await ensure_notes_collection(indexer)
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == indexer.logical_collection_name
    )
    assert await points(indexer, indexer.logical_collection_name) == before


async def test_owned_retained_metadata_only_summary_with_missing_source_fails_closed(
    corpus, tmp_path
):
    indexer, note = corpus
    foreign_store = NoteStore(tmp_path / "foreign-first")
    foreign_store.ensure_directories()
    foreign = NoteIndexer(foreign_store, indexer.storage, indexer.embedder, indexer.global_vocab)
    foreign._collection_name = indexer.logical_collection_name
    original = await ensure_notes_collection(foreign)
    before = await points(indexer, original.physical_name)
    indexer.note_store.get_note_path(note.id).unlink()
    replacement = NoteIndexer(
        indexer.note_store,
        indexer.storage,
        model(namespace="next-deployment"),
        indexer.global_vocab,
    )
    with pytest.raises(EmbeddingMigrationError, match="legacy metadata-only summaries"):
        await ensure_notes_collection(replacement)
    assert (
        await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        == original.physical_name
    )
    assert await points(indexer, original.physical_name) == before
    await replacement.embedder.close()


async def test_changed_chunk_boundaries_rebuild_complete_unchanged_note(corpus, monkeypatch):
    indexer, note = corpus
    indexer.note_store.update(
        note.id, content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    )
    parsed = parse_note(indexer.note_store.read(note.id).content)
    await indexer._index_note(parsed, None)
    client = await indexer.storage.get_client()
    await client.delete_payload(
        indexer.logical_collection_name,
        keys=["embedding_text", "embedding_text_field", "embedding_fragment"],
        points=[generate_point_id(f"chunk:{note.id}:0")],
        wait=True,
    )
    monkeypatch.setattr(settings, "max_chunk_chars", 35)
    expected = [chunk.content for chunk in chunk_note(parsed)]
    assert len(expected) > 1
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    chunks = sorted(
        (point.payload for point in result.values() if point.payload["type"] == "chunk"),
        key=lambda payload: payload["chunk_index"],
    )
    assert [stored_embedding_text(chunk) for chunk in chunks] == expected
    assert result[generate_point_id(f"note:{note.id}")].payload["source_reindex_pending"]


async def test_standalone_fact_index_prepares_note_aware_generation(corpus, monkeypatch):
    indexer, note = corpus
    indexer.note_store.update(note.id, content="Changed before fact indexing")

    async def index_facts(**kwargs):
        active = await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
        assert active != indexer.logical_collection_name
        result = await points(indexer, active)
        assert (
            "Changed before fact indexing"
            in result[generate_point_id(f"note:{note.id}")].payload["embedding_text"]
        )
        return {"indexed": 0}

    monkeypatch.setattr(mutation, "get_indexer", AsyncMock(return_value=indexer))
    monkeypatch.setattr(
        facts,
        "get_fact_indexer",
        AsyncMock(return_value=SimpleNamespace(index_all=index_facts)),
    )
    assert await facts.index_facts() == {"indexed": 0}


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


async def test_migration_reads_one_snapshot_for_all_notes_and_finalizer(corpus, monkeypatch):
    indexer, original = corpus
    added_ids = []
    for number in range(5):
        note = indexer.note_store.create(f"Note {number}", f"Original content {number}")
        await indexer._index_note(parse_note(note.content), None)
        added_ids.extend(
            [
                generate_point_id(f"note:{note.id}"),
                generate_point_id(f"chunk:{note.id}:0"),
            ]
        )
    client = await indexer.storage.get_client()
    await client.delete_payload(
        indexer.logical_collection_name,
        keys=["embedding_text", "embedding_text_field", "embedding_fragment"],
        points=added_ids,
        wait=True,
    )
    indexer.note_store.update(original.id, content="Changed content")
    snapshot = NotesMigration._snapshot
    scans = 0

    def counted_snapshot(coordinator):
        nonlocal scans
        scans += 1
        return snapshot(coordinator)

    monkeypatch.setattr(NotesMigration, "_snapshot", counted_snapshot)
    generation = await ensure_notes_collection(indexer)
    assert scans == 1
    assert len(await points(indexer, generation.physical_name)) == 14


async def test_failed_migration_retry_reads_new_source_snapshot(corpus):
    indexer, note = corpus
    indexer.note_store.update(note.id, content="First edit")
    calls = 0

    async def fail_rebuild(texts, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("candidate interrupted")
        return [[1.0] + [0.0] * 127 for _ in texts]

    indexer.embedder.embed_all.side_effect = fail_rebuild
    with pytest.raises(EmbeddingMigrationError, match="candidate interrupted"):
        await ensure_notes_collection(indexer)
    indexer.note_store.update(note.id, content="Later edit after failed migration")
    indexer.embedder.embed_all.side_effect = lambda texts, **kwargs: [
        [1.0] + [0.0] * 127 for _ in texts
    ]
    generation = await ensure_notes_collection(indexer)
    result = await points(indexer, generation.physical_name)
    text = result[generate_point_id(f"note:{note.id}")].payload["embedding_text"]
    assert "Later edit after failed migration" in text
    assert "First edit" not in text
