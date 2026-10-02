"""Full note span and fragment lifecycles using isolated retained sources."""

from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector
from vector_core import EmbeddingClient, QdrantStorage, generate_point_id
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.settings import settings as vector_settings
from vector_core.storage.embedding_fragments import (
    fragment_point,
    is_derived_fragment,
    upsert_fragment_group,
)
from vector_core.storage.embedding_migration import (
    active_embedding_collection,
    ensure_embedding_collection,
)
from vector_core.storage.embedding_sources import stored_embedding_text

from mcp_notes.indexing import indexer as indexer_module
from mcp_notes.indexing.chunker import chunk_note
from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.parser import parse_note


def isolated_embedder(capacity, namespace="first"):
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


@pytest.fixture
async def notes_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_settings, "collection_name", None)
    monkeypatch.setattr(settings, "max_chunk_chars", 100000)
    monkeypatch.setattr(settings, "section_overlap_chars", 0)
    store = NoteStore(tmp_path / "source")
    note = store.create("Context", "paragraph " * 4000 + "TAIL_MARKER")
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=128)
    storage._client = AsyncQdrantClient(location=":memory:")
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    embedder = isolated_embedder(256)
    indexer = NoteIndexer(store, storage, embedder, vocab)
    yield indexer, note
    await storage.close()
    await embedder.close()
    vocab.close()


async def payloads(indexer, collection):
    return [
        payload
        for payload in await indexer.storage.scroll_points(collection, max_results=0)
        if payload.get("type") != "__metadata__"
    ]


async def test_large_note_retains_every_body_span_and_full_highlight_text(notes_index):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    result = await payloads(
        indexer, await active_embedding_collection(indexer.storage, indexer.logical_collection_name)
    )
    chunks = sorted(
        (p for p in result if p.get("type") == "chunk" and not is_derived_fragment(p)),
        key=lambda p: p["chunk_index"],
    )
    body = "paragraph " * 4000 + "TAIL_MARKER"
    assert "".join(body[p["start_char"] : p["end_char"]] for p in chunks) == body
    assert all(stored_embedding_text(p) == p["content"] for p in chunks)
    assert any("TAIL_MARKER" in p["content"] for p in chunks)
    assert any(is_derived_fragment(p) and p["type"] == "note" for p in result)
    assert all(
        len(text) <= 256
        for call in indexer.embedder._embed_prepared_batch.await_args_list
        for text in call.args[0]
    )


async def test_shorter_replacement_retires_chunks_and_auxiliary_children(notes_index):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    indexer.note_store.update(note.id, content="Short replacement")
    await indexer.index_note(note.id)
    generation = await ensure_notes_collection(indexer)
    result = await payloads(indexer, generation.physical_name)
    assert {p["type"] for p in result} == {"note", "chunk"}
    assert len(result) == 2
    assert not any(is_derived_fragment(p) for p in result)
    assert all("TAIL_MARKER" not in stored_embedding_text(p) for p in result)


async def test_full_chunk_payload_and_highlight_survive_old_excerpt_cutoff(notes_index):
    indexer, note = notes_index
    wide = isolated_embedder(100000)
    indexer.embedder = wide
    try:
        await indexer.index_note(note.id)
        generation = await ensure_notes_collection(indexer)
        result = await payloads(indexer, generation.physical_name)
        chunks = [p for p in result if p["type"] == "chunk"]
        assert len(chunks) == 1
        assert len(chunks[0]["content"]) > 30000
        assert chunks[0]["content"].endswith("TAIL_MARKER")
        engine = NoteSearchEngine(indexer.note_store, indexer.storage, wide, indexer.global_vocab)
        matches = await engine.search("TAIL_MARKER", mode="chunk")
        assert len(matches) == 1
        assert any("TAIL_MARKER" in highlight for highlight in matches[0].highlights)
    finally:
        await wide.close()


async def test_capacity_migration_preserves_retained_input_without_original_files(notes_index):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    original = await ensure_notes_collection(indexer)
    before = await payloads(indexer, original.physical_name)
    indexer.note_store.notes_dir.rename(indexer.note_store.base_dir / "unavailable")
    changed = isolated_embedder(128, namespace="second")
    # A retained-source migration needs no file reconciliation or fabricated input.
    migrated = await ensure_embedding_collection(
        indexer.storage,
        indexer.logical_collection_name,
        changed,
        text_resolver=AsyncMock(side_effect=AssertionError("Full text should be retained")),
        vectorize=indexer.global_vocab.vectorize_document,
    )
    after = await payloads(indexer, migrated.physical_name)
    originals = sorted(stored_embedding_text(p) for p in before if not is_derived_fragment(p))
    retained = sorted(stored_embedding_text(p) for p in after if not is_derived_fragment(p))
    assert retained == originals
    assert any(is_derived_fragment(p) for p in after)
    assert await payloads(indexer, original.physical_name) == before
    await changed.close()


async def test_source_policy_only_change_reindexes_unchanged_note(notes_index, monkeypatch):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    parsed = parse_note(indexer.note_store.read(note.id).content)
    source_hash = indexer._hash_note(parsed, None)
    monkeypatch.setattr(settings, "max_chunk_chars", 180)
    assert indexer._hash_note(parsed, None) == source_hash
    async with indexer.collection_operation():
        assert (await indexer._get_indexed_hashes())[str(note.id)] != source_hash
    status = await indexer.index_all()
    assert status.index_healthy
    generation = await ensure_notes_collection(indexer)
    result = await payloads(indexer, generation.physical_name)
    summaries = [p for p in result if p["type"] == "note" and not is_derived_fragment(p)]
    assert len(summaries) == 1
    assert summaries[0]["note_hash"] == source_hash
    assert summaries[0]["index_policy"] == indexer._index_policy()
    assert all(len(p["content"]) <= 180 for p in result if p["type"] == "chunk")


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["chunk", "cleanup"])
async def test_partial_note_write_remains_pending_and_incremental_retry_repairs(
    notes_index, monkeypatch, existing, stage
):
    indexer, note = notes_index
    if existing:
        await indexer.index_note(note.id)
        indexer.note_store.update(note.id, content="new passage " * 300 + "NEW_TAIL")
    parsed = parse_note(indexer.note_store.read(note.id).content)
    original_write = indexer_module.upsert_fragment_group
    original_cleanup = indexer._delete_orphan_chunks

    async def fail_chunk(storage, collection, points):
        if points[0].payload.get("type") == "chunk" and points[0].payload.get("chunk_index") == 1:
            raise RuntimeError("injected chunk write failure")
        await original_write(storage, collection, points)

    async def fail_cleanup(*args):
        raise RuntimeError("injected cleanup failure")

    if stage == "chunk":
        monkeypatch.setattr(indexer_module, "upsert_fragment_group", fail_chunk)
    else:
        monkeypatch.setattr(indexer, "_delete_orphan_chunks", fail_cleanup)
    with pytest.raises(RuntimeError, match="injected"):
        await indexer.index_note(note.id)

    generation = await ensure_notes_collection(indexer)
    partial = await payloads(indexer, generation.physical_name)
    summary = next(p for p in partial if p["type"] == "note" and not is_derived_fragment(p))
    assert summary["source_reindex_pending"] is True
    assert summary["note_hash"] == summary["index_policy"] == ""
    async with indexer.collection_operation():
        assert (await indexer._get_indexed_hashes())[str(note.id)] == ""

    monkeypatch.setattr(indexer_module, "upsert_fragment_group", original_write)
    monkeypatch.setattr(indexer, "_delete_orphan_chunks", original_cleanup)
    assert (await indexer.index_all()).index_healthy
    repaired = await payloads(indexer, generation.physical_name)
    summary = next(p for p in repaired if p["type"] == "note" and not is_derived_fragment(p))
    assert summary["note_hash"] == indexer._hash_note(parsed, None)
    assert summary["index_policy"] == indexer._index_policy()
    assert summary["source_reindex_pending"] is False
    chunks = sorted((p for p in repaired if p["type"] == "chunk"), key=lambda p: p["chunk_index"])
    assert len(chunks) == len(chunk_note(parsed, indexer.embedder))
    assert "".join(parsed.body[p["start_char"] : p["end_char"]] for p in chunks) == parsed.body


async def test_filter_only_notes_exclude_children_before_candidate_limit(notes_index):
    indexer, note = notes_index
    other = indexer.note_store.create("Other", "Other body")
    empty_sparse = SparseVector(indices=[], values=[])
    async with indexer.collection_operation() as generation:
        for note_id, title, text in [
            (note.id, "Many fragments", "auxiliary source " * 2000),
            (other.id, "Other", "Other body"),
        ]:
            points = await fragment_point(
                indexer.embedder,
                point_id=generate_point_id(f"note:{note_id}"),
                payload={"type": "note", "note_id": str(note_id), "title": title},
                text=text,
                sparse=empty_sparse,
                vectorize=lambda text: empty_sparse,
            )
            if note_id == note.id:
                assert len(points) > 26
            await upsert_fragment_group(indexer.storage, generation.physical_name, points)
        raw = await payloads(indexer, generation.physical_name)
        assert sum(is_derived_fragment(p) for p in raw) > 26
        client = await indexer.storage.get_client()
        await client.delete_payload(
            generation.physical_name,
            keys=["embedding_fragment"],
            points=[generate_point_id(f"note:{other.id}")],
            wait=True,
        )

    engine = NoteSearchEngine(
        indexer.note_store, indexer.storage, indexer.embedder, indexer.global_vocab
    )
    results = await engine.search("", mode="note", limit=2)
    assert {result.note.id for result in results} == {note.id, other.id}
    assert len(results) == 2


@pytest.mark.parametrize("missing", ["file", "root"])
@pytest.mark.parametrize("orphan", [False, True])
async def test_notes_finalizer_preserves_retained_sources_with_missing_files(
    notes_index, missing, orphan
):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    original = await ensure_notes_collection(indexer)
    client = await indexer.storage.get_client()
    if orphan:
        await client.delete(
            original.physical_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="note")),
                    FieldCondition(key="note_id", match=MatchValue(value=str(note.id))),
                ]
            ),
            wait=True,
        )
    before = await payloads(indexer, original.physical_name)
    path = indexer.note_store.get_note_path(note.id)
    if missing == "file":
        path.unlink()
    else:
        indexer.note_store.notes_dir.rename(indexer.note_store.base_dir / "unavailable")
    assert indexer.note_store.get_note_path(note.id) == path
    changed = isolated_embedder(128, namespace="second")
    replacement = NoteIndexer(indexer.note_store, indexer.storage, changed, indexer.global_vocab)
    try:
        engine = NoteSearchEngine(
            indexer.note_store, indexer.storage, changed, indexer.global_vocab
        )
        results = await engine.search("TAIL_MARKER", mode="chunk")
        assert any(
            "TAIL_MARKER" in highlight for result in results for highlight in result.highlights
        )
        migrated = await ensure_notes_collection(replacement)
        assert migrated.physical_name != original.physical_name
        after = await payloads(indexer, migrated.physical_name)
        assert sorted(
            stored_embedding_text(p) for p in after if not is_derived_fragment(p)
        ) == sorted(stored_embedding_text(p) for p in before if not is_derived_fragment(p))
        assert {p["note_id"] for p in after} == {str(note.id)}
        assert await payloads(indexer, original.physical_name) == before
    finally:
        await changed.close()


async def test_explicit_note_index_deletion_still_removes_complete_group(notes_index):
    indexer, note = notes_index
    await indexer.index_note(note.id)
    generation = await ensure_notes_collection(indexer)
    indexer.note_store.delete(note.id)
    await indexer.delete_note_index(note.id)
    assert await payloads(indexer, generation.physical_name) == []
