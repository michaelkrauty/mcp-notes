"""Knowledge searches roll up real fragments and hydrate canonical display fields."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from vector_core import EmbeddingClient, QdrantStorage, generate_point_id
from vector_core.embeddings.client import CircuitBreakerOpenError
from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.embedding_fragments import fragment_point, upsert_fragment_group
from vector_core.storage.hybrid import SearchResult as HybridResult

from mcp_notes.search.engine import NoteSearchEngine


def entity_source(entity_type, entity_id, body):
    payload = {"type": entity_type, f"{entity_type}_id": str(entity_id)}
    if entity_type == "fact":
        payload.update(
            subject="Alice",
            predicate="references",
            object="Complete canonical object",
            context=body,
            subject_type="person",
            object_type="document",
        )
        return payload, f"Alice references Complete canonical object {body}"
    payload.update(term="TERM", expansion="Complete canonical expansion", definition=body)
    return payload, f"TERM Complete canonical expansion {body}"


@pytest.mark.parametrize("entity_type", ["fact", "glossary"])
@pytest.mark.parametrize("degraded", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_filtered_tail_fragments_preserve_distinct_entities_and_display_fields(
    tmp_path, monkeypatch, entity_type, degraded, mixed
):
    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    embedder = EmbeddingClient(
        model="isolated",
        dim=2,
        profile="raw",
        max_text_chars=128,
        max_input_bytes=0,
        max_input_tokens=None,
        tokenizer_path=None,
        cache_path=tmp_path / "embeddings.db",
        limiter_dir=tmp_path / "locks",
    )
    embedder._embed_prepared_batch = AsyncMock(
        side_effect=lambda texts: [
            [1.0, 0.0] if "TAIL_MARKER" in text else [0.0, 1.0] for text in texts
        ]
    )
    collection = "filtered_knowledge_fragments"
    heavy_id, other_id = uuid4(), uuid4()
    opening = "Canonical definition opening. "
    heavy_body = opening + "ordinary prefix " * 1000 + "TAIL_MARKER filler " * 1000
    vocab = MagicMock()
    vocab.get_codebase_doc_count.return_value = 1
    vocab.vectorize_query.return_value = SparseVector(indices=[42], values=[1.0])
    engine = NoteSearchEngine(MagicMock(base_dir=tmp_path), storage, embedder, vocab)
    engine._collection_name = collection

    async def ready(engine):
        return collection

    monkeypatch.setattr(NoteSearchEngine, "_ready_collection", ready)
    try:
        await storage.create_collection(collection, dense_dim=2)
        other_type = ("glossary" if entity_type == "fact" else "fact") if mixed else entity_type
        for entity_id, body in [(heavy_id, heavy_body), (other_id, "TAIL_MARKER other entity")]:
            current_type = entity_type if entity_id == heavy_id else other_type
            payload, text = entity_source(current_type, entity_id, body)
            points = await fragment_point(
                embedder,
                point_id=generate_point_id(f"{current_type}:{entity_id}"),
                payload=payload,
                text=text,
                sparse=SparseVector(indices=[42], values=[1.0]),
                vectorize=lambda text: SparseVector(
                    indices=[42] if "TAIL_MARKER" in text else [],
                    values=[10.0] if "TAIL_MARKER" in text else [],
                ),
            )
            if entity_id == heavy_id:
                assert len(points) > 26
                assert all(
                    "object" not in p.payload and "context" not in p.payload
                    if entity_type == "fact"
                    else "expansion" not in p.payload and "definition" not in p.payload
                    for p in points[1:]
                )
            await upsert_fragment_group(storage, collection, points)

        if degraded:
            embedder.embed_single_cached = AsyncMock(
                side_effect=CircuitBreakerOpenError("isolated", retry_after=1.0)
            )
        client = await storage.get_client()
        client.retrieve = AsyncMock(wraps=client.retrieve)
        results = await engine.search(
            "TAIL_MARKER", mode="both", type_filter="all" if mixed else entity_type, limit=2
        )
        assert {result.note.id for result in results} == {heavy_id, other_id}
        assert len(results) == 2
        assert {result.note.id: result.result_type for result in results} == {
            heavy_id: entity_type,
            other_id: other_type,
        }
        assert all(result.degraded == degraded for result in results)
        heavy = next(result for result in results if result.note.id == heavy_id)
        assert any("TAIL_MARKER" in highlight for highlight in heavy.highlights)
        if entity_type == "fact":
            assert heavy.note.title == "[Fact] Alice references Complete canonical object"
            assert "Complete canonical object" in heavy.note.excerpt
            assert opening.strip() in heavy.note.excerpt
        else:
            assert "Complete canonical expansion" in heavy.note.excerpt
            assert opening.strip() in heavy.note.excerpt
        client.retrieve.assert_awaited_once()
        assert client.retrieve.await_args.kwargs["ids"] == [
            generate_point_id(f"{entity_type}:{heavy_id}")
        ]
    finally:
        await storage.close()
        await embedder.close()


@pytest.fixture
async def isolated_entity_engine(tmp_path, monkeypatch):
    storage = QdrantStorage(url="http://127.0.0.1:1", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    embedder = EmbeddingClient(
        model="isolated",
        dim=2,
        profile="raw",
        max_text_chars=128,
        max_input_bytes=0,
        max_input_tokens=None,
        tokenizer_path=None,
        cache_path=tmp_path / "embeddings.db",
        limiter_dir=tmp_path / "locks",
    )
    embedder._embed_prepared_batch = AsyncMock(
        side_effect=lambda texts: [[1.0, 0.0] for text in texts]
    )
    vocab = MagicMock()
    vocab.get_codebase_doc_count.return_value = 1
    vocab.vectorize_query.return_value = SparseVector(indices=[42], values=[1.0])
    engine = NoteSearchEngine(MagicMock(base_dir=tmp_path), storage, embedder, vocab)
    collection = "isolated_entity_fragments"

    async def ready(engine):
        return collection

    monkeypatch.setattr(NoteSearchEngine, "_ready_collection", ready)
    try:
        await storage.create_collection(collection, dense_dim=2)
        yield engine, storage, embedder, collection
    finally:
        await storage.close()
        await embedder.close()


async def test_mixed_markerless_legacy_entities_survive_sparse_fallback(isolated_entity_engine):
    engine, storage, embedder, collection = isolated_entity_engine
    client = await storage.get_client()
    expected = {}
    points = []
    for entity_type in ("fact", "glossary"):
        entity_id = uuid4()
        payload, text = entity_source(entity_type, entity_id, "TAIL_MARKER legacy body")
        assert "embedding_fragment" not in payload
        expected[entity_id] = entity_type
        points.append(
            PointStruct(
                id=generate_point_id(f"{entity_type}:{entity_id}"),
                payload={**payload, "embedding_text": text},
                vector={"dense": [1.0, 0.0], "sparse": {"indices": [42], "values": [1.0]}},
            )
        )
    await client.upsert(collection, points=points)
    embedder.embed_single_cached = AsyncMock(
        side_effect=CircuitBreakerOpenError("isolated", retry_after=1.0)
    )
    client.retrieve = AsyncMock(wraps=client.retrieve)
    results = await engine.search("TAIL_MARKER", mode="both", type_filter="all", limit=2)
    assert len(results) == 2
    assert {result.note.id: result.result_type for result in results} == expected
    assert all(result.degraded for result in results)
    assert all("Complete canonical" in result.note.excerpt for result in results)
    assert all(any("TAIL_MARKER" in text for text in result.highlights) for result in results)
    client.retrieve.assert_not_awaited()


@pytest.mark.parametrize("entity_type", ["fact", "glossary"])
async def test_entity_hydration_preserves_winning_fragment_evidence(
    isolated_entity_engine, entity_type
):
    engine, storage, embedder, collection = isolated_entity_engine
    entity_id = uuid4()
    payload, text = entity_source(entity_type, entity_id, "ordinary prefix " * 20 + "TAIL_MARKER")
    points = await fragment_point(
        embedder,
        point_id=generate_point_id(f"{entity_type}:{entity_id}"),
        payload=payload,
        text=text,
        sparse=SparseVector(indices=[], values=[]),
        vectorize=lambda text: SparseVector(indices=[], values=[]),
    )
    await upsert_fragment_group(storage, collection, points)
    child = points[-1]
    assert child.payload["embedding_fragment"]["index"] > 0
    winner = HybridResult(id=child.id, score=1.0, payload=child.payload)
    hydrated = (await engine._entity_display_payloads(collection, [winner]))[child.id]
    assert hydrated["embedding_fragment"] == child.payload["embedding_fragment"]
    assert hydrated["content"] == child.payload["content"]
    assert "TAIL_MARKER" in hydrated["content"]
    display_key = "object" if entity_type == "fact" else "expansion"
    assert hydrated[display_key] == payload[display_key]


@pytest.mark.parametrize("entity_type", ["fact", "glossary"])
@pytest.mark.parametrize("mismatch", ["source_hash", "type", "entity_id"])
async def test_mixed_search_rejects_stale_canonical_lineage(
    isolated_entity_engine, entity_type, mismatch
):
    engine, storage, embedder, collection = isolated_entity_engine
    entity_id = uuid4()
    payload, text = entity_source(entity_type, entity_id, "ordinary prefix " * 20 + "TAIL_MARKER")
    points = await fragment_point(
        embedder,
        point_id=generate_point_id(f"{entity_type}:{entity_id}"),
        payload=payload,
        text=text,
        sparse=SparseVector(indices=[], values=[]),
        vectorize=lambda text: SparseVector(
            indices=[42] if "TAIL_MARKER" in text else [],
            values=[1.0] if "TAIL_MARKER" in text else [],
        ),
    )
    assert len(points) > 1
    assert "TAIL_MARKER" in points[-1].payload["content"]
    await upsert_fragment_group(storage, collection, points)
    client = await storage.get_client()
    canonical = dict(points[0].payload)
    if mismatch == "source_hash":
        canonical["embedding_fragment"] = {
            **canonical["embedding_fragment"],
            "source_hash": "0" * 64,
        }
    elif mismatch == "type":
        canonical["type"] = "glossary" if entity_type == "fact" else "fact"
    else:
        canonical[f"{entity_type}_id"] = str(uuid4())
    await client.upsert(collection, points=[points[0].model_copy(update={"payload": canonical})])
    embedder.embed_single_cached = AsyncMock(
        side_effect=CircuitBreakerOpenError("isolated", retry_after=1.0)
    )
    with pytest.raises(ValueError, match="(?i)(lineage|canonical|source|identity)"):
        await engine.search("TAIL_MARKER", mode="both", type_filter="all", limit=1)
