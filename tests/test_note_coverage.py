"""Passage-backed note retrieval, isolated from persistent storage and services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models
from vector_core import EmbeddingClient
from vector_core.embeddings.client import CircuitBreakerOpenError
from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.hybrid import HybridSearcher

from mcp_notes.search.engine import NoteSearchEngine


@pytest.fixture
def ready_engine(monkeypatch):
    async def ready(engine):
        return engine.collection_name

    monkeypatch.setattr(NoteSearchEngine, "_ready_collection", ready)


@pytest.mark.parametrize(
    "options", [{"mode": "note"}, {"type_filter": "note"}, {"mode": "note", "type_filter": "all"}]
)
async def test_note_search_groups_each_modality_and_rolls_up_passage(options, ready_engine):
    note_id = str(uuid4())
    payload = {
        "note_id": note_id,
        "type": "chunk",
        "title": "Tail match",
        "content": "buried tail marker",
        "tags": [],
    }
    hit = SimpleNamespace(id=str(uuid4()), score=0.95, payload=payload)
    client = AsyncMock()
    sparse_hit = SimpleNamespace(
        id=str(uuid4()),
        score=4.5,
        payload={**payload, "content": "different sparse passage"},
    )
    client.query_points_groups.side_effect = [
        SimpleNamespace(groups=[SimpleNamespace(id=note_id, hits=[hit])]),
        SimpleNamespace(groups=[SimpleNamespace(id=note_id, hits=[sparse_hit])]),
    ]
    storage = MagicMock()
    storage.get_client = AsyncMock(return_value=client)
    storage._get_client = AsyncMock(return_value=client)
    embedder = MagicMock()
    embedder.embed_single_cached = AsyncMock(return_value=[1.0, 0.0])
    vocab = MagicMock()
    vocab.get_codebase_doc_count.return_value = 1
    vocab.vectorize_query.return_value = SparseVector(indices=[1], values=[1.0])
    store = MagicMock(base_dir="/isolated/notes")
    engine = NoteSearchEngine(store, storage, embedder, vocab)

    results = await engine.search("tail marker", **options)

    assert len(results) == 1
    assert results[0].note.id == UUID(note_id)
    assert results[0].result_type == "note"
    assert results[0].note.excerpt == "buried tail marker"
    assert results[0].highlights
    assert payload["type"] == "chunk"
    assert client.query_points_groups.await_count == 2
    assert {call.kwargs["using"] for call in client.query_points_groups.await_args_list} == {
        "dense",
        "sparse",
    }
    for call in client.query_points_groups.await_args_list:
        assert call.kwargs["group_by"] == "note_id"
        assert call.kwargs["group_size"] == 1
        types = [
            condition for condition in call.kwargs["query_filter"].must if condition.key == "type"
        ]
        assert len(types) == 1
        assert set(types[0].match.any) == {"note", "chunk"}
    client.query_points.assert_not_awaited()


async def test_note_search_sparse_fallback_still_groups_passages(ready_engine):
    note_id = str(uuid4())
    hit = SimpleNamespace(
        id=str(uuid4()),
        score=0.8,
        payload={"note_id": note_id, "type": "chunk", "content": "tail marker", "title": "Tail"},
    )
    client = AsyncMock()
    client.query_points_groups.return_value = SimpleNamespace(groups=[SimpleNamespace(hits=[hit])])
    storage = MagicMock()
    storage.get_client = AsyncMock(return_value=client)
    embedder = MagicMock()
    embedder.embed_single_cached = AsyncMock(
        side_effect=CircuitBreakerOpenError("http://127.0.0.1:1", retry_after=1.0)
    )
    vocab = MagicMock()
    vocab.get_codebase_doc_count.return_value = 1
    vocab.vectorize_query.return_value = SparseVector(indices=[1], values=[1.0])
    engine = NoteSearchEngine(MagicMock(base_dir="/isolated/notes"), storage, embedder, vocab)

    results = await engine.search("tail marker", mode="note")

    assert [(result.result_type, result.degraded) for result in results] == [("note", True)]
    client.query_points_groups.assert_awaited_once()
    assert client.query_points_groups.call_args.kwargs["using"] == "sparse"
    assert client.query_points_groups.call_args.kwargs["group_by"] == "note_id"


class TailEmbeddingClient(EmbeddingClient):
    """Use the real source splitter and deterministic orthogonal dense vectors."""

    async def embed_single_cached(self, text, *, role="document"):
        # The marker is beyond the old summary limit, and only its passage has
        # the semantic direction used by the query and the similar note.
        return [1.0, 0.0] if "TAIL_MATCH_SENTINEL" in text else [0.0, 1.0]


async def test_dense_only_long_tail_retrieval_and_similarity(tmp_path, monkeypatch, ready_engine):
    client = AsyncQdrantClient(location=":memory:")
    embedder = TailEmbeddingClient(
        base_url="http://127.0.0.1:1",
        dim=2,
        profile="raw",
        max_text_chars=80,
        max_input_bytes=0,
        max_input_tokens=None,
        tokenizer_path=None,
        cache_path=tmp_path / "embeddings.db",
        limiter_dir=tmp_path / "locks",
    )
    collection = "notes_tail_regression"
    source_id, match_id, distractor_id = [str(uuid4()) for _ in range(3)]
    storage = MagicMock()
    storage.get_client = AsyncMock(return_value=client)
    storage._get_client = AsyncMock(return_value=client)
    vocab = MagicMock()
    vocab.get_codebase_doc_count.return_value = 1
    vocab.vectorize_query.return_value = SparseVector(indices=[], values=[])
    engine = NoteSearchEngine(MagicMock(base_dir=tmp_path), storage, embedder, vocab)
    engine._collection_name = collection

    def dense_only(storage, **weights):
        return HybridSearcher(storage, dense_weight=1.0, sparse_weight=0.0)

    monkeypatch.setattr("mcp_notes.search.engine.HybridSearcher", dense_only)
    try:
        await client.create_collection(
            collection,
            vectors_config={
                "dense": models.VectorParams(size=2, distance=models.Distance.COSINE),
            },
        )
        context = "# Note\n\n"
        # More than 128 source chunks exercises real Qdrant scroll pagination.
        body = "ordinary prefix " * 900 + "\nTAIL_MATCH_SENTINEL"
        spans = embedder.split_text(body, role="document", context_prefix=context)
        assert len(spans) > 128
        assert "".join(span.text for span in spans) == body
        points = []
        for note_id, text, title in [
            (source_id, body, "Long source"),
            (match_id, "TAIL_MATCH_SENTINEL", "Tail peer"),
            (distractor_id, "ordinary prefix", "Distractor"),
        ]:
            # Deliberately uninformative summary vectors prove that passages
            # determine both retrieval and note-to-note similarity.
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector={"dense": [0.0, 1.0]},
                    payload={
                        "note_id": note_id,
                        "type": "note",
                        "title": title,
                        "content": "summary",
                    },
                )
            )
            for index, span in enumerate(
                embedder.split_text(
                    text,
                    role="document",
                    context_prefix=context,
                )
            ):
                passage = context + span.text
                points.append(
                    models.PointStruct(
                        id=str(uuid4()),
                        vector={"dense": await embedder.embed_single_cached(passage)},
                        payload={
                            "note_id": note_id,
                            "type": "chunk",
                            "chunk_index": index,
                            "embedding_fragment": index > 0,
                            "title": title,
                            "content": passage,
                            "start_char": span.start,
                            "end_char": span.end,
                        },
                    )
                )
        await client.upsert(collection, points=points)

        results = await engine.search("TAIL_MATCH_SENTINEL", mode="note", limit=2)
        assert {str(result.note.id) for result in results} == {source_id, match_id}
        assert all(result.result_type == "note" for result in results)
        assert all("TAIL_MATCH_SENTINEL" in result.note.excerpt for result in results)

        similar = await engine.find_similar(UUID(source_id), limit=2)
        assert {str(result.note.id) for result in similar} == {match_id, distractor_id}
        peer = next(result for result in similar if str(result.note.id) == match_id)
        assert peer.score == pytest.approx(1.0)
        assert "TAIL_MATCH_SENTINEL" in peer.note.excerpt
        assert all(result.result_type == "note" for result in similar)
    finally:
        await client.close()
        await embedder.close()
