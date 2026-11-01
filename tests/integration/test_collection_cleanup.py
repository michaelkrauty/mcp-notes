"""Test teardown must not leave collections behind in a real Qdrant."""

from uuid import uuid4

from ..conftest import delete_test_collection, requires_full_stack


@requires_full_stack
async def test_delete_test_collection_removes_alias_and_generation(tmp_path):
    from vector_core import EmbeddingClient, QdrantStorage  # noqa: PLC0415
    from vector_core.embeddings.global_vocab import GlobalVocabulary  # noqa: PLC0415
    from vector_core.storage.embedding_migration import (  # noqa: PLC0415
        active_embedding_collection,
    )

    from mcp_notes.indexing.indexer import NoteIndexer  # noqa: PLC0415
    from mcp_notes.storage.filesystem import NoteStore  # noqa: PLC0415

    store = NoteStore(tmp_path / f"notes_{uuid4().hex[:8]}")
    note = store.create("Cleanup probe", "A note that only exists for this test.")
    storage = QdrantStorage()
    embedder = EmbeddingClient()
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    indexer = NoteIndexer(store, storage, embedder, vocab)
    try:
        await indexer.index_note(note.id)
        logical = indexer.logical_collection_name
        physical = await active_embedding_collection(storage, logical)
        assert physical is not None and physical != logical

        await delete_test_collection(logical)

        assert await active_embedding_collection(storage, logical) is None
        assert not await storage.collection_exists(physical)
        assert not await storage.collection_exists(logical)
    finally:
        await delete_test_collection(indexer.logical_collection_name)
        await storage.close()
        await embedder.close()
        vocab.close()
