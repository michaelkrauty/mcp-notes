"""Test teardown must clean up after itself in a real Qdrant, and only after itself."""

from pathlib import Path
from uuid import uuid4

import pytest

from .. import conftest
from ..conftest import (
    claim_notes_test_collection,
    claim_test_collection,
    delete_test_collection,
    requires_full_stack,
)


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
    logical = claim_notes_test_collection(store.base_dir)
    storage = QdrantStorage()
    embedder = EmbeddingClient()
    vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
    indexer = NoteIndexer(store, storage, embedder, vocab)
    try:
        # The indexer must write to the claimed, test-owned collection.
        assert indexer.logical_collection_name == logical
        note = store.create("Cleanup probe", "A note that only exists for this test.")
        await indexer.index_note(note.id)
        physical = await active_embedding_collection(storage, logical)
        assert physical is not None and physical != logical

        await delete_test_collection(logical)

        assert await active_embedding_collection(storage, logical) is None
        assert not await storage.collection_exists(physical)
        assert not await storage.collection_exists(logical)
        assert logical not in conftest._PENDING_TEST_COLLECTIONS
    finally:
        await delete_test_collection(logical)
        await storage.close()
        await embedder.close()
        vocab.close()


async def test_unclaimed_collection_is_never_deleted():
    with pytest.raises(RuntimeError, match="unclaimed"):
        await delete_test_collection("unified_knowledge")


def test_configured_shared_collection_cannot_be_claimed(tmp_path, monkeypatch):
    from vector_core.settings import settings as vector_settings  # noqa: PLC0415

    monkeypatch.setattr(vector_settings, "collection_name", "shared_production")
    with pytest.raises(RuntimeError, match="configured collection"):
        claim_notes_test_collection(tmp_path / "notes")
    assert "shared_production" not in conftest._OWNED_TEST_COLLECTIONS


def test_only_temporary_directories_and_test_names_can_be_claimed(tmp_path):
    with pytest.raises(RuntimeError, match="non-temporary"):
        claim_notes_test_collection(Path.home() / "notes")
    with pytest.raises(RuntimeError, match="non-test"):
        claim_test_collection("notes_e86a9495bc7a")


async def test_failed_cleanup_stays_pending_for_the_session_sweep(monkeypatch):
    from vector_core.storage.qdrant import QdrantStorage  # noqa: PLC0415

    async def unavailable(self):
        raise ConnectionError("Qdrant is briefly unavailable")

    monkeypatch.setattr(QdrantStorage, "get_client", unavailable)
    logical = claim_test_collection(f"test_cleanup_probe_{uuid4().hex[:12]}")
    await delete_test_collection(logical)
    assert logical in conftest._PENDING_TEST_COLLECTIONS
    conftest._PENDING_TEST_COLLECTIONS.discard(logical)
