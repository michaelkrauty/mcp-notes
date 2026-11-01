"""The test bootstrap isolates persistent data before application imports, and
offline_collection isolates the tool tests that opt in from the vector services."""

import json
import os
import subprocess
import sys
from pathlib import Path

from vector_core import EmbeddingClient, QdrantStorage
from vector_core.settings import settings as vector_settings

from mcp_notes.server import create_note, get_indexer, get_search, search_notes


def test_bootstrap_replaces_ambient_paths_and_cleans_actual_cache_writes(tmp_path):
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    sentinel = ambient / "sentinel"
    sentinel.write_text("untouched")
    environment = {
        **os.environ,
        "VECTOR_CACHE_DIR": str(ambient),
        "VECTOR_SHARED_DATA_DIR": str(ambient),
        "NOTES_DIR": str(ambient),
        "VECTOR_EMBEDDING_DIM": "128",
        "VECTOR_EMBEDDING_URL": "http://127.0.0.1:1",
        "VECTOR_QDRANT_URL": "http://127.0.0.1:1",
    }
    script = """
import asyncio
import json
import os
from unittest.mock import AsyncMock
from tests.conftest import TEST_DATA_DIR
from vector_core import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.settings import settings
from mcp_notes.settings import settings as notes_settings

async def write_cache():
    client = EmbeddingClient(
        model="isolated-cache-proof", dim=128, profile="raw", cache_namespace="proof"
    )
    client.resolve_identity = AsyncMock(return_value=client.configured_identity())
    client._embed_prepared_batch = AsyncMock(return_value=[[1.0] + [0.0] * 127])
    await client.embed_all(["synthetic input"])
    assert client._cache_path.is_file()
    await client.close()

asyncio.run(write_cache())
vocab = GlobalVocabulary.get_instance()
vocab.register_codebase("notes", [{"synthetic"}])
vocab.unregister_codebase("notes")
vocab.close()
assert (settings.cache_dir / "global_vocabulary.db").is_file()
assert settings.cache_dir == TEST_DATA_DIR / "cache"
assert settings.shared_data_dir == TEST_DATA_DIR / "shared"
assert notes_settings.dir == TEST_DATA_DIR / "notes"
assert all(os.environ[key].startswith(str(TEST_DATA_DIR)) for key in (
    "VECTOR_CACHE_DIR", "VECTOR_SHARED_DATA_DIR", "NOTES_DIR"
))
print(json.dumps({"root": str(TEST_DATA_DIR), "cache_written": True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["cache_written"] is True
    assert not Path(proof["root"]).exists()
    assert list(ambient.iterdir()) == [sentinel]
    assert sentinel.read_text() == "untouched"


async def test_offline_collection_indexes_and_searches_without_services(
    tmp_notes_dir, offline_collection
):
    """An opted-in tool call really indexes, into the in-memory database."""
    created = await create_note(title="Offline", content="Indexed without either service.")

    stored = []
    for collection in (await offline_collection.get_collections()).collections:
        points, _ = await offline_collection.scroll(collection.name, limit=100)
        stored.extend(point.payload for point in points)
    assert {p["type"] for p in stored if p.get("note_id") == created["id"]} == {"note", "chunk"}
    assert (await get_indexer()).embedder._embed_prepared_batch.await_count > 0
    # The search engine reads what the indexer wrote through its own handle,
    # on the dense path rather than the sparse-only fallback.
    results = await search_notes(query="indexed")
    assert {r["note"]["id"] for r in results} == {created["id"]}
    assert not any(r["degraded"] for r in results)


async def test_tools_keep_configured_services_without_offline_opt_in(tmp_notes_dir):
    """Without the opt-in the singletons build clients for the configured
    services, so a requires_full_stack test exercises the real path."""
    for component in (await get_indexer(), await get_search()):
        assert type(component.storage) is QdrantStorage
        assert component.storage.url == vector_settings.qdrant_url
        assert type(component.embedder) is EmbeddingClient
        assert component.embedder.base_url == vector_settings.embedding_url.rstrip("/")
