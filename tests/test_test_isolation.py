"""The test bootstrap isolates persistent data before application imports."""

import json
import os
import subprocess
import sys
from pathlib import Path


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
