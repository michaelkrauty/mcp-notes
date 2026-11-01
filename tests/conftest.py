"""Shared pytest fixtures for mcp-notes tests."""

import atexit
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

# Override ambient data paths before settings imports. Service endpoints may
# be explicitly opted into, but tests and their subprocesses always own their
# caches, vocabulary, shared stores, and default note directory.
_TEST_DATA = TemporaryDirectory(prefix="mcp-notes-tests-")
TEST_DATA_DIR = Path(_TEST_DATA.name)
atexit.register(_TEST_DATA.cleanup)
for variable, directory in (
    ("VECTOR_CACHE_DIR", "cache"),
    ("VECTOR_SHARED_DATA_DIR", "shared"),
    ("NOTES_DIR", "notes"),
):
    os.environ[variable] = str(TEST_DATA_DIR / directory)
os.environ.setdefault("VECTOR_EMBEDDING_URL", "http://127.0.0.1:1")
os.environ.setdefault("VECTOR_QDRANT_URL", "http://127.0.0.1:1")
# Every test collection is named after its own temporary directory. A shared
# collection name from the environment would make tests write into, and their
# teardown delete, a collection that is not theirs.
os.environ.pop("VECTOR_COLLECTION_NAME", None)

# Settings read the environment once, at import, and an unset embedding
# dimension leaves collection creation raising "embedding_dim not yet
# initialized" for any test that stores a vector. The suite therefore needs a
# dimension established before anything imports the settings object, which is
# why this runs at the top of the root conftest.
#
# It cannot simply be a constant. Leaving the dimension unset is what lets the
# library auto-detect it from the first embedding call, so hardcoding one would
# silently pin the suite to a width the configured service may not return, and
# every collection built from it would be incompatible. So: ask the service.
# One request settles both questions this suite needs answered — whether the
# service is usable at all, and how wide its vectors are.

_FALLBACK_EMBEDDING_DIM = 128


def _probe_embedding_service() -> int | None:
    """Return the configured service's embedding width, or None if unusable.

    Deliberately a real embedding request rather than a liveness ping: a
    service can serve /v1/models while rejecting the configured model, and a
    ping cannot report a dimension. Any failure means the full-stack tests
    could not have run anyway.
    """
    url = os.environ.get("VECTOR_EMBEDDING_URL", "http://localhost:8080")
    model = os.environ.get("VECTOR_EMBEDDING_MODEL", "")
    try:
        import httpx

        response = httpx.post(
            f"{url.rstrip('/')}/v1/embeddings",
            json={"model": model, "input": "probe"},
            timeout=10.0,
        )
        if response.status_code != 200:
            return None
        return len(response.json()["data"][0]["embedding"]) or None
    except Exception:
        return None


EMBEDDING_DIM_FROM_SERVICE = (
    None if os.environ.get("VECTOR_EMBEDDING_DIM") else _probe_embedding_service()
)

# An explicitly exported dimension always wins: a developer pointing the suite
# at a specific service has already said what width to expect.
os.environ.setdefault(
    "VECTOR_EMBEDDING_DIM",
    str(EMBEDDING_DIM_FROM_SERVICE or _FALLBACK_EMBEDDING_DIM),
)

import asyncio  # noqa: E402 - must follow the environment default above
from contextlib import asynccontextmanager  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402 - must follow the environment default above


def qdrant_and_embeddings_available() -> bool:
    """Whether the full stack can actually serve these tests.

    The embedding half is checked by embedding something, not by pinging
    /v1/models: a service can list models while rejecting the configured one,
    and only a real response reveals the vector width. That width has to match
    what settings expect, because a mismatch fails every upsert for reasons
    that say nothing about the code under test.
    """
    import httpx

    from mcp_notes.settings import settings

    try:
        if httpx.get(f"{settings.qdrant_url}/collections", timeout=2.0).status_code != 200:
            return False

        response = httpx.post(
            f"{settings.embedding_url.rstrip('/')}/v1/embeddings",
            json={"model": settings.embedding_model, "input": "probe"},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False
        return len(response.json()["data"][0]["embedding"]) == settings.embedding_dim
    except Exception:
        return False


# Tests that index or search for real need both services. Without them the
# work fails at the first embedding call, which says nothing about the code
# under test, so they are skipped rather than failed.
requires_full_stack = pytest.mark.skipif(
    not qdrant_and_embeddings_available(),
    reason="Qdrant and/or embedding service not available",
)


# Logical collection names tests have claimed. Only these can be deleted by
# delete_test_collection; the pending set holds those not yet cleaned up, for
# the end-of-session sweep.
_OWNED_TEST_COLLECTIONS: set[str] = set()
_PENDING_TEST_COLLECTIONS: set[str] = set()


def _claim(logical_name: str) -> str:
    from vector_core.settings import settings as vector_settings  # noqa: PLC0415

    if vector_settings.collection_name and logical_name == vector_settings.collection_name:
        raise RuntimeError(
            f"Refusing to treat the configured collection {logical_name!r} as a test collection"
        )
    _OWNED_TEST_COLLECTIONS.add(logical_name)
    _PENDING_TEST_COLLECTIONS.add(logical_name)
    return logical_name


def claim_notes_test_collection(notes_dir: Path) -> str:
    """Claim the collection named after a temporary notes directory.

    Refuses a directory outside the system temporary directory, and a name
    that is really a configured shared collection, so a test can never adopt
    a collection it did not create.
    """
    from vector_core.storage.qdrant import generate_collection_name  # noqa: PLC0415

    from mcp_notes.settings import settings  # noqa: PLC0415

    if not Path(notes_dir).resolve().is_relative_to(Path(gettempdir()).resolve()):
        raise RuntimeError(f"Refusing a test collection for non-temporary {notes_dir}")
    return _claim(generate_collection_name(str(notes_dir), prefix=settings.collection_prefix))


def claim_test_collection(logical_name: str) -> str:
    """Claim a test-only logical collection name (it must start with ``test_``)."""
    if not logical_name.startswith("test_"):
        raise RuntimeError(f"Refusing to claim non-test collection name {logical_name!r}")
    return _claim(logical_name)


async def delete_test_collection(logical_name: str) -> None:
    """Delete everything stored for a claimed test logical collection name.

    A logical name is an alias (``<name>__active``) for the physical
    ``vcgen_*`` collection of its current embedding generation, so deleting a
    collection called ``logical_name`` alone removes nothing. This removes the
    alias, every generation whose manifest names ``logical_name``, and a legacy
    collection that still uses the name directly. Only names a test claimed
    are touched, and a name stays pending for the session sweep until its
    cleanup succeeds. Best-effort: Qdrant may be unavailable.
    """
    if logical_name not in _OWNED_TEST_COLLECTIONS:
        raise RuntimeError(f"Refusing to delete unclaimed collection {logical_name!r}")

    from qdrant_client.models import DeleteAlias, DeleteAliasOperation  # noqa: PLC0415
    from vector_core.storage.qdrant import QdrantStorage  # noqa: PLC0415

    storage = QdrantStorage()
    try:
        client = await storage.get_client()
        alias = f"{logical_name}__active"
        if any(found.alias_name == alias for found in (await client.get_aliases()).aliases):
            await client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))
                ]
            )
        for collection in (await client.get_collections()).collections:
            name = collection.name
            if name != logical_name:
                if not name.startswith("vcgen_"):
                    continue
                metadata = await storage.get_metadata(name) or {}
                if metadata.get("embedding_generation", {}).get("logical_name") != logical_name:
                    continue
            await client.delete_collection(name)
    except Exception:
        return  # Cleanup failure shouldn't fail the test; the sweep retries it
    finally:
        try:
            await storage.close()
        except Exception:
            pass
    _PENDING_TEST_COLLECTIONS.discard(logical_name)


def delete_test_collection_sync(logical_name: str) -> None:
    """Run ``delete_test_collection`` from synchronous fixture teardown.

    Runs on its own thread and event loop so it neither depends on nor
    disturbs a loop the test may have left behind.
    """
    import threading  # noqa: PLC0415

    thread = threading.Thread(
        target=lambda: asyncio.run(delete_test_collection(logical_name)), daemon=True
    )
    thread.start()
    thread.join(timeout=30.0)


@pytest.fixture
def mock_collection_mutation(monkeypatch):
    """Stub the writer boundary explicitly for mocked tool tests only."""
    from mcp_notes.tools import mutation  # noqa: PLC0415

    @asynccontextmanager
    async def collection_operation():
        yield

    indexer = SimpleNamespace(collection_operation=collection_operation)
    getter = AsyncMock(return_value=indexer)
    monkeypatch.setattr(mutation, "get_indexer", getter)
    return getter


@pytest.fixture
def tmp_notes_dir(tmp_path, monkeypatch):
    """Create temporary notes directory and configure settings."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    # Patch settings to use temp directory
    monkeypatch.setenv("NOTES_DIR", str(notes_dir))

    # Reset GlobalVocabulary singleton from vector-core (must be done BEFORE mcp-notes singletons)
    # Also unregister "notes" codebase to clear stale data from shared DB
    from vector_core.embeddings.global_vocab import GlobalVocabulary

    GlobalVocabulary.reset_instance()
    # Get fresh instance and clear any stale "notes" registration from shared DB
    vocab = GlobalVocabulary.get_instance()
    try:
        vocab.unregister_codebase("notes")
    except Exception:
        pass  # May not exist, that's fine

    # Reset global instances in singletons module
    import mcp_notes.singletons as singletons_module

    # Save sync singleton instances (get current instance if any)
    original_store = singletons_module._note_store.get_if_initialized()
    original_git = singletons_module._git_manager.get_if_initialized()
    original_links = singletons_module._link_resolver.get_if_initialized()
    original_glossary_store = singletons_module._glossary_store.get_if_initialized()
    original_fact_store = singletons_module._fact_store.get_if_initialized()
    original_integrity = singletons_module._integrity_manager.get_if_initialized()

    # Also reset settings
    from mcp_notes.settings import settings

    original_notes_dir = settings.dir
    settings.dir = notes_dir

    # Create temp glossary database (to isolate from shared DB)
    from vector_core.glossary import GlossaryStore

    glossary_db_path = tmp_path / "glossary.db"
    temp_glossary_store = GlossaryStore(db_path=glossary_db_path)

    # Create temp fact database (to isolate from shared DB)
    from vector_core.facts import FactStore

    fact_db_path = tmp_path / "facts.db"
    temp_fact_store = FactStore(db_path=fact_db_path)

    # Reset sync singletons and inject temp instances
    singletons_module._note_store.reset()
    singletons_module._git_manager.reset()
    singletons_module._link_resolver.reset()
    singletons_module._glossary_store.set_instance(temp_glossary_store)
    singletons_module._fact_store.set_instance(temp_fact_store)
    singletons_module._integrity_manager.reset()

    # Reset async singletons using AsyncSingleton.reset()
    singletons_module._indexer.reset()
    singletons_module._search_engine.reset()
    singletons_module._glossary_indexer.reset()
    singletons_module._fact_indexer.reset()
    singletons_module._note_service.reset()  # CRITICAL: Must reset to use new store

    # Track collection name for cleanup
    collection_name = claim_notes_test_collection(notes_dir)

    yield notes_dir

    # Cleanup: delete the test collection, its alias and generations from Qdrant
    delete_test_collection_sync(collection_name)

    # Restore sync singleton state using set_instance
    singletons_module._note_store.set_instance(original_store)
    singletons_module._git_manager.set_instance(original_git)
    singletons_module._link_resolver.set_instance(original_links)
    singletons_module._glossary_store.set_instance(original_glossary_store)
    singletons_module._fact_store.set_instance(original_fact_store)
    singletons_module._integrity_manager.set_instance(original_integrity)

    # Reset async singletons (they will reinitialize on next use)
    singletons_module._indexer.reset()
    singletons_module._search_engine.reset()
    singletons_module._glossary_indexer.reset()
    singletons_module._fact_indexer.reset()
    singletons_module._note_service.reset()

    # Reset GlobalVocabulary to clear test data
    GlobalVocabulary.reset_instance()

    settings.dir = original_notes_dir


@pytest.fixture
def sample_note_content():
    """Sample note content for testing."""
    return """# Test Note

This is a sample note for testing.

## Section 1

Some content in section 1.

## Section 2

More content here with [[wikilinks]] and #tags.
"""


@pytest.fixture
def sample_frontmatter_note():
    """Sample note with frontmatter."""
    return """---
title: Test Note with Frontmatter
tags:
  - python
  - testing
category: work/projects
---

# Main Content

This note has YAML frontmatter.
"""


def pytest_sessionfinish(session, exitstatus):
    """Clean up orphaned collections at end of test session."""
    from vector_core.storage.qdrant import QdrantStorage

    async def cleanup():
        # Collections whose teardown did not run or did not succeed
        for logical_name in sorted(_PENDING_TEST_COLLECTIONS):
            await delete_test_collection(logical_name)
        storage = QdrantStorage()
        try:
            collections = await storage.list_collections(prefix="notes_")
            for col in collections:
                # Clean up collections for temp paths
                try:
                    metadata = await storage.get_metadata(col)
                    if metadata and metadata.get("codebase_path", "").startswith("/tmp/pytest"):
                        await storage.delete_collection(col)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            await storage.close()

    try:
        asyncio.run(cleanup())
    except Exception:
        pass
    finally:
        _TEST_DATA.cleanup()
