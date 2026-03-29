"""Shared pytest fixtures for mcp-notes tests."""

import asyncio

import pytest


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
    from vector_core.storage.qdrant import generate_collection_name
    collection_name = generate_collection_name(
        str(notes_dir),
        prefix=settings.collection_prefix,
    )

    yield notes_dir

    # Cleanup: Delete the test collection from Qdrant
    # Note: This cleanup is best-effort - failures are logged but don't fail tests
    async def cleanup_collection():
        from vector_core.storage.qdrant import QdrantStorage
        storage = QdrantStorage()
        try:
            if await storage.collection_exists(collection_name):
                await storage.delete_collection(collection_name)
        except Exception:
            # Cleanup failure shouldn't fail the test - Qdrant may be unavailable
            pass
        finally:
            try:
                await storage.close()
            except Exception:
                pass

    # Use a separate thread to run cleanup to avoid event loop conflicts
    import threading

    def run_cleanup_sync():
        """Run cleanup in a new event loop on a separate thread."""
        def _run():
            asyncio.run(cleanup_collection())
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5.0)  # Wait up to 5 seconds for cleanup

    run_cleanup_sync()

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
