"""Integration test fixtures for mcp-notes."""

import asyncio
from uuid import uuid4

import pytest


@pytest.fixture
def integration_notes_dir(tmp_path, monkeypatch):
    """Create temporary notes directory for integration tests."""
    notes_dir = tmp_path / f"notes_{uuid4().hex[:8]}"
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

    # Save sync singleton instances using SyncSingleton API
    original_store = singletons_module._note_store.get_if_initialized()
    original_git = singletons_module._git_manager.get_if_initialized()
    original_links = singletons_module._link_resolver.get_if_initialized()

    # Reset sync singletons using SyncSingleton API
    singletons_module._note_store.reset()
    singletons_module._git_manager.reset()
    singletons_module._link_resolver.reset()

    # Reset async singletons using AsyncSingleton.reset()
    singletons_module._indexer.reset()
    singletons_module._search_engine.reset()
    singletons_module._glossary_indexer.reset()
    singletons_module._fact_indexer.reset()
    singletons_module._note_service.reset()  # CRITICAL: Must reset to use new store

    # Also reset settings
    from mcp_notes.settings import settings
    original_notes_dir = settings.dir
    settings.dir = notes_dir

    # Track collection name for cleanup
    from vector_core import generate_collection_name
    collection_name = generate_collection_name(
        str(notes_dir),
        prefix=settings.collection_prefix,
    )

    yield notes_dir

    # Cleanup: Delete Qdrant collection
    async def cleanup_collection():
        from vector_core import QdrantStorage
        storage = QdrantStorage()
        try:
            if await storage.collection_exists(collection_name):
                await storage.delete_collection(collection_name)
        except Exception:
            pass
        finally:
            try:
                await storage.close()
            except Exception:
                pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(cleanup_collection())
        else:
            loop.run_until_complete(cleanup_collection())
    except (RuntimeError, Exception):
        try:
            asyncio.run(cleanup_collection())
        except Exception:
            pass

    # Restore sync singleton state using SyncSingleton API
    singletons_module._note_store.set_instance(original_store)
    singletons_module._git_manager.set_instance(original_git)
    singletons_module._link_resolver.set_instance(original_links)

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
async def indexed_notes(integration_notes_dir):
    """Create and index several notes for search testing."""
    from mcp_notes.server import create_note, get_indexer

    # Create notes with various content
    notes = []

    note1 = await create_note(
        title="Python Programming Guide",
        content="Python is a versatile programming language. It supports object-oriented, procedural, and functional paradigms.",
        tags=["python", "programming"],
        category="tutorials/programming",
    )
    notes.append(note1)

    note2 = await create_note(
        title="JavaScript Basics",
        content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js.",
        tags=["javascript", "web"],
        category="tutorials/web",
    )
    notes.append(note2)

    note3 = await create_note(
        title="Database Design Patterns",
        content="Database normalization is crucial for data integrity. Learn about 1NF, 2NF, 3NF and BCNF.",
        tags=["database", "patterns"],
        category="tutorials/database",
    )
    notes.append(note3)

    note4 = await create_note(
        title="Python Web Frameworks",
        content="Flask and Django are popular Python web frameworks. FastAPI is gaining popularity for APIs.",
        tags=["python", "web", "frameworks"],
        category="tutorials/programming",
    )
    notes.append(note4)

    note5 = await create_note(
        title="Meeting Notes: Q4 Planning",
        content="Discussed roadmap for Q4. Key priorities: improve search, add collaboration features.",
        tags=["meetings", "planning"],
        category="work/meetings",
    )
    notes.append(note5)

    # Force index all notes
    indexer = await get_indexer()
    await indexer.index_all(force=True)

    # Verify GlobalVocabulary is properly registered
    # (ensures tests won't fail due to vocabulary issues)
    from vector_core.embeddings.global_vocab import GlobalVocabulary
    vocab = GlobalVocabulary.get_instance()
    from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID
    doc_count = vocab.get_codebase_doc_count(NOTES_CODEBASE_ID)
    if doc_count == 0:
        raise RuntimeError(
            f"GlobalVocabulary not registered for '{NOTES_CODEBASE_ID}' after indexing. "
            f"Expected > 0 docs, got {doc_count}"
        )

    return notes
