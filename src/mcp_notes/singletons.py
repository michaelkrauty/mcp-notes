"""Singleton instances and getters for mcp-notes.

Provides thread-safe lazy initialization for all shared resources.
"""

import atexit
from typing import Any

from vector_core import (
    AsyncSingleton,
    SyncSingleton,
    sync_cleanup_wrapper,
)
from vector_core.glossary import (
    GlossaryIndexer,
    GlossaryStore,
)

from mcp_notes.facts import (
    FactIndexer,
    FactStore,
    SourceIntegrityManager,
)
from mcp_notes.indexing.indexer import NoteIndexer
from mcp_notes.links.resolver import LinkResolver
from mcp_notes.search.engine import NoteSearchEngine
from mcp_notes.services import NoteService
from mcp_notes.storage.filesystem import NoteStore
from mcp_notes.storage.git import GitManager

# ============= Sync Singletons =============
# Provides thread-safe initialization with reentrant lock

_note_store: SyncSingleton[NoteStore] = SyncSingleton("note_store")
_git_manager: SyncSingleton[GitManager] = SyncSingleton("git_manager")
_link_resolver: SyncSingleton[LinkResolver] = SyncSingleton("link_resolver")
_glossary_store: SyncSingleton[GlossaryStore] = SyncSingleton("glossary_store")
_fact_store: SyncSingleton[FactStore] = SyncSingleton("fact_store")
_integrity_manager: SyncSingleton[SourceIntegrityManager] = SyncSingleton("integrity_manager")


# ============= Async Singletons =============
# Provides async-safe initialization with proper cleanup handling

_indexer: AsyncSingleton[NoteIndexer] = AsyncSingleton("indexer")
_search_engine: AsyncSingleton[NoteSearchEngine] = AsyncSingleton("search_engine")
_glossary_indexer: AsyncSingleton[GlossaryIndexer] = AsyncSingleton("glossary_indexer")
_fact_indexer: AsyncSingleton[FactIndexer] = AsyncSingleton("fact_indexer")
_note_service: AsyncSingleton[NoteService] = AsyncSingleton("note_service")


# ============= Sync Getters =============


def get_store() -> NoteStore:
    """Get or create NoteStore instance (thread-safe via SyncSingleton)."""

    def _create_store() -> NoteStore:
        store = NoteStore()
        store.ensure_directories()
        return store

    return _note_store.get(_create_store)


def get_git() -> GitManager:
    """Get or create GitManager instance (thread-safe via SyncSingleton)."""
    return _git_manager.get(GitManager)


def get_links() -> LinkResolver:
    """Get or create LinkResolver instance (thread-safe via SyncSingleton)."""
    return _link_resolver.get(lambda: LinkResolver(note_store=get_store()))


def get_glossary_store() -> GlossaryStore:
    """Get or create GlossaryStore instance (thread-safe via SyncSingleton)."""
    return _glossary_store.get(GlossaryStore)


def get_fact_store() -> FactStore:
    """Get or create FactStore instance (thread-safe via SyncSingleton)."""
    return _fact_store.get(FactStore)


def get_integrity_manager() -> SourceIntegrityManager:
    """Get or create SourceIntegrityManager instance (thread-safe via SyncSingleton)."""
    return _integrity_manager.get(lambda: SourceIntegrityManager(fact_store=get_fact_store()))


# ============= Async Getters =============


async def get_indexer() -> NoteIndexer:
    """Get or create NoteIndexer instance (async-safe via AsyncSingleton)."""

    async def _create_indexer() -> NoteIndexer:
        indexer = NoteIndexer(note_store=get_store())
        await indexer._ensure_global_vocab()
        return indexer

    return await _indexer.get(_create_indexer)


async def get_search() -> NoteSearchEngine:
    """Get or create NoteSearchEngine instance (async-safe via AsyncSingleton)."""

    async def _create_search_engine() -> NoteSearchEngine:
        engine = NoteSearchEngine(note_store=get_store())
        await engine._ensure_global_vocab()
        return engine

    return await _search_engine.get(_create_search_engine)


async def get_glossary_indexer() -> GlossaryIndexer:
    """Get or create GlossaryIndexer instance (async-safe via AsyncSingleton)."""

    async def _create_glossary_indexer() -> GlossaryIndexer:
        # Get indexer for shared resources
        indexer = await get_indexer()
        return GlossaryIndexer(
            glossary_store=get_glossary_store(),
            collection_name=indexer.collection_name,
            storage=indexer.storage,
            embedder=indexer.embedder,
            global_vocab=indexer.global_vocab,
        )

    return await _glossary_indexer.get(_create_glossary_indexer)


async def get_fact_indexer() -> FactIndexer:
    """Get or create FactIndexer instance (async-safe via AsyncSingleton)."""

    async def _create_fact_indexer() -> FactIndexer:
        # Get indexer for shared resources
        indexer = await get_indexer()
        return FactIndexer(
            fact_store=get_fact_store(),
            storage=indexer.storage,
            embedder=indexer.embedder,
            global_vocab=indexer.global_vocab,
            collection_name=indexer.collection_name,
        )

    return await _fact_indexer.get(_create_fact_indexer)


async def get_note_service() -> NoteService:
    """Get or create NoteService instance (async-safe via AsyncSingleton)."""

    async def _create_note_service() -> NoteService:
        return NoteService(
            store=get_store(),
            git=get_git(),
            indexer=await get_indexer(),
            integrity=get_integrity_manager(),
        )

    return await _note_service.get(_create_note_service)


# ============= Cleanup =============


async def cleanup_async_resources() -> None:
    """Cleanup async resources on shutdown using AsyncSingleton's cleanup."""
    # Release dependents before the shared indexer and its clients.
    await _note_service.close()
    await _fact_indexer.close(lambda f: f.close() if hasattr(f, "close") else None)
    await _glossary_indexer.close(lambda g: g.close() if hasattr(g, "close") else None)
    await _search_engine.close(lambda s: s.close() if hasattr(s, "close") else None)
    await _indexer.close(lambda i: i.close() if hasattr(i, "close") else None)

    # close() returns early when initialization failed before producing an
    # instance, so reset explicitly to clear cached errors and loop-bound locks.
    for singleton in (
        _note_service,
        _fact_indexer,
        _glossary_indexer,
        _search_engine,
        _indexer,
    ):
        singleton.reset()


def _sync_cleanup() -> None:
    """Sync wrapper for cleanup, called on exit."""
    singletons: list[AsyncSingleton[Any]] = [
        _note_service,
        _indexer,
        _search_engine,
        _glossary_indexer,
        _fact_indexer,
    ]
    if not any(s.is_initialized for s in singletons):
        return
    sync_cleanup_wrapper(cleanup_async_resources, singletons)


# Register cleanup handler
atexit.register(_sync_cleanup)
