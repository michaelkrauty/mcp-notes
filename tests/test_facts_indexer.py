"""Tests for facts indexer."""

from pathlib import Path
from uuid import uuid4

import pytest

from mcp_notes.facts import Fact, FactIndexer, FactStore, generate_fact_text


class TestGenerateFactText:
    """Tests for generate_fact_text function."""

    def test_basic_fact(self):
        """Generate text from basic fact."""
        fact = Fact(
            id=uuid4(),
            subject="John",
            subject_type="",
            predicate="knows",
            object_value="Jane",
            object_type="",
            spo_hash="test",
            created=None,
            modified=None,
        )
        # Create a minimal fact with datetime
        from datetime import UTC, datetime
        fact.created = datetime.now(UTC)
        fact.modified = datetime.now(UTC)

        text = generate_fact_text(fact)

        assert "John" in text
        assert "knows" in text
        assert "Jane" in text

    def test_fact_with_types(self):
        """Generate text includes types."""
        from datetime import UTC, datetime
        fact = Fact(
            id=uuid4(),
            subject="Python",
            subject_type="language",
            predicate="created_by",
            object_value="Guido",
            object_type="person",
            spo_hash="test",
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        text = generate_fact_text(fact)

        assert "Python" in text
        assert "language" in text
        assert "created by" in text  # underscores replaced with spaces
        assert "Guido" in text
        assert "person" in text

    def test_fact_with_context(self):
        """Generate text includes context."""
        from datetime import UTC, datetime
        fact = Fact(
            id=uuid4(),
            subject="John",
            subject_type="person",
            predicate="works_at",
            object_value="Acme",
            object_type="company",
            spo_hash="test",
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
            context="as senior engineer since 2020",
        )

        text = generate_fact_text(fact)

        assert "as senior engineer since 2020" in text


@pytest.fixture
def temp_fact_store(tmp_path):
    """Create temporary fact store for tests."""
    db_path = tmp_path / "test_facts.db"
    store = FactStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def fact_indexer(tmp_path, temp_fact_store):
    """Create FactIndexer with temporary dependencies."""
    # We'll use mock/minimal dependencies for unit tests
    # Integration tests with actual Qdrant would be separate
    indexer = FactIndexer(
        fact_store=temp_fact_store,
        storage=None,  # Will use default
        embedder=None,  # Will use default
        global_vocab=None,  # Will use default
        collection_name="test_facts_collection",  # Required by vector-core
    )
    return indexer


class TestFactIndexer:
    """Tests for FactIndexer class."""

    def test_collection_name(self, fact_indexer):
        """Collection name matches what was provided."""
        name = fact_indexer.collection_name
        assert name is not None
        assert name == "test_facts_collection"

    def test_collection_name_cached(self, fact_indexer):
        """Collection name is cached."""
        name1 = fact_indexer.collection_name
        name2 = fact_indexer.collection_name
        assert name1 == name2  # Same value (cached)


class TestFactIndexerIntegration:
    """Integration tests requiring Qdrant (skipped if unavailable)."""

    @pytest.fixture
    async def integration_indexer(self, tmp_path, temp_fact_store):
        """Create FactIndexer for integration tests."""
        import asyncio
        from vector_core import EmbeddingClient, QdrantStorage
        from vector_core.embeddings.global_vocab import GlobalVocabulary

        # Create real components
        storage = QdrantStorage()
        embedder = EmbeddingClient()
        global_vocab = GlobalVocabulary()

        collection_name = "test_facts_integration"

        indexer = FactIndexer(
            fact_store=temp_fact_store,
            storage=storage,
            embedder=embedder,
            global_vocab=global_vocab,
            collection_name=collection_name,
        )

        yield indexer

        # Cleanup: delete the collection and close resources
        try:
            await storage.delete_collection(collection_name)
        except Exception:
            pass  # Ignore cleanup errors
        await storage.close()

    @pytest.mark.asyncio
    async def test_index_empty(self, integration_indexer):
        """Index with no facts."""
        result = await integration_indexer.index_all()

        assert result["total"] == 0
        assert result["indexed"] == 0

    @pytest.mark.asyncio
    async def test_index_single_fact(self, integration_indexer, temp_fact_store):
        """Index single fact."""
        # Add a fact
        fact = temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )

        result = await integration_indexer.index_all()

        assert result["total"] == 1
        assert result["indexed"] == 1

    @pytest.mark.asyncio
    async def test_index_multiple_facts(self, integration_indexer, temp_fact_store):
        """Index multiple facts."""
        # Add several facts
        temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )
        temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="works_at",
            object_value="Acme",
            object_type="company",
        )
        temp_fact_store.create(
            subject="Jane",
            subject_type="person",
            predicate="lives_in",
            object_value="NYC",
            object_type="city",
        )

        result = await integration_indexer.index_all()

        assert result["total"] == 3
        assert result["indexed"] == 3

    @pytest.mark.asyncio
    async def test_incremental_index(self, integration_indexer, temp_fact_store):
        """Incremental indexing only indexes new facts."""
        # First indexing
        temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )
        await integration_indexer.index_all()

        # Add another fact
        temp_fact_store.create(
            subject="Alice",
            subject_type="person",
            predicate="knows",
            object_value="Bob",
            object_type="person",
        )

        # Second incremental indexing
        result = await integration_indexer.index_all(force=False)

        assert result["total"] == 2
        assert result["indexed"] == 1  # Only the new one

    @pytest.mark.asyncio
    async def test_force_reindex(self, integration_indexer, temp_fact_store):
        """Force reindex reindexes all facts."""
        # First indexing
        temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )
        await integration_indexer.index_all()

        # Force reindex
        result = await integration_indexer.index_all(force=True)

        assert result["total"] == 1
        assert result["indexed"] == 1  # Reindexed

    @pytest.mark.asyncio
    async def test_delete_fact_index(self, integration_indexer, temp_fact_store):
        """Delete fact from index."""
        # Add and index a fact
        fact = temp_fact_store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )
        await integration_indexer.index_all()

        # Delete from index
        await integration_indexer.delete_fact_index(fact.id)

        # Verify by checking indexed IDs
        indexed = await integration_indexer._get_indexed_fact_ids()
        assert str(fact.id) not in indexed

    @pytest.mark.asyncio
    async def test_cleanup(self, integration_indexer):
        """Cleanup closes resources."""
        await integration_indexer.close()
        # Should not raise
