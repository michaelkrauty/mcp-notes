"""Integration tests for facts MCP tools."""

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import mcp_notes.tools.facts as facts_mod
from mcp_notes.facts import FactStore


# Import server functions directly for testing
from mcp_notes.server import (
    add_fact,
    add_facts_batch,
    delete_fact,
    find_connections,
    get_entity,
    get_fact_store,
    list_facts,
    query_facts,
    update_fact,
)


@pytest.fixture
def temp_fact_store(tmp_path, monkeypatch):
    """Create temporary fact store for tests."""
    import mcp_notes.singletons as singletons_module

    # Save original using SyncSingleton API
    original_store = singletons_module._fact_store.get_if_initialized()

    # Create temp store and inject using SyncSingleton API
    db_path = tmp_path / "test_facts.db"
    temp_store = FactStore(db_path=db_path)
    singletons_module._fact_store.set_instance(temp_store)

    yield temp_store

    # Cleanup and restore using SyncSingleton API
    temp_store.close()
    singletons_module._fact_store.set_instance(original_store)


class TestAddFact:
    """Tests for add_fact tool."""

    @pytest.mark.asyncio
    async def test_add_basic_fact(self, temp_fact_store):
        """Add basic fact."""
        result = await add_fact(
            subject="John Smith",
            predicate="works_at",
            object="Acme Corp",
        )

        assert "id" in result
        assert result["subject"] == "John Smith"
        assert result["predicate"] == "works_at"
        assert result["object"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_add_fact_with_types(self, temp_fact_store):
        """Add fact with types."""
        result = await add_fact(
            subject="John Smith",
            predicate="served_in",
            object="101st Airborne",
            subject_type="person",
            object_type="military_unit",
        )

        assert result["subject_type"] == "person"
        assert result["object_type"] == "military_unit"

    @pytest.mark.asyncio
    async def test_add_fact_with_dates(self, temp_fact_store):
        """Add fact with validity dates."""
        result = await add_fact(
            subject="John",
            predicate="worked_at",
            object="Company",
            valid_from="2020-01-01",
            valid_to="2022-12-31",
        )

        assert result["valid_from"] == "2020-01-01"
        assert result["valid_to"] == "2022-12-31"

    @pytest.mark.asyncio
    async def test_add_fact_invalid_date(self, temp_fact_store):
        """Invalid date returns error."""
        result = await add_fact(
            subject="John",
            predicate="works_at",
            object="Company",
            valid_from="not-a-date",
        )

        assert "error_code" in result
        assert "date format" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_add_fact_with_source(self, temp_fact_store):
        """Add fact with source."""
        note_id = str(uuid4())
        result = await add_fact(
            subject="John",
            predicate="knows",
            object="Jane",
            source_type="note",
            source_id=note_id,
            source_location="paragraph 3",
        )

        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_type"] == "note"
        assert result["sources"][0]["source_id"] == note_id

    @pytest.mark.asyncio
    async def test_add_duplicate_fact(self, temp_fact_store):
        """Duplicate fact returns error."""
        await add_fact(
            subject="John",
            predicate="knows",
            object="Jane",
        )

        result = await add_fact(
            subject="John",
            predicate="knows",
            object="Jane",
        )

        assert "error_code" in result
        assert "already exists" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_add_fact_invalid_source_type(self, temp_fact_store):
        """Invalid source type returns error."""
        result = await add_fact(
            subject="John",
            predicate="knows",
            object="Jane",
            source_type="invalid",
        )

        assert "error_code" in result
        assert "source_type" in result["message"]


class TestAddFactsBatch:
    """Tests for add_facts_batch tool."""

    @pytest.mark.asyncio
    async def test_add_batch(self, temp_fact_store):
        """Add multiple facts in batch."""
        facts = [
            {"subject": "A", "predicate": "p", "object": "B"},
            {"subject": "C", "predicate": "p", "object": "D"},
            {"subject": "E", "predicate": "p", "object": "F"},
        ]

        result = await add_facts_batch(facts)

        assert result["added"] == 3
        assert result["duplicates"] == 0
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_add_batch_with_duplicates(self, temp_fact_store):
        """Batch handles duplicates gracefully."""
        # Add one first
        await add_fact(subject="A", predicate="p", object="B")

        facts = [
            {"subject": "A", "predicate": "p", "object": "B"},  # Duplicate
            {"subject": "C", "predicate": "p", "object": "D"},  # New
        ]

        result = await add_facts_batch(facts)

        assert result["added"] == 1
        assert result["duplicates"] == 1

    @pytest.mark.asyncio
    async def test_add_batch_with_errors(self, temp_fact_store):
        """Batch reports errors for invalid entries."""
        facts = [
            {"subject": "A", "predicate": "p", "object": "B"},
            {"subject": "C"},  # Missing predicate and object
        ]

        result = await add_facts_batch(facts)

        assert result["added"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["index"] == 1


class TestUpdateFact:
    """Tests for update_fact tool."""

    @pytest.mark.asyncio
    async def test_update_context(self, temp_fact_store):
        """Update fact context."""
        created = await add_fact(
            subject="John",
            predicate="works_at",
            object="Company",
        )

        result = await update_fact(
            fact_id=created["id"],
            context="as engineer",
        )

        assert result["context"] == "as engineer"

    @pytest.mark.asyncio
    async def test_update_confidence(self, temp_fact_store):
        """Update fact confidence."""
        created = await add_fact(
            subject="John",
            predicate="works_at",
            object="Company",
            confidence=0.5,
        )

        result = await update_fact(
            fact_id=created["id"],
            confidence=0.9,
        )

        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_update_not_found(self, temp_fact_store):
        """Update non-existent fact returns error."""
        fake_id = str(uuid4())
        result = await update_fact(
            fact_id=fake_id,
            context="test",
        )

        assert "error_code" in result
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_update_invalid_uuid(self, temp_fact_store):
        """Update with invalid UUID returns error."""
        result = await update_fact(
            fact_id="not-a-uuid",
            context="test",
        )

        assert "error_code" in result
        assert "invalid uuid" in result["message"].lower()


class TestDeleteFact:
    """Tests for delete_fact tool."""

    @pytest.mark.asyncio
    async def test_delete_fact(self, temp_fact_store):
        """Delete existing fact."""
        created = await add_fact(
            subject="John",
            predicate="knows",
            object="Jane",
        )

        result = await delete_fact(created["id"])

        assert result["success"] is True
        assert result["deleted_id"] == created["id"]

    @pytest.mark.asyncio
    async def test_delete_not_found(self, temp_fact_store):
        """Delete non-existent fact returns error."""
        fake_id = str(uuid4())
        result = await delete_fact(fake_id)

        assert "error_code" in result
        assert "not found" in result["message"].lower()


class TestQueryFacts:
    """Tests for query_facts tool."""

    @pytest.mark.asyncio
    async def test_query_by_subject(self, temp_fact_store):
        """Query by subject."""
        await add_fact(subject="John", predicate="knows", object="Jane")
        await add_fact(subject="John", predicate="works_with", object="Bob")
        await add_fact(subject="Alice", predicate="knows", object="Bob")

        result = await query_facts(subject="John")

        assert len(result) == 2
        assert all(f["subject"] == "John" for f in result)

    @pytest.mark.asyncio
    async def test_query_by_predicate(self, temp_fact_store):
        """Query by predicate."""
        await add_fact(subject="John", predicate="knows", object="Jane")
        await add_fact(subject="John", predicate="works_with", object="Bob")

        result = await query_facts(predicate="knows")

        assert len(result) == 1
        assert result[0]["predicate"] == "knows"

    @pytest.mark.asyncio
    async def test_query_case_insensitive(self, temp_fact_store):
        """Query is case-insensitive."""
        await add_fact(subject="John Smith", predicate="KNOWS", object="Jane")

        result = await query_facts(subject="john smith")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_query_with_limit(self, temp_fact_store):
        """Query respects limit."""
        for i in range(10):
            await add_fact(subject=f"Person{i}", predicate="exists", object="World")

        result = await query_facts(predicate="exists", limit=5)

        assert len(result) == 5


class TestGetEntity:
    """Tests for get_entity tool."""

    @pytest.mark.asyncio
    async def test_get_entity(self, temp_fact_store):
        """Get entity with facts."""
        await add_fact(subject="John", predicate="knows", object="Jane")
        await add_fact(subject="John", predicate="works_at", object="Acme")
        await add_fact(subject="Alice", predicate="knows", object="John")

        result = await get_entity("John")

        assert result["entity"] == "John"
        assert len(result["as_subject"]) == 2
        assert len(result["as_object"]) == 1
        assert result["total_facts"] == 3

    @pytest.mark.asyncio
    async def test_get_entity_with_type(self, temp_fact_store):
        """Get entity filtered by type."""
        await add_fact(
            subject="Python",
            subject_type="language",
            predicate="created_by",
            object="Guido",
            object_type="person",
        )
        await add_fact(
            subject="Python",
            subject_type="snake",
            predicate="is_a",
            object="Reptile",
            object_type="class",
        )

        result = await get_entity("Python", entity_type="language")

        assert result["total_facts"] == 1


class TestListFacts:
    """Tests for list_facts tool."""

    @pytest.mark.asyncio
    async def test_list_facts(self, temp_fact_store):
        """List facts as summaries."""
        await add_fact(subject="A", predicate="p", object="B")
        await add_fact(subject="C", predicate="p", object="D")

        result = await list_facts()

        assert len(result) == 2
        # Summaries have source_count
        assert "source_count" in result[0]

    @pytest.mark.asyncio
    async def test_list_facts_with_filter(self, temp_fact_store):
        """List facts with type filter."""
        await add_fact(
            subject="John",
            subject_type="person",
            predicate="works_at",
            object="Acme",
            object_type="organization",
        )
        await add_fact(
            subject="Python",
            subject_type="language",
            predicate="created_by",
            object="Guido",
            object_type="person",
        )

        result = await list_facts(subject_type="person")

        assert len(result) == 1
        assert result[0]["subject"] == "John"


class TestFindConnections:
    """Tests for find_connections tool."""

    @pytest.mark.asyncio
    async def test_entity_chain_forward_edges(self, temp_fact_store):
        """Entity chain is correct when the path follows subject->object."""
        await add_fact(subject="Alice", predicate="manages", object="Bob")
        await add_fact(subject="Bob", predicate="mentors", object="Carol")

        result = await find_connections(source_entity="Alice", target_entity="Carol")

        assert len(result) >= 1
        assert result[0]["entities"] == ["Alice", "Bob", "Carol"]

    @pytest.mark.asyncio
    async def test_entity_chain_backward_edges(self, temp_fact_store):
        """Entity chain is correct when the path traverses edges backwards.

        The store's BFS is undirected; the chain must not assume every fact
        was traversed subject->object (issue #13).
        """
        await add_fact(subject="Bob", predicate="manages", object="Alice")
        await add_fact(subject="Carol", predicate="mentors", object="Bob")

        result = await find_connections(source_entity="Alice", target_entity="Carol")

        assert len(result) >= 1
        assert result[0]["entities"] == ["Alice", "Bob", "Carol"]

    @pytest.mark.asyncio
    async def test_entity_chain_case_insensitive_cursor(self, temp_fact_store):
        """Chain rebuild matches entities case-insensitively like the BFS."""
        await add_fact(subject="Bob", predicate="manages", object="Alice")
        await add_fact(subject="Carol", predicate="mentors", object="Bob")

        result = await find_connections(source_entity="alice", target_entity="carol")

        assert len(result) >= 1
        # Entity names come from the stored facts, not the query casing
        assert result[0]["entities"] == ["Alice", "Bob", "Carol"]

    @pytest.mark.asyncio
    async def test_type_filters_match_mixed_case(self, temp_fact_store):
        """source_type/target_type filters are case-insensitive (vector-core#18).

        Adjacency rows store types lowercased; passing a type exactly as facts
        display it (e.g. "Person") previously returned no paths.
        """
        await add_fact(
            subject="Alice",
            predicate="manages",
            object="Bob",
            subject_type="Person",
            object_type="Person",
        )

        result = await find_connections(
            source_entity="Alice",
            target_entity="Bob",
            source_type="Person",
            target_type="Person",
        )

        assert len(result) >= 1
        assert result[0]["entities"] == ["Alice", "Bob"]


class TestFactDateRangeValidation:
    """add_fact/add_facts_batch/update_fact must reject an inverted
    valid_from>valid_to range with a structured error. vector-core v1.2.7 raises
    ValueError for it; the tool layer must not surface a generic error or abort a
    batch (it previously caught only DuplicateFactError around store.create)."""

    @pytest.mark.asyncio
    async def test_add_fact_rejects_inverted_range(self, temp_fact_store):
        result = await add_fact(
            subject="A", predicate="r", object="B",
            valid_from="2025-01-01", valid_to="2024-01-01",
        )
        assert "error_code" in result
        assert "valid_from" in str(result)
        assert temp_fact_store.count() == 0

    @pytest.mark.asyncio
    async def test_add_fact_accepts_ordered_range(self, temp_fact_store):
        result = await add_fact(
            subject="A", predicate="r", object="B",
            valid_from="2024-01-01", valid_to="2025-01-01",
        )
        assert "id" in result

    @pytest.mark.asyncio
    async def test_batch_inverted_item_does_not_abort_others(self, temp_fact_store):
        facts = [
            {"subject": "A", "predicate": "r", "object": "B"},
            {"subject": "C", "predicate": "r", "object": "D",
             "valid_from": "2025-01-01", "valid_to": "2024-01-01"},
            {"subject": "E", "predicate": "r", "object": "F"},
        ]
        result = await add_facts_batch(facts)
        assert result["added"] == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0]["index"] == 1
        assert "valid_from" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_update_fact_rejects_inverting_range(self, temp_fact_store):
        created = await add_fact(
            subject="A", predicate="r", object="B", valid_to="2024-01-01",
        )
        result = await update_fact(fact_id=created["id"], valid_from="2025-01-01")
        assert "error_code" in result
        assert "valid_from" in str(result)
        # the fact's existing value is untouched
        assert temp_fact_store.read(UUID(created["id"])).valid_from is None


class TestFactIndexSync:
    """delete_fact and update_fact must keep the semantic fact index in sync:
    search_facts reads straight from the Qdrant payload with no existence check
    against the store, so a deleted fact's stale point would keep being returned
    and an updated fact would return a stale payload."""

    @pytest.mark.asyncio
    async def test_delete_fact_removes_point_from_index(
        self, temp_fact_store, monkeypatch
    ):
        indexer = AsyncMock()
        monkeypatch.setattr(
            facts_mod, "get_fact_indexer", AsyncMock(return_value=indexer)
        )

        created = await add_fact(subject="Zorblax", predicate="rules", object="Mars")
        indexer.reset_mock()  # ignore any indexing during add

        result = await delete_fact(created["id"])

        assert result["success"] is True
        indexer.delete_fact_index.assert_awaited_once()
        assert str(indexer.delete_fact_index.await_args.args[0]) == created["id"]
