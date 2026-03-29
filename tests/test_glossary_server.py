"""Tests for glossary MCP tools."""

from uuid import UUID

import pytest

from mcp_notes.server import (
    add_glossary_entry,
    delete_glossary_entry,
    get_glossary_indexer,
    get_glossary_store,
    list_glossary,
    lookup_term,
    search_glossary,
    update_glossary_entry,
)


class TestGlossarySingletons:
    """Tests for glossary singleton getters."""

    def test_get_glossary_store_singleton(self):
        """get_glossary_store returns same instance."""
        import mcp_notes.singletons as singletons_module

        # Save and reset using SyncSingleton API
        original = singletons_module._glossary_store.get_if_initialized()
        singletons_module._glossary_store.reset()

        try:
            s1 = get_glossary_store()
            s2 = get_glossary_store()
            assert s1 is s2
        finally:
            # Restore using SyncSingleton API
            singletons_module._glossary_store.set_instance(original)

    @pytest.mark.asyncio
    async def test_get_glossary_indexer_singleton(self):
        """get_glossary_indexer returns same instance (AsyncSingleton pattern)."""
        import mcp_notes.singletons as singletons_module

        # Reset singleton for clean test
        singletons_module._glossary_indexer.reset()

        try:
            i1 = await get_glossary_indexer()
            i2 = await get_glossary_indexer()
            assert i1 is i2
        finally:
            singletons_module._glossary_indexer.reset()


class TestAddGlossaryEntry:
    """Tests for add_glossary_entry tool."""

    @pytest.mark.asyncio
    async def test_add_basic(self, tmp_notes_dir):
        """Add a basic glossary entry."""
        result = await add_glossary_entry(
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch of the US Armed Forces.",
        )

        assert "error" not in result
        assert result["term"] == "USAF"
        assert result["expansion"] == "United States Air Force"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_add_with_domain(self, tmp_notes_dir):
        """Add entry with domain."""
        result = await add_glossary_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
            domain="tech",
        )

        assert "error" not in result
        assert result["domain"] == "tech"

    @pytest.mark.asyncio
    async def test_add_with_aliases(self, tmp_notes_dir):
        """Add entry with aliases."""
        result = await add_glossary_entry(
            term="POTUS",
            expansion="President of the United States",
            definition="The head of state.",
            aliases=["President", "US President"],
        )

        assert "error" not in result
        assert "President" in result["aliases"]

    @pytest.mark.asyncio
    async def test_add_duplicate_error(self, tmp_notes_dir):
        """Adding duplicate term returns error."""
        await add_glossary_entry(
            term="HTTP",
            expansion="Hypertext Transfer Protocol",
            definition="A protocol.",
        )

        result = await add_glossary_entry(
            term="http",  # Case-insensitive duplicate
            expansion="Different",
            definition="Different.",
        )

        assert "error_code" in result


class TestLookupTerm:
    """Tests for lookup_term tool."""

    @pytest.mark.asyncio
    async def test_lookup_success(self, tmp_notes_dir):
        """Lookup existing term."""
        await add_glossary_entry(
            term="NATO",
            expansion="North Atlantic Treaty Organization",
            definition="A military alliance.",
        )

        result = await lookup_term("NATO")

        assert "error" not in result
        assert result["term"] == "NATO"

    @pytest.mark.asyncio
    async def test_lookup_case_insensitive(self, tmp_notes_dir):
        """Lookup is case-insensitive."""
        await add_glossary_entry(
            term="CPU",
            expansion="Central Processing Unit",
            definition="The processor.",
        )

        # Various cases should all work
        result1 = await lookup_term("cpu")
        result2 = await lookup_term("CPU")
        result3 = await lookup_term("Cpu")

        assert result1["term"] == "CPU"
        assert result2["term"] == "CPU"
        assert result3["term"] == "CPU"

    @pytest.mark.asyncio
    async def test_lookup_by_alias(self, tmp_notes_dir):
        """Lookup via alias."""
        await add_glossary_entry(
            term="RAM",
            expansion="Random Access Memory",
            definition="Volatile memory.",
            aliases=["memory", "main memory"],
        )

        result = await lookup_term("memory")

        assert "error" not in result
        assert result["term"] == "RAM"

    @pytest.mark.asyncio
    async def test_lookup_not_found(self, tmp_notes_dir):
        """Lookup unknown term returns error."""
        result = await lookup_term("UNKNOWN")

        assert "error_code" in result


class TestListGlossary:
    """Tests for list_glossary tool."""

    @pytest.mark.asyncio
    async def test_list_all(self, tmp_notes_dir):
        """List all entries."""
        await add_glossary_entry(
            term="A",
            expansion="Alpha",
            definition="First.",
        )
        await add_glossary_entry(
            term="B",
            expansion="Bravo",
            definition="Second.",
        )

        result = await list_glossary()

        assert len(result) == 2
        terms = [e["term"] for e in result]
        assert "A" in terms
        assert "B" in terms

    @pytest.mark.asyncio
    async def test_list_by_domain(self, tmp_notes_dir):
        """List filtered by domain."""
        await add_glossary_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="Tech.",
            domain="tech",
        )
        await add_glossary_entry(
            term="ROI",
            expansion="Return on Investment",
            definition="Finance.",
            domain="finance",
        )

        result = await list_glossary(domain="tech")

        assert len(result) == 1
        assert result[0]["term"] == "API"

    @pytest.mark.asyncio
    async def test_list_with_limit(self, tmp_notes_dir):
        """List with limit."""
        for i in range(10):
            await add_glossary_entry(
                term=f"Term{i}",
                expansion=f"Expansion{i}",
                definition=f"Definition{i}.",
            )

        result = await list_glossary(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_notes_dir):
        """List empty glossary."""
        result = await list_glossary()
        assert result == []


class TestUpdateGlossaryEntry:
    """Tests for update_glossary_entry tool."""

    @pytest.mark.asyncio
    async def test_update_by_term(self, tmp_notes_dir):
        """Update entry by term."""
        await add_glossary_entry(
            term="HTTP",
            expansion="Hypertext Transfer Protocol",
            definition="A protocol.",
        )

        result = await update_glossary_entry(
            term_or_id="HTTP",
            definition="A protocol for web communication.",
        )

        assert "error" not in result
        assert "web communication" in result["definition"]

    @pytest.mark.asyncio
    async def test_update_by_uuid(self, tmp_notes_dir):
        """Update entry by UUID."""
        added = await add_glossary_entry(
            term="DNS",
            expansion="Domain Name System",
            definition="Name resolution.",
        )

        result = await update_glossary_entry(
            term_or_id=added["id"],
            domain="tech",
        )

        assert "error" not in result
        assert result["domain"] == "tech"

    @pytest.mark.asyncio
    async def test_update_not_found(self, tmp_notes_dir):
        """Update unknown entry returns error."""
        result = await update_glossary_entry(
            term_or_id="UNKNOWN",
            definition="New definition.",
        )

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_update_term_conflict(self, tmp_notes_dir):
        """Update to conflicting term returns error."""
        await add_glossary_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="First.",
        )
        await add_glossary_entry(
            term="SDK",
            expansion="Software Development Kit",
            definition="Second.",
        )

        result = await update_glossary_entry(
            term_or_id="SDK",
            term="API",  # Conflicts
        )

        assert "error_code" in result


class TestDeleteGlossaryEntry:
    """Tests for delete_glossary_entry tool."""

    @pytest.mark.asyncio
    async def test_delete_by_term(self, tmp_notes_dir):
        """Delete entry by term."""
        await add_glossary_entry(
            term="TEMP",
            expansion="Temporary",
            definition="To be deleted.",
        )

        result = await delete_glossary_entry("TEMP")

        assert result["success"] is True

        # Verify deleted
        lookup = await lookup_term("TEMP")
        assert "error_code" in lookup

    @pytest.mark.asyncio
    async def test_delete_by_uuid(self, tmp_notes_dir):
        """Delete entry by UUID."""
        added = await add_glossary_entry(
            term="TEMP2",
            expansion="Temporary2",
            definition="To be deleted.",
        )

        result = await delete_glossary_entry(added["id"])

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tmp_notes_dir):
        """Delete unknown entry returns error."""
        result = await delete_glossary_entry("UNKNOWN")

        assert "error_code" in result


class TestSearchGlossary:
    """Tests for search_glossary tool (semantic search).

    Note: These tests may return empty results if Qdrant is unavailable.
    The tests verify the API contract, not the search quality.
    """

    @pytest.mark.asyncio
    async def test_search_basic(self, tmp_notes_dir):
        """Basic semantic search."""
        await add_glossary_entry(
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch of the United States Armed Forces.",
            domain="military",
        )

        result = await search_glossary(query="military aviation")

        # Search should return results or error dict (if Qdrant unavailable)
        assert isinstance(result, (list, dict))
        if isinstance(result, dict):
            # Error response is acceptable if Qdrant unavailable
            assert "error_code" in result or "results" in result

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, tmp_notes_dir):
        """Search with domain filter."""
        await add_glossary_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="Programming interfaces.",
            domain="tech",
        )
        await add_glossary_entry(
            term="ROI",
            expansion="Return on Investment",
            definition="Investment returns.",
            domain="finance",
        )

        result = await search_glossary(
            query="interfaces",
            domain="tech",
        )

        # Search should return results or error dict (if Qdrant unavailable)
        assert isinstance(result, (list, dict))
        # If results returned as list, check domain filtering
        if isinstance(result, list):
            for r in result:
                if "domain" in r.get("note", {}):
                    assert "tech" in r["note"].get("tags", [])


class TestGlossaryIntegration:
    """Integration tests for glossary workflow."""

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, tmp_notes_dir):
        """Test complete CRUD workflow."""
        # Create
        added = await add_glossary_entry(
            term="TEST",
            expansion="Test Term",
            definition="A test term for validation.",
            domain="test",
            aliases=["testing"],
        )
        assert "error" not in added

        # Read
        read = await lookup_term("TEST")
        assert read["term"] == "TEST"

        # Read by alias
        read_alias = await lookup_term("testing")
        assert read_alias["term"] == "TEST"

        # Update
        updated = await update_glossary_entry(
            term_or_id="TEST",
            definition="Updated definition.",
        )
        assert "Updated" in updated["definition"]

        # List
        entries = await list_glossary(domain="test")
        assert len(entries) == 1

        # Delete
        deleted = await delete_glossary_entry("TEST")
        assert deleted["success"] is True

        # Verify gone
        gone = await lookup_term("TEST")
        assert "error_code" in gone

    @pytest.mark.asyncio
    async def test_multiple_entries_different_domains(self, tmp_notes_dir):
        """Test multiple entries across domains."""
        domains = ["military", "tech", "finance"]
        for i, domain in enumerate(domains):
            await add_glossary_entry(
                term=f"TERM{i}",
                expansion=f"Term {i} Expansion",
                definition=f"Definition for domain {domain}.",
                domain=domain,
            )

        # Check all entries created
        all_entries = await list_glossary()
        assert len(all_entries) == 3

        # Check domain filtering
        for domain in domains:
            filtered = await list_glossary(domain=domain)
            assert len(filtered) == 1
            assert filtered[0]["domain"] == domain
