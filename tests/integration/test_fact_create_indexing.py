"""Integration tests for semantic indexing on fact creation."""

import pytest

from mcp_notes.server import add_fact, add_facts_batch, search_facts


@pytest.mark.asyncio
async def test_add_fact_is_immediately_semantically_searchable(tmp_notes_dir):
    created = await add_fact(
        subject="Zorblax",
        predicate="discovered",
        object="Quasarfruit",
    )

    results = await search_facts(query="Zorblax Quasarfruit")

    assert created["id"] in {result["fact_id"] for result in results}


@pytest.mark.asyncio
async def test_add_facts_batch_indexes_every_created_fact(tmp_notes_dir):
    result = await add_facts_batch([
        {"subject": "Velmora", "predicate": "charted", "object": "Asterdeep"},
        {"subject": "Kintara", "predicate": "founded", "object": "Brindleforge"},
    ])

    assert result["added"] == 2
    for query in ("Velmora Asterdeep", "Kintara Brindleforge"):
        matches = await search_facts(query=query)
        assert any(query.split()[0] in match["title"] for match in matches)
