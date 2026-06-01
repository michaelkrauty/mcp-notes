"""Tests for tool input validation hardening:

- revalidate_fact_sources refuses a no-argument global reset
- add_fact / update_fact / add_facts_batch reject out-of-range confidence
- rename_tag / merge_tags normalize a spaced source tag to the stored form
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vector_core.errors import ErrorCode, is_error_response

import mcp_notes.singletons as singletons_module
from mcp_notes.facts import FactStore
from mcp_notes.server import (
    add_fact,
    add_facts_batch,
    merge_tags,
    rename_tag,
    revalidate_fact_sources,
    update_fact,
)


@pytest.fixture
def temp_fact_store(tmp_path):
    """Inject a temporary SQLite-backed FactStore (no Qdrant/embeddings needed)."""
    original = singletons_module._fact_store.get_if_initialized()
    store = FactStore(db_path=tmp_path / "test_facts.db")
    singletons_module._fact_store.set_instance(store)
    yield store
    store.close()
    singletons_module._fact_store.set_instance(original)


class TestRevalidateGuard:
    """revalidate_fact_sources must not reset every source on an empty call."""

    async def test_no_args_refuses_global_reset(self):
        # Guard returns before the integrity manager is touched, so no fixture
        # is needed: an accidental empty call must fail fast.
        result = await revalidate_fact_sources()
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value

    async def test_blank_strings_refuse_global_reset(self):
        result = await revalidate_fact_sources(source_id="", source_type="")
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value


class TestConfidenceBounds:
    """confidence is documented as 0.0-1.0; out-of-range values are rejected."""

    async def test_add_fact_rejects_out_of_range(self, temp_fact_store):
        result = await add_fact(
            subject="A", predicate="rel", object="B", confidence=1.5
        )
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value

    async def test_add_fact_rejects_negative(self, temp_fact_store):
        result = await add_fact(
            subject="A", predicate="rel", object="B", confidence=-0.1
        )
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value

    async def test_add_fact_accepts_in_range(self, temp_fact_store):
        result = await add_fact(
            subject="A", predicate="rel", object="B", confidence=0.5
        )
        assert "id" in result
        assert result["confidence"] == 0.5

    async def test_update_fact_rejects_out_of_range(self, temp_fact_store):
        created = await add_fact(subject="A", predicate="rel", object="B")
        result = await update_fact(fact_id=created["id"], confidence=2.0)
        assert is_error_response(result)
        assert result["error_code"] == ErrorCode.INVALID_INPUT.value

    async def test_update_fact_none_confidence_is_allowed(self, temp_fact_store):
        created = await add_fact(subject="A", predicate="rel", object="B")
        # confidence omitted (None) must not trip the bounds check.
        result = await update_fact(fact_id=created["id"], context="updated")
        assert "id" in result

    async def test_batch_rejects_only_the_bad_item(self, temp_fact_store):
        result = await add_facts_batch([
            {"subject": "A", "predicate": "rel", "object": "B", "confidence": 0.9},
            {"subject": "C", "predicate": "rel", "object": "D", "confidence": 5.0},
        ])
        assert result["added"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["index"] == 1
        assert "confidence" in result["errors"][0]["error"]


class TestTagNormalization:
    """rename_tag / merge_tags must normalize a spaced source tag to the
    stored hyphenated form so it actually matches."""

    @staticmethod
    def _mocks_with_tag(stored_tag: str):
        summary = MagicMock()
        summary.id = "note-1"
        summary.title = "Title"
        summary.tags = [stored_tag, "keep"]
        store = MagicMock()
        store.list_all.return_value = [summary]
        store.get_note_path.return_value = "/notes/note-1.md"
        indexer = MagicMock()
        indexer.index_all = AsyncMock()
        return store, indexer

    async def test_rename_tag_matches_spaced_source(self):
        store, indexer = self._mocks_with_tag("my-tag")
        with (
            patch("mcp_notes.tools.tags.get_store", return_value=store),
            patch("mcp_notes.tools.tags.get_git", return_value=MagicMock()),
            patch("mcp_notes.tools.tags.get_indexer", new=AsyncMock(return_value=indexer)),
        ):
            result = await rename_tag(old_tag="my tag", new_tag="renamed")

        assert result["updated_count"] == 1
        new_tags = store.update.call_args.kwargs["tags"]
        assert "renamed" in new_tags
        assert "my-tag" not in new_tags

    async def test_merge_tags_matches_spaced_source(self):
        store, indexer = self._mocks_with_tag("my-tag")
        with (
            patch("mcp_notes.tools.tags.get_store", return_value=store),
            patch("mcp_notes.tools.tags.get_git", return_value=MagicMock()),
            patch("mcp_notes.tools.tags.get_indexer", new=AsyncMock(return_value=indexer)),
        ):
            result = await merge_tags(source_tags=["my tag"], target_tag="merged")

        assert result["updated_count"] == 1
        new_tags = store.update.call_args.kwargs["tags"]
        assert "merged" in new_tags
        assert "my-tag" not in new_tags
