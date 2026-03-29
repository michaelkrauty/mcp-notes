"""Property-based tests using Hypothesis for mcp-notes.

These tests verify that parsing and processing functions never crash
regardless of input, providing robustness guarantees.
"""

import string
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mcp_notes.search.filters import parse_search_query
from mcp_notes.storage.parser import (
    extract_inline_links,
    serialize_note,
)


class TestParseSearchQueryPropertyBased:
    """Property-based tests for search query parsing."""

    @given(st.text(max_size=1000))
    @settings(max_examples=200)
    def test_parse_never_crashes(self, query: str) -> None:
        """parse_search_query should never raise an exception for any input."""
        result = parse_search_query(query)

        # Basic sanity checks
        assert hasattr(result, "query")
        assert hasattr(result, "tags")
        assert hasattr(result, "category")

    @given(st.text(alphabet=string.printable, max_size=500))
    @settings(max_examples=100)
    def test_parse_printable_text(self, query: str) -> None:
        """parse_search_query handles all printable characters."""
        result = parse_search_query(query)
        assert isinstance(result.query, str)

    @given(
        st.text(max_size=50),
        st.sampled_from(["tag:", "category:", "after:", "before:", "title:"]),
    )
    @settings(max_examples=100)
    def test_parse_with_prefixes(self, suffix: str, prefix: str) -> None:
        """parse_search_query handles filter prefix syntax."""
        query = prefix + suffix
        result = parse_search_query(query)
        assert isinstance(result.query, str)

    @given(st.lists(st.text(max_size=30), max_size=10))
    @settings(max_examples=50)
    def test_parse_multi_term_queries(self, terms: list[str]) -> None:
        """parse_search_query handles multi-term queries."""
        query = " ".join(terms)
        result = parse_search_query(query)
        assert isinstance(result.query, str)


class TestExtractInlineLinksPropertyBased:
    """Property-based tests for inline link extraction."""

    @given(st.text(max_size=5000))
    @settings(max_examples=200)
    def test_extract_never_crashes(self, content: str) -> None:
        """extract_inline_links should never raise an exception."""
        result = extract_inline_links(content)
        assert isinstance(result, list)

    @given(st.uuids())
    def test_extract_valid_uuid_links(self, uuid_val) -> None:
        """Valid [[uuid]] links should be extracted."""
        content = f"Check out [[{uuid_val}]] for more info."
        result = extract_inline_links(content)
        assert uuid_val in result  # result is list[UUID]

    @given(st.text(max_size=36, alphabet=string.hexdigits + "-"))
    @settings(max_examples=100)
    def test_extract_uuid_like_strings(self, uuid_like: str) -> None:
        """UUID-like strings in [[]] should not crash extraction."""
        content = f"Link: [[{uuid_like}]]"
        result = extract_inline_links(content)
        assert isinstance(result, list)

    @given(st.lists(st.uuids(), max_size=20))
    @settings(max_examples=50)
    def test_extract_multiple_links(self, uuids: list) -> None:
        """Multiple links should all be extracted."""
        content = " ".join(f"[[{u}]]" for u in uuids)
        result = extract_inline_links(content)
        for u in uuids:
            assert u in result  # result is list[UUID]


class TestSerializeNotePropertyBased:
    """Property-based tests for note serialization."""

    @given(
        st.text(min_size=1, max_size=200),  # title
        st.text(max_size=5000),  # body
    )
    @settings(max_examples=100)
    def test_serialize_never_crashes(self, title: str, body: str) -> None:
        """serialize_note should never crash for valid inputs."""
        assume(title.strip())  # Title must be non-empty after strip

        result = serialize_note(
            note_id=uuid4(),
            title=title,
            body=body,
            tags=None,
            category=None,
            links=None,
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )
        assert isinstance(result, str)
        assert "---" in result  # Should have YAML frontmatter

    @given(
        st.text(min_size=1, max_size=100),  # title
        st.text(max_size=1000),  # body
        st.lists(st.text(min_size=1, max_size=30, alphabet=string.ascii_lowercase + "-"), max_size=10),  # tags
    )
    @settings(max_examples=100)
    def test_serialize_with_tags(self, title: str, body: str, tags: list[str]) -> None:
        """serialize_note handles various tag formats."""
        assume(title.strip())
        # Filter out empty tags
        tags = [t for t in tags if t.strip()]

        result = serialize_note(
            note_id=uuid4(),
            title=title,
            body=body,
            tags=tags if tags else None,
            category=None,
            links=None,
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )
        assert isinstance(result, str)

    @given(
        st.text(min_size=1, max_size=100),  # title
        st.text(max_size=500),  # body
        st.text(max_size=100, alphabet=string.ascii_lowercase + "/"),  # category
    )
    @settings(max_examples=100)
    def test_serialize_with_category(self, title: str, body: str, category: str) -> None:
        """serialize_note handles various category formats."""
        assume(title.strip())

        result = serialize_note(
            note_id=uuid4(),
            title=title,
            body=body,
            tags=None,
            category=category if category.strip() else None,
            links=None,
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )
        assert isinstance(result, str)


class TestSearchQueryEdgeCases:
    """Test edge cases that might cause parsing issues."""

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=500))
    @settings(max_examples=100)
    def test_unicode_handling(self, query: str) -> None:
        """parse_search_query handles Unicode correctly."""
        result = parse_search_query(query)
        assert isinstance(result.query, str)

    @pytest.mark.parametrize("edge_case", [
        "",  # Empty
        " ",  # Space only
        "\n",  # Newline only
        "\t",  # Tab only
        "   \n\t  ",  # Mixed whitespace
        "tag:",  # Prefix only
        "category:",  # Prefix only
        "after:invalid-date",  # Invalid date
        "before:not-a-date",  # Invalid date
        '"',  # Single quote
        '""',  # Empty quotes
        "a" * 10000,  # Very long query
        "tag:a tag:b tag:c",  # Multiple tags
        "category:a/b/c",  # Nested category
        "日本語検索",  # Japanese
        "🔍 search 🔎",  # Emoji
        "title:[[uuid]]",  # Nested syntax
        "tag:work category:projects after:2024-01-01",  # Combined filters
    ])
    def test_edge_case_queries(self, edge_case: str) -> None:
        """Known edge cases should not crash."""
        result = parse_search_query(edge_case)
        assert hasattr(result, "query")


class TestTagNormalization:
    """Property-based tests for tag normalization logic."""

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_tag_normalization_never_crashes(self, tag: str) -> None:
        """Tag normalization should never crash."""
        # This mimics the normalization in NoteStore.create
        try:
            normalized = tag.lower().strip().replace(" ", "-")
            assert isinstance(normalized, str)
        except Exception as e:
            pytest.fail(f"Tag normalization crashed with: {e}")
