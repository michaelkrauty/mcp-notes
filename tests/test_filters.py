"""Tests for search filters (query parsing and filtering)."""

from datetime import UTC, datetime

from mcp_notes.search.filters import (
    SearchFilters,
    apply_post_filters,
    filters_to_qdrant,
    parse_search_query,
)


class TestParseSearchQuery:
    """Tests for parse_search_query function."""

    def test_plain_query(self):
        """Plain query without filters."""
        result = parse_search_query("meeting notes about authentication")

        assert result.query == "meeting notes about authentication"
        assert result.tags == []
        assert result.category is None
        assert result.after is None
        assert result.before is None

    def test_single_tag(self):
        """Query with single tag filter."""
        result = parse_search_query("meeting notes tag:work")

        assert result.query == "meeting notes"
        assert result.tags == ["work"]

    def test_multiple_tags(self):
        """Query with multiple tag filters."""
        result = parse_search_query("notes tag:python tag:testing")

        assert result.query == "notes"
        assert "python" in result.tags
        assert "testing" in result.tags

    def test_exclude_tag(self):
        """Query with excluded tag."""
        result = parse_search_query("-tag:personal notes")

        assert result.exclude_tags == ["personal"]
        assert result.tags == []
        assert result.query == "notes"

    def test_mixed_include_exclude_tags(self):
        """Query with both include and exclude tags."""
        result = parse_search_query("-tag:draft notes tag:work")

        assert "work" in result.tags
        assert "draft" in result.exclude_tags
        assert "notes" in result.query

    def test_category_filter(self):
        """Query with category filter."""
        result = parse_search_query("notes category:projects/mcp")

        assert result.query == "notes"
        assert result.category == "projects/mcp"

    def test_category_filter_normalized_to_slug(self):
        """A non-slug category is normalized to its stored slug form, so a
        ``category:Work`` filter matches the stored ``work`` instead of
        silently matching nothing (sibling of tag normalization)."""
        assert parse_search_query("notes category:Work").category == "work"
        assert (
            parse_search_query("notes category:Work/ProjectX").category
            == "work/projectx"
        )

    def test_after_date(self):
        """Query with after date filter."""
        result = parse_search_query("notes after:2024-01-15")

        assert result.query == "notes"
        assert result.after is not None
        assert result.after.year == 2024
        assert result.after.month == 1
        assert result.after.day == 15
        assert result.after.tzinfo == UTC

    def test_before_date(self):
        """Query with before date filter."""
        result = parse_search_query("notes before:2024-06-30")

        assert result.query == "notes"
        assert result.before is not None
        assert result.before.year == 2024
        assert result.before.month == 6
        assert result.before.day == 30

    def test_date_range(self):
        """Query with both after and before dates."""
        result = parse_search_query("notes after:2024-01-01 before:2024-12-31")

        assert result.after is not None
        assert result.before is not None
        assert result.after < result.before

    def test_title_filter(self):
        """Query with title filter."""
        result = parse_search_query("notes title:meeting")

        assert result.query == "notes"
        assert result.title_contains == "meeting"

    def test_all_filters_combined(self):
        """Query with multiple different filters."""
        query = "search term tag:work category:projects after:2024-01-01 title:auth"

        result = parse_search_query(query)

        assert result.query == "search term"
        assert "work" in result.tags
        assert result.category == "projects"
        assert result.after is not None
        assert result.title_contains == "auth"

    def test_tags_normalized_lowercase(self):
        """Tags are normalized to lowercase."""
        result = parse_search_query("notes tag:UPPERCASE tag:MixedCase")

        assert "uppercase" in result.tags
        assert "mixedcase" in result.tags

    def test_duplicate_tags_deduplicated(self):
        """Duplicate tags are deduplicated."""
        result = parse_search_query("notes tag:work tag:work tag:work")

        assert result.tags.count("work") == 1

    def test_invalid_date_ignored(self):
        """Invalid date format is ignored."""
        # The AFTER_PATTERN only matches YYYY-MM-DD format
        # 'invalid-date' doesn't match the pattern, so it stays in query
        result = parse_search_query("notes after:not-a-date")

        # Pattern doesn't match, so 'after:not-a-date' remains in query
        assert "after:not-a-date" in result.query or result.after is None

    def test_invalid_uuid_ignored(self):
        """Invalid UUID in link filter is left in query (links_to not a supported filter)."""
        result = parse_search_query("notes links_to:not-a-uuid")

        # links_to is not a recognized filter, so it stays in the query text
        assert "links_to:not-a-uuid" in result.query

    def test_extra_whitespace_cleaned(self):
        """Extra whitespace in query is cleaned."""
        result = parse_search_query("  search   term   tag:work  ")

        assert result.query == "search term"

    def test_empty_query(self):
        """Empty query produces empty result."""
        result = parse_search_query("")

        assert result.query == ""
        assert result.tags == []

    def test_only_filters_no_query(self):
        """Query with only filters produces empty query string."""
        result = parse_search_query("tag:work category:projects")

        assert result.query == ""
        assert "work" in result.tags
        assert result.category == "projects"


class TestFiltersToQdrant:
    """Tests for filters_to_qdrant function."""

    def test_tag_filter(self):
        """Tag filter produces Qdrant condition."""
        filters = SearchFilters(tags=["work"])

        conditions = filters_to_qdrant(filters)

        assert len(conditions) == 1
        assert conditions[0].key == "tags"

    def test_multiple_tags(self):
        """Multiple tags produce multiple conditions."""
        filters = SearchFilters(tags=["work", "important"])

        conditions = filters_to_qdrant(filters)

        assert len(conditions) == 2

    def test_category_filter(self):
        """Category filter produces Qdrant condition."""
        filters = SearchFilters(category="projects/mcp")

        conditions = filters_to_qdrant(filters)

        assert len(conditions) == 1
        assert conditions[0].key == "category"

    def test_empty_filters(self):
        """Empty filters produce no conditions."""
        filters = SearchFilters()

        conditions = filters_to_qdrant(filters)

        assert conditions == []

    def test_date_filters_not_in_qdrant(self):
        """Date filters are not included (done post-filtering)."""
        filters = SearchFilters(
            after=datetime(2024, 1, 1, tzinfo=UTC),
            before=datetime(2024, 12, 31, tzinfo=UTC),
        )

        conditions = filters_to_qdrant(filters)

        # Date filters require post-filtering, not Qdrant conditions
        assert conditions == []

    def test_exclude_tags_not_in_qdrant(self):
        """Excluded tags are not in Qdrant conditions (done post-filtering)."""
        filters = SearchFilters(exclude_tags=["draft"])

        conditions = filters_to_qdrant(filters)

        # Exclude filters require post-filtering
        assert conditions == []


class TestApplyPostFilters:
    """Tests for apply_post_filters function."""

    def test_exclude_tags(self):
        """Results with excluded tags are filtered out."""
        results = [
            {"payload": {"tags": ["work", "draft"]}},
            {"payload": {"tags": ["work"]}},
            {"payload": {"tags": ["personal"]}},
        ]
        filters = SearchFilters(exclude_tags=["draft"])

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2
        assert not any("draft" in r["payload"]["tags"] for r in filtered)

    def test_after_date_filter(self):
        """Results before after date are filtered out."""
        results = [
            {"payload": {"created": "2024-06-01T00:00:00Z"}},
            {"payload": {"created": "2024-01-01T00:00:00Z"}},
            {"payload": {"created": "2024-12-01T00:00:00Z"}},
        ]
        filters = SearchFilters(after=datetime(2024, 3, 1, tzinfo=UTC))

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2

    def test_before_date_filter(self):
        """Results after before date are filtered out."""
        results = [
            {"payload": {"created": "2024-06-01T00:00:00Z"}},
            {"payload": {"created": "2024-01-01T00:00:00Z"}},
            {"payload": {"created": "2024-12-01T00:00:00Z"}},
        ]
        filters = SearchFilters(before=datetime(2024, 7, 1, tzinfo=UTC))

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2

    def test_date_range(self):
        """Date range filter works correctly."""
        results = [
            {"payload": {"created": "2024-06-01T00:00:00Z"}},  # In range
            {"payload": {"created": "2024-01-01T00:00:00Z"}},  # Before range
            {"payload": {"created": "2024-12-01T00:00:00Z"}},  # After range
        ]
        filters = SearchFilters(
            after=datetime(2024, 3, 1, tzinfo=UTC),
            before=datetime(2024, 9, 1, tzinfo=UTC),
        )

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 1
        assert "2024-06" in filtered[0]["payload"]["created"]

    def test_title_filter(self):
        """Title filter works correctly."""
        results = [
            {"payload": {"title": "Meeting Notes"}},
            {"payload": {"title": "Project Plan"}},
            {"payload": {"title": "Weekly Meeting Summary"}},
        ]
        filters = SearchFilters(title_contains="Meeting")

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2
        assert all("meeting" in r["payload"]["title"].lower() for r in filtered)

    def test_title_filter_case_insensitive(self):
        """Title filter is case insensitive."""
        results = [
            {"payload": {"title": "UPPERCASE MEETING"}},
            {"payload": {"title": "lowercase meeting"}},
            {"payload": {"title": "Not relevant"}},
        ]
        filters = SearchFilters(title_contains="meeting")

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2

    def test_combined_filters(self):
        """Multiple post-filters work together."""
        results = [
            {"payload": {"tags": ["work"], "created": "2024-06-01T00:00:00Z", "title": "Meeting"}},
            {"payload": {"tags": ["work", "draft"], "created": "2024-06-01T00:00:00Z", "title": "Meeting"}},
            {"payload": {"tags": ["work"], "created": "2024-01-01T00:00:00Z", "title": "Meeting"}},
            {"payload": {"tags": ["work"], "created": "2024-06-01T00:00:00Z", "title": "Other"}},
        ]
        filters = SearchFilters(
            exclude_tags=["draft"],
            after=datetime(2024, 3, 1, tzinfo=UTC),
            title_contains="Meeting",
        )

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 1

    def test_empty_filters_returns_all(self):
        """Empty filters return all results."""
        results = [
            {"payload": {"tags": ["a"]}},
            {"payload": {"tags": ["b"]}},
        ]
        filters = SearchFilters()

        filtered = apply_post_filters(results, filters)

        assert len(filtered) == 2

    def test_invalid_date_in_payload_handled(self):
        """Invalid date in payload doesn't crash."""
        results = [
            {"payload": {"created": "invalid-date"}},
            {"payload": {"created": "2024-06-01T00:00:00Z"}},
        ]
        filters = SearchFilters(after=datetime(2024, 1, 1, tzinfo=UTC))

        # Should not raise
        filtered = apply_post_filters(results, filters)

        # Result with invalid date may or may not be included depending on implementation
        assert len(filtered) >= 1

    def test_missing_payload_key_handled(self):
        """Missing payload keys don't crash."""
        results = [
            {"payload": {}},  # Missing tags, created, title
            {"payload": {"title": "Has title"}},
        ]
        filters = SearchFilters(
            exclude_tags=["draft"],
            title_contains="title",
        )

        # Should not raise
        filtered = apply_post_filters(results, filters)

        # First result has no matching title, second does
        assert len(filtered) == 1


class TestParseSearchQueryInvalidValues:
    """Tests for invalid filter values in parse_search_query."""

    def test_invalid_after_date_ignored(self):
        """Invalid after date is ignored (lines 98-99)."""
        # 2024-99-99 matches regex pattern but fails strptime
        result = parse_search_query("notes after:2024-99-99")

        # The filter should be ignored (ValueError caught)
        assert result.after is None
        # Pattern matched and was removed from query
        assert "after:" not in result.query

    def test_invalid_before_date_ignored(self):
        """Invalid before date is ignored (lines 108-109)."""
        # 2024-13-45 matches regex but is invalid date
        result = parse_search_query("notes before:2024-13-45")

        assert result.before is None
        assert "before:" not in result.query

    def test_impossible_date_ignored(self):
        """Impossible date like Feb 30 is ignored."""
        # 2024-02-30 matches regex but Feb doesn't have 30 days
        result = parse_search_query("notes after:2024-02-30")

        assert result.after is None
        assert "after:" not in result.query

    def test_valid_and_invalid_date_mixed(self):
        """Valid filters work even with invalid ones present."""
        result = parse_search_query("notes after:2024-99-01 before:2024-06-15")

        # after should be ignored (invalid month), before should work
        assert result.after is None
        assert result.before is not None
        assert result.before.day == 15

