"""Tests for mcp_notes.constants tag helpers."""

from mcp_notes.constants import normalize_tag, validate_tag


class TestNormalizeTag:
    """normalize_tag is the single source of truth for stored tag form."""

    def test_lowercases(self):
        assert normalize_tag("Work") == "work"

    def test_strips_surrounding_whitespace(self):
        assert normalize_tag("  work  ") == "work"

    def test_spaces_become_hyphens(self):
        assert normalize_tag("my tag") == "my-tag"

    def test_combined_normalization(self):
        assert normalize_tag("  My Tag ") == "my-tag"

    def test_blank_becomes_empty_string(self):
        assert normalize_tag("   ") == ""


class TestValidateTagUsesNormalize:
    """validate_tag delegates to normalize_tag and keeps its contract."""

    def test_valid_tag_is_normalized(self):
        normalized, error = validate_tag("My Tag")
        assert normalized == "my-tag"
        assert error is None

    def test_blank_tag_rejected(self):
        normalized, error = validate_tag("   ")
        assert normalized == ""
        assert error is not None
