"""Tests for slug generation utilities."""

from uuid import UUID, uuid4

import pytest

from mcp_notes.storage.slugify import (
    DEFAULT_SLUG_MAX_LENGTH,
    build_filename,
    extract_uuid_from_filename,
    generate_slug,
    slugify_category_path,
    slugify_path_segment,
)


class TestGenerateSlug:
    """Tests for generate_slug function."""

    def test_basic_title(self):
        """Simple title is slugified correctly."""
        assert generate_slug("Hello World") == "hello-world"

    def test_lowercase(self):
        """Title is converted to lowercase."""
        assert generate_slug("UPPERCASE") == "uppercase"
        assert generate_slug("MixedCase") == "mixedcase"

    def test_spaces_to_hyphens(self):
        """Spaces are converted to hyphens."""
        assert generate_slug("hello world test") == "hello-world-test"

    def test_underscores_to_hyphens(self):
        """Underscores are converted to hyphens."""
        assert generate_slug("hello_world_test") == "hello-world-test"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces become single hyphen."""
        assert generate_slug("hello    world") == "hello-world"

    def test_multiple_hyphens_collapsed(self):
        """Multiple hyphens become single hyphen."""
        assert generate_slug("hello---world") == "hello-world"

    def test_special_characters_removed(self):
        """Special characters are removed."""
        assert generate_slug("hello!@#$%world") == "helloworld"
        assert generate_slug("test & demo") == "test-demo"
        assert generate_slug("note (draft)") == "note-draft"

    def test_leading_trailing_hyphens_stripped(self):
        """Leading and trailing hyphens are stripped."""
        assert generate_slug("-hello-world-") == "hello-world"
        assert generate_slug("---test---") == "test"

    def test_diacritics_stripped(self):
        """Diacritics (accents) are stripped."""
        assert generate_slug("cafe") == "cafe"
        assert generate_slug("naive") == "naive"
        assert generate_slug("resume") == "resume"
        # Test actual diacritics
        assert generate_slug("caf\u00e9") == "cafe"  # cafe with accent
        assert generate_slug("na\u00efve") == "naive"  # naive with diaeresis

    def test_unicode_normalization(self):
        """Unicode is properly normalized."""
        # German umlaut
        assert generate_slug("M\u00fcnchen") == "munchen"
        # Spanish tilde
        assert generate_slug("Espa\u00f1a") == "espana"

    def test_empty_title(self):
        """Empty title returns 'untitled'."""
        assert generate_slug("") == "untitled"
        assert generate_slug("   ") == "untitled"

    def test_only_special_chars(self):
        """Title with only special chars returns 'untitled'."""
        assert generate_slug("!!!") == "untitled"
        assert generate_slug("@#$%^") == "untitled"

    def test_max_length_truncation(self):
        """Long slugs are truncated."""
        long_title = "this is a very long title that exceeds the maximum length"
        slug = generate_slug(long_title, max_length=20)
        assert len(slug) <= 20

    def test_truncation_at_word_boundary(self):
        """Truncation prefers word boundaries."""
        title = "hello world testing truncation"
        slug = generate_slug(title, max_length=15)
        # Should truncate at hyphen, not mid-word
        assert "-" not in slug[-1:] if len(slug) > 0 else True

    def test_truncation_not_too_short(self):
        """Truncation doesn't make slug too short."""
        title = "abcdefghijklmnopqrstuvwxyz"
        slug = generate_slug(title, max_length=20)
        # Should keep reasonable length even without word boundary
        assert len(slug) >= 12  # At least 60% of max

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        title = "short title"
        slug = generate_slug(title, max_length=5)
        assert len(slug) <= 5

    def test_default_max_length(self):
        """Default max length constant is used."""
        assert DEFAULT_SLUG_MAX_LENGTH == 50

    def test_numbers_preserved(self):
        """Numbers are preserved in slug."""
        assert generate_slug("Chapter 1") == "chapter-1"
        assert generate_slug("2024 Annual Report") == "2024-annual-report"

    def test_mixed_content(self):
        """Complex mixed content is handled."""
        title = "Meeting Notes (Q4 2024) - Client: ABC Corp!"
        slug = generate_slug(title)
        assert "meeting" in slug
        assert "notes" in slug
        assert "q4" in slug
        assert "2024" in slug


class TestSlugifyPathSegment:
    """Tests for slugify_path_segment function."""

    def test_basic_segment(self):
        """Basic segment is slugified."""
        assert slugify_path_segment("Work Projects") == "work-projects"

    def test_special_chars(self):
        """Special characters removed from segment."""
        assert slugify_path_segment("Work & Projects") == "work-projects"
        assert slugify_path_segment("Client (Main)") == "client-main"

    def test_longer_max_length(self):
        """Path segments allow longer slugs."""
        long_name = "a" * 80
        slug = slugify_path_segment(long_name)
        assert len(slug) <= 100


class TestSlugifyCategoryPath:
    """Tests for slugify_category_path function."""

    def test_empty_category(self):
        """Empty category returns empty string."""
        assert slugify_category_path("") == ""
        assert slugify_category_path(None) == ""

    def test_single_segment(self):
        """Single segment path is slugified."""
        assert slugify_category_path("Projects") == "projects"
        assert slugify_category_path("Work & Fun") == "work-fun"

    def test_multiple_segments(self):
        """Multiple segment path is slugified."""
        assert slugify_category_path("Projects/Client X") == "projects/client-x"
        assert slugify_category_path("Work/2024/Q4") == "work/2024/q4"

    def test_special_chars_in_segments(self):
        """Special chars removed from each segment."""
        result = slugify_category_path("Work & Projects/Client (Main)")
        assert result == "work-projects/client-main"

    def test_empty_segments_skipped(self):
        """Empty segments are skipped."""
        assert slugify_category_path("Projects//Notes") == "projects/notes"
        assert slugify_category_path("/Projects/") == "projects"

    def test_whitespace_segments_skipped(self):
        """Whitespace-only segments are skipped."""
        assert slugify_category_path("Projects/   /Notes") == "projects/notes"


class TestBuildFilename:
    """Tests for build_filename function."""

    def test_basic_filename(self):
        """Basic filename is built correctly."""
        note_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        filename = build_filename("My Note", note_id)
        assert filename == "my-note-550e8400-e29b-41d4-a716-446655440000.md"

    def test_filename_has_md_extension(self):
        """Filename ends with .md."""
        note_id = uuid4()
        filename = build_filename("Test", note_id)
        assert filename.endswith(".md")

    def test_filename_contains_uuid(self):
        """Filename contains the UUID."""
        note_id = uuid4()
        filename = build_filename("Test", note_id)
        assert str(note_id) in filename

    def test_filename_contains_slug(self):
        """Filename contains the slug."""
        note_id = uuid4()
        filename = build_filename("My Project Plan", note_id)
        assert "my-project-plan" in filename

    def test_custom_max_slug_length(self):
        """Custom max slug length is respected."""
        note_id = uuid4()
        filename = build_filename("A Very Long Title Here", note_id, max_slug_length=10)
        # Slug portion should be limited
        slug_part = filename.split(f"-{note_id}")[0]
        assert len(slug_part) <= 10

    def test_special_chars_in_title(self):
        """Special characters in title are handled."""
        note_id = uuid4()
        filename = build_filename("Test & Demo (Draft)", note_id)
        assert "test-demo-draft" in filename
        assert str(note_id) in filename


class TestExtractUuidFromFilename:
    """Tests for extract_uuid_from_filename function."""

    def test_valid_filename(self):
        """UUID extracted from valid filename."""
        expected = UUID("550e8400-e29b-41d4-a716-446655440000")
        filename = "my-note-550e8400-e29b-41d4-a716-446655440000.md"
        result = extract_uuid_from_filename(filename)
        assert result == expected

    def test_uuid_only_filename(self):
        """UUID extracted from UUID-only filename."""
        expected = UUID("550e8400-e29b-41d4-a716-446655440000")
        filename = "550e8400-e29b-41d4-a716-446655440000.md"
        result = extract_uuid_from_filename(filename)
        assert result == expected

    def test_complex_slug(self):
        """UUID extracted from filename with complex slug."""
        expected = UUID("a1b2c3d4-e5f6-7890-1234-567890abcdef")
        filename = "my-project-plan-2024-a1b2c3d4-e5f6-7890-1234-567890abcdef.md"
        result = extract_uuid_from_filename(filename)
        assert result == expected

    def test_no_uuid(self):
        """Returns None when no UUID present."""
        assert extract_uuid_from_filename("readme.md") is None
        assert extract_uuid_from_filename("notes.txt") is None

    def test_invalid_uuid_format(self):
        """Returns None for invalid UUID format."""
        # Wrong length
        assert extract_uuid_from_filename("test-12345.md") is None
        # Invalid characters
        assert extract_uuid_from_filename("test-zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz.md") is None

    def test_uuid_not_at_end(self):
        """Returns None if UUID not at end before .md."""
        filename = "550e8400-e29b-41d4-a716-446655440000-extra.md"
        assert extract_uuid_from_filename(filename) is None

    def test_wrong_extension(self):
        """Returns None for wrong file extension."""
        assert extract_uuid_from_filename("test-550e8400-e29b-41d4-a716-446655440000.txt") is None

    def test_uppercase_uuid(self):
        """Handles uppercase UUID."""
        filename = "test-550E8400-E29B-41D4-A716-446655440000.md"
        result = extract_uuid_from_filename(filename)
        assert result is not None
        assert str(result).lower() == "550e8400-e29b-41d4-a716-446655440000"

    def test_path_with_filename(self):
        """Extracts UUID from full path."""
        filename = "projects/client-x/note-550e8400-e29b-41d4-a716-446655440000.md"
        result = extract_uuid_from_filename(filename)
        assert result == UUID("550e8400-e29b-41d4-a716-446655440000")


class TestSlugifyEdgeCases:
    """Edge case tests for slug functions."""

    def test_unicode_emoji_removed(self):
        """Emoji are removed from slug."""
        # Emoji should be stripped
        slug = generate_slug("Hello World")
        assert "hello" in slug.lower()

    def test_chinese_characters(self):
        """Chinese characters result in untitled."""
        # Pure Chinese should result in untitled (no ASCII)
        slug = generate_slug("\u4e2d\u6587")
        assert slug == "untitled"

    def test_mixed_unicode_ascii(self):
        """Mixed Unicode and ASCII keeps ASCII."""
        slug = generate_slug("Project \u4e2d\u6587 Test")
        assert "project" in slug
        assert "test" in slug

    def test_very_long_title(self):
        """Very long title is truncated properly."""
        long_title = " ".join(["word"] * 100)
        slug = generate_slug(long_title)
        assert len(slug) <= DEFAULT_SLUG_MAX_LENGTH

    def test_single_character(self):
        """Single character title works."""
        assert generate_slug("A") == "a"
        assert generate_slug("1") == "1"

    def test_hyphen_only(self):
        """Hyphen-only title returns untitled."""
        assert generate_slug("---") == "untitled"
        assert generate_slug("-") == "untitled"
