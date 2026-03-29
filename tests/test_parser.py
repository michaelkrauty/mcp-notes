"""Tests for note parser (YAML frontmatter parsing)."""

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from mcp_notes.storage.parser import (
    FrontmatterTooLargeError,
    MAX_FRONTMATTER_SIZE,
    extract_inline_links,
    parse_note,
    serialize_note,
)


class TestParseNote:
    """Tests for parse_note function."""

    def test_valid_minimal_note(self):
        """Parse note with minimal required fields."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test Note
created: 2024-01-15T10:00:00+00:00
modified: 2024-01-15T10:00:00+00:00
---

This is the body."""

        parsed = parse_note(content)

        assert parsed.id == note_id
        assert parsed.title == "Test Note"
        assert parsed.body == "This is the body."
        assert parsed.tags == []
        assert parsed.category is None
        assert parsed.links == []

    def test_valid_full_note(self):
        """Parse note with all fields populated."""
        note_id = uuid4()
        link_id = uuid4()
        content = f"""---
id: {note_id}
title: Full Note
created: 2024-01-15T10:00:00Z
modified: 2024-01-16T15:30:00Z
tags:
  - python
  - testing
category: projects/mcp
links:
  - {link_id}
---

Full body content here."""

        parsed = parse_note(content)

        assert parsed.id == note_id
        assert parsed.title == "Full Note"
        assert parsed.tags == ["python", "testing"]
        assert parsed.category == "projects/mcp"
        assert parsed.links == [link_id]
        assert parsed.body == "Full body content here."

    def test_missing_frontmatter_raises(self):
        """Missing frontmatter raises ValueError."""
        content = "Just body content without frontmatter"

        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parse_note(content)

    def test_invalid_yaml_raises(self):
        """Invalid YAML in frontmatter raises ValueError."""
        content = """---
id: test
title: [invalid yaml
---

Body"""

        with pytest.raises(ValueError, match="Invalid YAML"):
            parse_note(content)

    def test_missing_required_id_raises(self):
        """Missing id field raises ValueError."""
        content = """---
title: No ID
created: 2024-01-15
modified: 2024-01-15
---

Body"""

        with pytest.raises(ValueError, match="missing required field: id"):
            parse_note(content)

    def test_missing_required_title_raises(self):
        """Missing title field raises ValueError."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
created: 2024-01-15
modified: 2024-01-15
---

Body"""

        with pytest.raises(ValueError, match="missing required field: title"):
            parse_note(content)

    def test_missing_required_created_raises(self):
        """Missing created field raises ValueError."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
modified: 2024-01-15
---

Body"""

        with pytest.raises(ValueError, match="missing required field: created"):
            parse_note(content)

    def test_missing_required_modified_raises(self):
        """Missing modified field raises ValueError."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15
---

Body"""

        with pytest.raises(ValueError, match="missing required field: modified"):
            parse_note(content)

    def test_invalid_uuid_raises(self):
        """Invalid UUID in id field raises ValueError."""
        content = """---
id: not-a-uuid
title: Test
created: 2024-01-15
modified: 2024-01-15
---

Body"""

        with pytest.raises(ValueError, match="Invalid UUID"):
            parse_note(content)

    def test_single_tag_converted_to_list(self):
        """Single tag value is converted to list."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
tags: single-tag
---

Body"""

        parsed = parse_note(content)
        assert parsed.tags == ["single-tag"]

    def test_tags_normalized_to_lowercase(self):
        """Tags are normalized to lowercase."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
tags:
  - UPPERCASE
  - MixedCase
---

Body"""

        parsed = parse_note(content)
        assert parsed.tags == ["uppercase", "mixedcase"]

    def test_invalid_link_uuids_skipped(self):
        """Invalid UUIDs in links are skipped."""
        note_id = uuid4()
        valid_link = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
links:
  - {valid_link}
  - not-a-uuid
  - also-invalid
---

Body"""

        parsed = parse_note(content)
        assert parsed.links == [valid_link]

    def test_date_formats_iso(self):
        """ISO date format is parsed correctly."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T10:30:00+00:00
modified: 2024-01-15T10:30:00Z
---

Body"""

        parsed = parse_note(content)
        assert parsed.created.year == 2024
        assert parsed.created.month == 1
        assert parsed.created.day == 15
        assert parsed.created.tzinfo is not None

    def test_date_formats_simple(self):
        """Simple date format (as string) is parsed with UTC timezone."""
        note_id = uuid4()
        # Use datetime string format since YAML converts 2024-01-15 to date object
        content = f"""---
id: {note_id}
title: Test
created: "2024-01-15"
modified: "2024-01-15"
---

Body"""

        parsed = parse_note(content)
        assert parsed.created.tzinfo == UTC

    def test_empty_body_handled(self):
        """Note with empty body is handled."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
---

"""

        parsed = parse_note(content)
        assert parsed.body == ""

    def test_multiline_body_preserved(self):
        """Multiline body content is preserved."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
---

Line 1
Line 2

Line 4 after blank"""

        parsed = parse_note(content)
        assert "Line 1" in parsed.body
        assert "Line 2" in parsed.body
        assert "Line 4 after blank" in parsed.body


class TestSerializeNote:
    """Tests for serialize_note function."""

    def test_minimal_serialize(self):
        """Serialize note with minimal fields."""
        note_id = uuid4()
        result = serialize_note(
            note_id=note_id,
            title="Test Note",
            body="The body content",
        )

        assert f"id: {note_id}" in result
        assert "title: Test Note" in result
        assert "created:" in result
        assert "modified:" in result
        assert "---" in result
        assert "The body content" in result

    def test_full_serialize(self):
        """Serialize note with all fields."""
        note_id = uuid4()
        link_id = uuid4()
        created = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        modified = datetime(2024, 1, 16, 15, 30, 0, tzinfo=UTC)

        result = serialize_note(
            note_id=note_id,
            title="Full Note",
            body="Body here",
            tags=["python", "test"],
            category="projects/mcp",  # Ignored - category is derived from path
            links=[link_id],
            created=created,
            modified=modified,
        )

        assert f"id: {note_id}" in result
        assert "title: Full Note" in result
        assert "tags:" in result
        assert "python" in result
        # Note: category is NOT stored in frontmatter - it's derived from path
        assert "category:" not in result
        assert str(link_id) in result

    def test_roundtrip(self):
        """Serialize then parse produces same data (except category)."""
        note_id = uuid4()
        link_id = uuid4()
        created = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        modified = datetime(2024, 1, 16, 15, 30, 0, tzinfo=UTC)

        serialized = serialize_note(
            note_id=note_id,
            title="Roundtrip Test",
            body="Body content here",
            tags=["tag1", "tag2"],
            category="test/category",  # Ignored - category is derived from path
            links=[link_id],
            created=created,
            modified=modified,
        )

        parsed = parse_note(serialized)

        assert parsed.id == note_id
        assert parsed.title == "Roundtrip Test"
        assert parsed.body == "Body content here"
        assert set(parsed.tags) == {"tag1", "tag2"}
        # Note: category is NOT stored in frontmatter - it's derived from path
        assert parsed.category is None
        assert parsed.links == [link_id]

    def test_no_optional_fields(self):
        """Omitting optional fields doesn't include them."""
        note_id = uuid4()
        result = serialize_note(
            note_id=note_id,
            title="Minimal",
            body="Body",
        )

        assert "tags:" not in result
        assert "category:" not in result
        assert "links:" not in result


class TestExtractInlineLinks:
    """Tests for extract_inline_links function."""

    def test_single_link(self):
        """Extract single [[uuid]] link."""
        link_id = uuid4()
        content = f"Some text with [[{link_id}]] in it."

        links = extract_inline_links(content)

        assert links == [link_id]

    def test_multiple_links(self):
        """Extract multiple [[uuid]] links."""
        link1 = uuid4()
        link2 = uuid4()
        content = f"Links: [[{link1}]] and [[{link2}]]"

        links = extract_inline_links(content)

        assert len(links) == 2
        assert link1 in links
        assert link2 in links

    def test_no_links(self):
        """Return empty list when no links."""
        content = "Text without any links."

        links = extract_inline_links(content)

        assert links == []

    def test_invalid_uuid_ignored(self):
        """Invalid UUIDs in brackets are ignored."""
        valid_id = uuid4()
        content = f"[[not-uuid]] and [[{valid_id}]] and [[12345]]"

        links = extract_inline_links(content)

        assert links == [valid_id]

    def test_duplicate_links_kept(self):
        """Duplicate links are kept in result."""
        link_id = uuid4()
        content = f"[[{link_id}]] appears [[{link_id}]] twice"

        links = extract_inline_links(content)

        assert len(links) == 2
        assert all(link == link_id for link in links)

    def test_links_in_multiline_content(self):
        """Links are extracted from multiline content."""
        link1 = uuid4()
        link2 = uuid4()
        content = f"""First paragraph with [[{link1}]]

Second paragraph.

Third with [[{link2}]]"""

        links = extract_inline_links(content)

        assert len(links) == 2
        assert link1 in links
        assert link2 in links

    def test_case_insensitive_uuid(self):
        """UUID matching is case-insensitive."""
        # UUIDs with uppercase hex characters
        content = "[[A1B2C3D4-E5F6-7890-1234-567890ABCDEF]]"

        links = extract_inline_links(content)

        assert len(links) == 1

    def test_invalid_format_uuid_in_brackets(self):
        """Invalid format UUIDs in brackets are skipped (triggers line 225-226)."""
        # 36 chars of valid hex+hyphen that matches regex but wrong UUID format
        # UUID format is 8-4-4-4-12, but this is 4-4-4-4-4-4-4-4-4 (wrong structure)
        content = "[[0000-0000-0000-0000-0000-0000-0000-0]]"

        links = extract_inline_links(content)

        # Should skip invalid UUID gracefully
        assert links == []


class TestParserEdgeCases:
    """Tests for parser edge cases and error paths."""

    def test_frontmatter_not_dict(self):
        """Frontmatter that parses to non-dict raises ValueError (line 61)."""
        # YAML that parses to a string, not a dict
        content = """---
just a string here
---

Body"""

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse_note(content)

    def test_links_single_value_not_list(self):
        """Single link value (not list) is converted to list (line 96)."""
        note_id = uuid4()
        link_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
links: {link_id}
---

Body"""

        parsed = parse_note(content)
        assert parsed.links == [link_id]

    def test_datetime_without_tzinfo(self):
        """Datetime object without tzinfo gets UTC added (line 123)."""
        from datetime import datetime

        from mcp_notes.storage.parser import _parse_datetime

        # Create naive datetime (no timezone)
        naive_dt = datetime(2024, 1, 15, 10, 30, 0)

        result = _parse_datetime(naive_dt)

        assert result.tzinfo == UTC
        assert result.year == 2024
        assert result.month == 1

    def test_datetime_with_tzinfo_preserved(self):
        """Datetime with tzinfo is preserved."""
        from datetime import datetime, timedelta

        from mcp_notes.storage.parser import _parse_datetime

        # Create aware datetime with specific timezone
        tz = timezone(timedelta(hours=5))
        aware_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=tz)

        result = _parse_datetime(aware_dt)

        assert result.tzinfo is not None
        assert result == aware_dt

    def test_datetime_format_with_time(self):
        """Parse date-time format 'YYYY-MM-DD HH:MM:SS' (line 143-145)."""
        from mcp_notes.storage.parser import _parse_datetime

        result = _parse_datetime("2024-01-15 10:30:00")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == UTC

    def test_datetime_format_slash_with_time(self):
        """Parse date-time format 'YYYY/MM/DD HH:MM:SS' (line 143-145)."""
        from mcp_notes.storage.parser import _parse_datetime

        result = _parse_datetime("2024/01/15 10:30:00")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_datetime_format_slash_date_only(self):
        """Parse date-only format 'YYYY/MM/DD' (line 143-145)."""
        from mcp_notes.storage.parser import _parse_datetime

        result = _parse_datetime("2024/01/15")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == UTC

    def test_datetime_invalid_format_raises(self):
        """Invalid datetime format raises ValueError (line 149)."""
        from mcp_notes.storage.parser import _parse_datetime

        with pytest.raises(ValueError, match="Cannot parse datetime"):
            _parse_datetime("not-a-date-at-all")

    def test_datetime_iso_string_without_tzinfo(self):
        """ISO string without timezone gets UTC (line 130-132)."""
        from mcp_notes.storage.parser import _parse_datetime

        result = _parse_datetime("2024-01-15T10:30:00")

        assert result.tzinfo == UTC
        assert result.hour == 10

    def test_tags_single_value_not_list(self):
        """Single tag value (not list) is converted to list (line 86)."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
tags: single-tag
---

Body"""

        parsed = parse_note(content)
        assert parsed.tags == ["single-tag"]

    def test_links_empty_falsy_value(self):
        """Empty/falsy links value handled (line 96 else branch)."""
        note_id = uuid4()
        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
links: ""
---

Body"""

        parsed = parse_note(content)
        assert parsed.links == []


class TestFrontmatterSizeLimits:
    """Tests for frontmatter size limit protection."""

    def test_frontmatter_at_limit_succeeds(self):
        """Frontmatter exactly at limit should parse successfully."""
        note_id = uuid4()
        # Create frontmatter close to but under limit
        # Header is about 150 bytes, so add padding
        padding_size = MAX_FRONTMATTER_SIZE - 200
        padding = "x" * padding_size

        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
description: "{padding}"
---

Body"""

        # Should not raise - frontmatter is at/under limit
        parsed = parse_note(content)
        assert parsed.id == note_id

    def test_frontmatter_over_limit_raises(self):
        """Frontmatter over size limit raises FrontmatterTooLargeError."""
        note_id = uuid4()
        # Create frontmatter that exceeds limit
        padding_size = MAX_FRONTMATTER_SIZE + 1000
        padding = "x" * padding_size

        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
description: "{padding}"
---

Body"""

        with pytest.raises(FrontmatterTooLargeError) as exc_info:
            parse_note(content)

        # Verify error contains useful information
        assert exc_info.value.max_size == MAX_FRONTMATTER_SIZE
        assert exc_info.value.size > MAX_FRONTMATTER_SIZE
        assert "too large" in str(exc_info.value).lower()

    def test_frontmatter_too_large_error_attributes(self):
        """FrontmatterTooLargeError has correct attributes."""
        error = FrontmatterTooLargeError(size=100000, max_size=65536)

        assert error.size == 100000
        assert error.max_size == 65536
        assert "100,000" in str(error)  # Formatted with commas
        assert "65,536" in str(error)

    def test_max_frontmatter_size_constant(self):
        """MAX_FRONTMATTER_SIZE is 64KB."""
        assert MAX_FRONTMATTER_SIZE == 64 * 1024
        assert MAX_FRONTMATTER_SIZE == 65536

    def test_yaml_bomb_prevented(self):
        """YAML billion laughs attack is prevented by size limit."""
        # Attempt a YAML "billion laughs" style attack
        # This would expand to a huge amount of memory without the size limit
        note_id = uuid4()

        # Create nested YAML anchors that would expand exponentially
        yaml_bomb = """
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h]
"""
        # Make it exceed the size limit by repeating
        huge_yaml = yaml_bomb * (MAX_FRONTMATTER_SIZE // len(yaml_bomb) + 1)

        content = f"""---
id: {note_id}
title: Test
created: 2024-01-15T00:00:00Z
modified: 2024-01-15T00:00:00Z
{huge_yaml}
---

Body"""

        # Should be blocked by size limit before YAML parsing
        with pytest.raises(FrontmatterTooLargeError):
            parse_note(content)
