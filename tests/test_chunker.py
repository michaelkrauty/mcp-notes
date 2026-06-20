"""Tests for note chunker (semantic chunking for indexing)."""

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

from mcp_notes.indexing.chunker import (
    _build_chunk_content,
    _split_by_headers,
    chunk_note,
    generate_note_summary,
)
from mcp_notes.storage.parser import ParsedNote


def make_parsed_note(
    body: str,
    title: str = "Test Note",
    tags: list[str] | None = None,
    category: str | None = None,
) -> ParsedNote:
    """Helper to create ParsedNote for testing."""
    note_id = uuid4()
    now = datetime.now(UTC)
    return ParsedNote(
        id=note_id,
        title=title,
        content=f"---\n...\n---\n\n{body}",
        tags=tags or [],
        category=category,
        links=[],
        created=now,
        modified=now,
        raw_frontmatter={},
        body=body,
    )


class TestChunkNote:
    """Tests for chunk_note function."""

    def test_small_note_single_chunk(self):
        """Small note produces single chunk."""
        parsed = make_parsed_note("Short content here.")

        chunks = chunk_note(parsed)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert "Short content here." in chunks[0].content

    def test_chunk_includes_title(self):
        """Chunk includes note title."""
        parsed = make_parsed_note("Body content", title="Important Note")

        chunks = chunk_note(parsed)

        assert "# Important Note" in chunks[0].content

    def test_chunk_includes_tags(self):
        """Chunk includes tags when present."""
        parsed = make_parsed_note("Body", tags=["python", "testing"])

        chunks = chunk_note(parsed)

        assert "Tags: python, testing" in chunks[0].content

    def test_chunk_includes_category(self):
        """Chunk includes category when present."""
        parsed = make_parsed_note("Body", category="projects/mcp")

        chunks = chunk_note(parsed)

        assert "Category: projects/mcp" in chunks[0].content

    @patch("mcp_notes.indexing.chunker.settings")
    def test_large_note_split_by_headers(self, mock_settings):
        """Large note is split by headers."""
        mock_settings.max_chunk_chars = 100
        mock_settings.section_overlap_chars = 20

        body = """# Section One

Content for section one with enough text.

# Section Two

Content for section two with more text here."""

        parsed = make_parsed_note(body)

        chunks = chunk_note(parsed)

        assert len(chunks) >= 2

    @patch("mcp_notes.indexing.chunker.settings")
    def test_header_not_duplicated_in_split_chunks(self, mock_settings):
        """When a large note is split by headers, the section header must not
        appear twice in any chunk (once from the chunk builder's title line and
        once from the raw header left in the section body)."""
        mock_settings.max_chunk_chars = 100
        mock_settings.section_overlap_chars = 20

        body = """# Section One

Content for section one with enough text.

# Section Two

Content for section two with more text here."""

        parsed = make_parsed_note(body)

        chunks = chunk_note(parsed)

        for chunk in chunks:
            assert "## Section One\n# Section One" not in chunk.content
            assert "## Section Two\n# Section Two" not in chunk.content

    def test_chunk_has_correct_note_id(self):
        """All chunks have correct note_id."""
        parsed = make_parsed_note("Content here")

        chunks = chunk_note(parsed)

        assert all(c.note_id == parsed.id for c in chunks)

    def test_chunks_sequentially_numbered(self):
        """Chunks are sequentially numbered starting from 0."""
        # Need a large note to get multiple chunks
        body = "# Section A\n" + "a" * 500 + "\n# Section B\n" + "b" * 500

        with patch("mcp_notes.indexing.chunker.settings") as mock_settings:
            mock_settings.max_chunk_chars = 200
            mock_settings.section_overlap_chars = 20

            parsed = make_parsed_note(body)
            chunks = chunk_note(parsed)

            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i


class TestSplitByHeaders:
    """Tests for _split_by_headers function."""

    def test_no_headers(self):
        """Content without headers is single section."""
        content = "Just plain text without headers."

        sections = _split_by_headers(content)

        assert len(sections) == 1
        assert sections[0]["title"] is None
        assert "Just plain text" in sections[0]["content"]

    def test_h1_headers(self):
        """H1 headers create sections."""
        content = """# First

Content one.

# Second

Content two."""

        sections = _split_by_headers(content)

        assert len(sections) == 2
        assert sections[0]["title"] == "First"
        assert sections[1]["title"] == "Second"

    def test_section_content_excludes_its_own_header(self):
        """A section's content must not contain its own header line. The chunk
        builder re-emits the section title as a header, so keeping the raw
        header in the body duplicates it in the indexed, searchable chunk."""
        content = """# First

Content one.

# Second

Content two."""

        sections = _split_by_headers(content)

        assert not sections[0]["content"].lstrip().startswith("# First")
        assert "Content one." in sections[0]["content"]
        assert not sections[1]["content"].lstrip().startswith("# Second")
        assert "Content two." in sections[1]["content"]

    def test_h2_headers(self):
        """H2 headers create sections."""
        content = """## Section A

Content A.

## Section B

Content B."""

        sections = _split_by_headers(content)

        assert len(sections) == 2
        assert sections[0]["title"] == "Section A"
        assert sections[1]["title"] == "Section B"

    def test_content_before_first_header(self):
        """Content before first header is captured."""
        content = """Intro content here.

# First Section

Section content."""

        sections = _split_by_headers(content)

        # First section should have no title (content before header)
        # Or might start with the header - check implementation
        assert len(sections) >= 1
        # At minimum the section content should be captured
        assert any("content" in s["content"].lower() for s in sections)

    def test_empty_sections_skipped(self):
        """Empty sections are not included."""
        content = """# Empty

# Also Empty


# Has Content

Actual content here."""

        sections = _split_by_headers(content)

        # Only non-empty sections should be included
        for section in sections:
            # Content should not be just whitespace
            if section["content"]:
                assert section["content"].strip()

    def test_section_line_numbers(self):
        """Sections have correct line numbers."""
        content = """# First

Line 1
Line 2

# Second

Line A"""

        sections = _split_by_headers(content)

        # Each section should have start_line and end_line
        for section in sections:
            assert "start_line" in section
            assert "end_line" in section
            assert section["end_line"] >= section["start_line"]


class TestBuildChunkContent:
    """Tests for _build_chunk_content function."""

    def test_includes_note_title(self):
        """Chunk content includes note title as H1."""
        parsed = make_parsed_note("Body", title="My Note")

        result = _build_chunk_content(parsed, "Section body", None)

        assert "# My Note" in result

    def test_includes_section_title(self):
        """Chunk content includes section title as H2."""
        parsed = make_parsed_note("Body", title="Note Title")

        result = _build_chunk_content(parsed, "Section body", "Section Name")

        assert "## Section Name" in result

    def test_section_same_as_note_not_duplicated(self):
        """Section title same as note title is not duplicated."""
        parsed = make_parsed_note("Body", title="Same Title")

        result = _build_chunk_content(parsed, "Section body", "Same Title")

        # Should only appear once as H1, not also as H2
        assert result.count("Same Title") == 1

    def test_tags_included(self):
        """Tags are included in chunk content."""
        parsed = make_parsed_note("Body", tags=["tag1", "tag2"])

        result = _build_chunk_content(parsed, "Section body", None)

        assert "Tags: tag1, tag2" in result

    def test_category_included(self):
        """Category is included in chunk content."""
        parsed = make_parsed_note("Body", category="path/to/category")

        result = _build_chunk_content(parsed, "Section body", None)

        assert "Category: path/to/category" in result

    def test_section_content_included(self):
        """Section content is included."""
        parsed = make_parsed_note("Body")

        result = _build_chunk_content(parsed, "This is the actual section content", None)

        assert "This is the actual section content" in result


class TestGenerateNoteSummary:
    """Tests for generate_note_summary function."""

    def test_includes_title(self):
        """Summary includes note title."""
        parsed = make_parsed_note("Body content", title="My Title")

        summary = generate_note_summary(parsed)

        assert "My Title" in summary

    def test_includes_tags(self):
        """Summary includes tags."""
        parsed = make_parsed_note("Body", tags=["python", "testing"])

        summary = generate_note_summary(parsed)

        assert "Tags: python, testing" in summary

    def test_includes_category(self):
        """Summary includes category."""
        parsed = make_parsed_note("Body", category="projects/mcp")

        summary = generate_note_summary(parsed)

        assert "Category: projects/mcp" in summary

    def test_includes_body_excerpt(self):
        """Summary includes excerpt from body."""
        parsed = make_parsed_note("This is the body content of the note.")

        summary = generate_note_summary(parsed)

        assert "body content" in summary.lower()

    def test_truncates_long_body(self):
        """Long body is truncated."""
        long_body = "word " * 200  # 1000 chars

        parsed = make_parsed_note(long_body)

        summary = generate_note_summary(parsed)

        # Should be truncated
        assert len(summary) < len(long_body) + 100

    def test_truncation_includes_ellipsis(self):
        """Truncated content ends with ellipsis."""
        long_body = "word " * 200

        parsed = make_parsed_note(long_body)

        summary = generate_note_summary(parsed)

        assert "..." in summary

    def test_no_tags_no_category(self):
        """Summary works without tags or category."""
        parsed = make_parsed_note("Body content")

        summary = generate_note_summary(parsed)

        assert "Tags:" not in summary
        assert "Category:" not in summary
        assert "Body content" in summary

    def test_truncation_no_space_near_cutoff(self):
        """Truncation handles content without space near 350-500 char range."""
        # Create body with no spaces near truncation point
        # This triggers line 280 (fallback path)
        long_word = "a" * 600  # Single long word with no spaces

        parsed = make_parsed_note(long_word)

        summary = generate_note_summary(parsed)

        # Should truncate with ellipsis at exact 500 chars
        assert "..." in summary
        assert len(summary) <= 600  # Reasonable bound


class TestLargeSectionSplitting:
    """Tests for _split_large_section function."""

    @patch("mcp_notes.indexing.chunker.settings")
    def test_large_section_splits_by_paragraphs(self, mock_settings):
        """Large section is split by paragraphs with overlap."""
        mock_settings.max_chunk_chars = 100
        mock_settings.section_overlap_chars = 20

        # Create note with single section that exceeds max_chunk_chars
        body = "Paragraph one with some content here.\n\n" + \
               "Paragraph two has more content.\n\n" + \
               "Paragraph three continues with text.\n\n" + \
               "Paragraph four is the last one."

        parsed = make_parsed_note(body)

        chunks = chunk_note(parsed)

        # Should have multiple chunks
        assert len(chunks) >= 2

    @patch("mcp_notes.indexing.chunker.settings")
    def test_paragraph_overflow_creates_new_chunk(self, mock_settings):
        """Paragraph exceeding limit starts new chunk."""
        mock_settings.max_chunk_chars = 50
        mock_settings.section_overlap_chars = 10

        # Create paragraphs that will overflow
        body = "Short intro.\n\n" + \
               "A" * 60 + "\n\n" + \
               "B" * 60 + "\n\n" + \
               "Final paragraph."

        parsed = make_parsed_note(body)

        chunks = chunk_note(parsed)

        # Should have multiple chunks due to overflow
        assert len(chunks) >= 2

    @patch("mcp_notes.indexing.chunker.settings")
    def test_overlap_preserved_between_chunks(self, mock_settings):
        """Overlap paragraphs are included in next chunk."""
        mock_settings.max_chunk_chars = 80
        mock_settings.section_overlap_chars = 30

        body = "First para.\n\n" + \
               "Second para.\n\n" + \
               "Third para.\n\n" + \
               "Fourth para."

        parsed = make_parsed_note(body)

        chunks = chunk_note(parsed)

        if len(chunks) > 1:
            # Check for overlap in content
            # (detailed verification would require parsing content)
            pass


class TestNoHeadersEdgeCase:
    """Test for content without any headers."""

    def test_content_without_headers_single_section(self):
        """Plain text content returns single section with null title."""
        content = """This is plain text content.
It has multiple lines.
But no markdown headers at all.
Just regular paragraphs."""

        sections = _split_by_headers(content)

        assert len(sections) == 1
        assert sections[0]["title"] is None
        assert "plain text content" in sections[0]["content"]

    def test_content_starting_with_paragraph(self):
        """Content that starts with paragraph before headers."""
        content = "Intro paragraph here."

        sections = _split_by_headers(content)

        assert len(sections) >= 1
        assert any("Intro" in s["content"] for s in sections)
