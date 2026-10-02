"""Semantic chunking for markdown notes."""

import re
from typing import TYPE_CHECKING

from mcp_notes.models import NoteChunk
from mcp_notes.settings import settings
from mcp_notes.storage.parser import ParsedNote

if TYPE_CHECKING:
    from vector_core import EmbeddingClient


def chunk_note(parsed: ParsedNote, embedder: "EmbeddingClient | None" = None) -> list[NoteChunk]:
    """Cover the complete body with exact spans and repeated note context.

    Character limits include metadata. Model limits are checked by the shared
    splitter, including role formatting. Paragraph boundaries are preferred,
    but a long paragraph or line is split without dropping whitespace or tails.
    """
    body = parsed.body
    chunks: list[NoteChunk] = []
    for section_start, section_end, title in _source_sections(body):
        context = _build_chunk_content(parsed, "", None)
        if title and title != parsed.title:
            context += f"Section: {title}\n"
        budget = settings.max_chunk_chars - len(context)
        if budget <= 0:
            raise ValueError("Note metadata exceeds the configured chunk character limit")
        start = section_start
        while start < section_end or not chunks:
            end = min(start + budget, section_end)
            if end < section_end:
                for separator in ("\n\n", "\n", " "):
                    boundary = body.rfind(separator, start + budget // 2, end)
                    if boundary >= 0:
                        end = boundary + len(separator)
                        break
            text = body[start:end]
            if embedder is None or not text:
                spans = [(0, len(text), text)]
            else:
                spans = [
                    (span.start, span.end, span.text)
                    for span in embedder.split_text(text, role="document", context_prefix=context)
                ]
            for span_start, span_end, span_text in spans:
                source_start, source_end = start + span_start, start + span_end
                chunks.append(
                    NoteChunk(
                        note_id=parsed.id,
                        chunk_index=len(chunks),
                        content=context + span_text,
                        section_title=title,
                        start_line=body.count("\n", 0, source_start) + 1,
                        end_line=body.count("\n", 0, source_end) + 1,
                        start_char=source_start,
                        end_char=source_end,
                    )
                )
            if end == section_end:
                break
            overlap = min(settings.section_overlap_chars, (end - start) // 4)
            start = end - overlap
    return chunks


def _source_sections(body: str) -> list[tuple[int, int, str | None]]:
    """Partition original text at headings, retaining even header-only sections."""
    headings = list(re.finditer(r"(?m)^#{1,2}[ \t]+(.+)$", body))
    starts: list[tuple[int, str | None]] = [
        (match.start(), match.group(1).strip()) for match in headings
    ]
    if not starts or starts[0][0]:
        starts.insert(0, (0, None))
    return [
        (start, starts[index + 1][0] if index + 1 < len(starts) else len(body), title)
        for index, (start, title) in enumerate(starts)
    ]


def _build_chunk_content(
    parsed: ParsedNote,
    section_content: str,
    section_title: str | None,
) -> str:
    """
    Build chunk content with note context.

    Includes note title, tags, and category for semantic richness.
    """
    parts = []

    # Note title
    parts.append(f"# {parsed.title}")

    # Metadata context
    if parsed.tags:
        parts.append(f"Tags: {', '.join(parsed.tags)}")
    if parsed.category:
        parts.append(f"Category: {parsed.category}")

    parts.append("")  # Blank line

    # Section title if different from note title
    if section_title and section_title != parsed.title:
        parts.append(f"## {section_title}")

    # Content
    parts.append(section_content)

    return "\n".join(parts)


def generate_note_summary(parsed: ParsedNote) -> str:
    """
    Generate a summary string for file-level indexing.

    Includes title, tags, category, and first part of content.
    """
    parts = [parsed.title]

    if parsed.tags:
        parts.append(f"Tags: {', '.join(parsed.tags)}")
    if parsed.category:
        parts.append(f"Category: {parsed.category}")

    # Add first 500 chars of body
    excerpt = parsed.body[:500]
    if len(parsed.body) > 500:
        last_space = excerpt.rfind(" ")
        if last_space > 350:
            excerpt = excerpt[:last_space] + "..."
        else:
            excerpt += "..."

    parts.append(excerpt)

    return "\n".join(parts)
