"""Semantic chunking for markdown notes."""

import re

from mcp_notes.models import NoteChunk
from mcp_notes.settings import settings
from mcp_notes.storage.parser import ParsedNote


def chunk_note(parsed: ParsedNote) -> list[NoteChunk]:
    """
    Chunk a note into semantic units for indexing.

    Strategy:
    1. If note < max_chunk_chars, embed as single chunk
    2. Otherwise, split by H1/H2 headers into sections
    3. Each section includes note title + metadata for context

    Args:
        parsed: Parsed note

    Returns:
        List of NoteChunk objects
    """
    body = parsed.body
    total_chars = len(body)

    # Small notes: single chunk
    if total_chars <= settings.max_chunk_chars:
        return [
            NoteChunk(
                note_id=parsed.id,
                chunk_index=0,
                content=_build_chunk_content(parsed, body, None),
                section_title=None,
                start_line=1,
                end_line=body.count("\n") + 1,
            )
        ]

    # Large notes: split by headers
    sections = _split_by_headers(body)

    chunks = []
    for i, section in enumerate(sections):
        chunk_content = _build_chunk_content(
            parsed,
            section["content"],
            section["title"],
        )

        # If still too large, split further
        if len(chunk_content) > settings.max_chunk_chars:
            sub_chunks = _split_large_section(
                parsed,
                section,
                i,
                len(sections),
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                NoteChunk(
                    note_id=parsed.id,
                    chunk_index=i,
                    content=chunk_content,
                    section_title=section["title"],
                    start_line=section["start_line"],
                    end_line=section["end_line"],
                )
            )

    # Re-number chunks sequentially
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


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


def _split_by_headers(content: str) -> list[dict]:
    """
    Split markdown content by H1/H2 headers.

    Returns list of {"title", "content", "start_line", "end_line"}
    """
    lines = content.split("\n")
    sections = []
    current_section = None

    header_pattern = re.compile(r"^(#{1,2})\s+(.+)$")

    for i, line in enumerate(lines):
        match = header_pattern.match(line)

        if match:
            # Save previous section
            if current_section:
                current_section["end_line"] = i
                start = int(current_section["start_line"])
                current_section["content"] = "\n".join(
                    lines[start:i]
                ).strip()
                if current_section["content"]:
                    sections.append(current_section)

            # Start new section. start_line points at the line AFTER the
            # header so the header itself is excluded from the section body;
            # the chunk builder re-emits the title as a header, and keeping the
            # raw header here would duplicate it in the indexed chunk text.
            current_section = {
                "title": match.group(2).strip(),
                "start_line": i + 1,
                "end_line": len(lines),
                "content": "",
            }
        elif current_section is None:
            # Content before first header
            current_section = {
                "title": None,
                "start_line": 0,
                "end_line": len(lines),
                "content": "",
            }

    # Save last section
    if current_section:
        start = int(current_section["start_line"])
        current_section["content"] = "\n".join(
            lines[start:]
        ).strip()
        if current_section["content"]:
            sections.append(current_section)

    # If no headers found, return whole content as one section
    if not sections:
        sections.append({
            "title": None,
            "content": content.strip(),
            "start_line": 1,
            "end_line": len(lines),
        })

    return sections


def _split_large_section(
    parsed: ParsedNote,
    section: dict,
    section_index: int,
    total_sections: int,
) -> list[NoteChunk]:
    """
    Split a large section into smaller chunks.

    Uses paragraph-based splitting with overlap.
    """
    content = section["content"]
    max_chars = settings.max_chunk_chars
    overlap = settings.section_overlap_chars

    # Split by paragraphs (double newline)
    paragraphs = re.split(r"\n\n+", content)

    chunks: list[NoteChunk] = []
    current_chunk: list[str] = []
    current_len = 0
    chunk_start = section["start_line"]

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > max_chars and current_chunk:
            # Save current chunk
            chunk_content = _build_chunk_content(
                parsed,
                "\n\n".join(current_chunk),
                section["title"],
            )

            chunks.append(
                NoteChunk(
                    note_id=parsed.id,
                    chunk_index=len(chunks),
                    content=chunk_content,
                    section_title=section["title"],
                    start_line=chunk_start,
                    end_line=chunk_start + sum(c.count("\n") for c in current_chunk),
                )
            )

            # Start new chunk with overlap
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break

            current_chunk = overlap_paras
            current_len = overlap_len
            chunk_lines = sum(c.count("\n") for c in current_chunk)
            chunk_start = chunk_start + chunk_lines - len(overlap_paras)

        current_chunk.append(para)
        current_len += para_len

    # Save final chunk
    if current_chunk:
        chunk_content = _build_chunk_content(
            parsed,
            "\n\n".join(current_chunk),
            section["title"],
        )

        chunks.append(
            NoteChunk(
                note_id=parsed.id,
                chunk_index=len(chunks),
                content=chunk_content,
                section_title=section["title"],
                start_line=chunk_start,
                end_line=section["end_line"],
            )
        )

    return chunks


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
