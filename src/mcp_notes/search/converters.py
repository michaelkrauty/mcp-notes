"""Payload-to-SearchResult converters for different content types.

Extracts the repetitive conversion logic from the search engine into
clean, testable functions.
"""

from uuid import UUID

from vector_core import parse_payload_timestamps

from mcp_notes.models import NoteSummary, SearchResult
from mcp_notes.settings import settings


def _safe_parse_uuid(value: str | None) -> UUID | None:
    """Safely parse a UUID string, returning None on failure."""
    if not value:
        return None
    try:
        return UUID(value)
    except (ValueError, TypeError):
        return None


def convert_note_payload(
    payload: dict,
    score: float,
    highlights: list[str] | None = None,
    degraded: bool = False,
) -> SearchResult | None:
    """
    Convert a note/chunk payload to SearchResult.

    Args:
        payload: Qdrant point payload
        score: Search score
        highlights: Optional highlight snippets
        degraded: True if using sparse-only fallback (embedding service unavailable)

    Returns:
        SearchResult or None if conversion fails
    """
    note_id = _safe_parse_uuid(payload.get("note_id"))
    if note_id is None:
        return None

    created, modified = parse_payload_timestamps(payload)
    point_type = payload.get("type", "note")
    content = payload.get("content", "")

    summary = NoteSummary(
        id=note_id,
        title=payload.get("title", ""),
        tags=payload.get("tags", []),
        category=payload.get("category"),
        created=created,
        modified=modified,
        excerpt=content[:settings.excerpt_length] if content else "",
    )

    return SearchResult(
        note=summary,
        score=score,
        highlights=highlights or [],
        result_type=point_type,
        degraded=degraded,
    )


def convert_glossary_payload(
    payload: dict,
    score: float,
    highlights: list[str] | None = None,
    degraded: bool = False,
) -> SearchResult | None:
    """
    Convert a glossary payload to SearchResult.

    Args:
        payload: Qdrant point payload
        score: Search score
        highlights: Optional highlight snippets
        degraded: True if using sparse-only fallback (embedding service unavailable)

    Returns:
        SearchResult or None if conversion fails
    """
    glossary_id = _safe_parse_uuid(payload.get("glossary_id"))
    if glossary_id is None:
        return None

    created, modified = parse_payload_timestamps(payload)

    # Build display fields
    term = payload.get("term", "")
    expansion = payload.get("expansion", "")
    definition = payload.get("definition", "")
    domain_val = payload.get("domain")

    excerpt = f"{term}: {expansion}. {definition}"[:settings.excerpt_length]

    summary = NoteSummary(
        id=glossary_id,
        title=f"[Glossary] {term}",
        tags=[domain_val] if isinstance(domain_val, str) else [],
        category=None,
        created=created,
        modified=modified,
        excerpt=excerpt,
    )

    return SearchResult(
        note=summary,
        score=score,
        highlights=highlights or [],
        result_type="glossary",
        degraded=degraded,
    )


def convert_fact_payload(
    payload: dict,
    score: float,
    highlights: list[str] | None = None,
    degraded: bool = False,
) -> SearchResult | None:
    """
    Convert a fact payload to SearchResult.

    Args:
        payload: Qdrant point payload
        score: Search score
        highlights: Optional highlight snippets
        degraded: True if using sparse-only fallback (embedding service unavailable)

    Returns:
        SearchResult or None if conversion fails
    """
    fact_id = _safe_parse_uuid(payload.get("fact_id"))
    if fact_id is None:
        return None

    created, modified = parse_payload_timestamps(payload)

    # Build fact display
    subject = payload.get("subject", "")
    predicate = payload.get("predicate", "").replace("_", " ")
    object_val = payload.get("object", "")
    context = payload.get("context", "")

    title = f"[Fact] {subject} {predicate} {object_val}"
    excerpt = f"{subject} {predicate} {object_val}"
    if context:
        excerpt = f"{excerpt} ({context})"
    excerpt = excerpt[:settings.excerpt_length]

    # Use entity types as pseudo-tags
    fact_tags = []
    if payload.get("subject_type"):
        fact_tags.append(payload["subject_type"])
    if payload.get("object_type"):
        fact_tags.append(payload["object_type"])

    summary = NoteSummary(
        id=fact_id,
        title=title,
        tags=fact_tags,
        category=None,
        created=created,
        modified=modified,
        excerpt=excerpt,
    )

    return SearchResult(
        note=summary,
        score=score,
        highlights=highlights or [],
        result_type="fact",
        degraded=degraded,
    )


def convert_payload(
    payload: dict,
    score: float,
    highlights: list[str] | None = None,
    degraded: bool = False,
) -> SearchResult | None:
    """
    Convert any payload type to SearchResult.

    Dispatches to the appropriate converter based on payload "type" field.

    Args:
        payload: Qdrant point payload
        score: Search score
        highlights: Optional highlight snippets
        degraded: True if using sparse-only fallback (embedding service unavailable)

    Returns:
        SearchResult or None if conversion fails
    """
    point_type = payload.get("type", "note")

    if point_type == "glossary":
        return convert_glossary_payload(payload, score, highlights, degraded)
    elif point_type == "fact":
        return convert_fact_payload(payload, score, highlights, degraded)
    else:
        # note or chunk
        return convert_note_payload(payload, score, highlights, degraded)
