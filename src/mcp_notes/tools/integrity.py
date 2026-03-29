"""Source integrity operations for facts.

Tools:
- get_facts_with_stale_sources: Get facts with stale sources
- get_source_statistics: Get statistics about fact source integrity
- check_fact_integrity: Check integrity of a specific fact's sources
- revalidate_fact_sources: Reset sources back to active after verification
"""

from uuid import UUID

from vector_core import validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.facts import SourceType
from mcp_notes.singletons import get_integrity_manager


@mcp.tool()
async def get_facts_with_stale_sources(
    status: str = "all",
    limit: int = 50,
) -> list[dict]:
    """
    Get facts with stale (deleted or modified) sources.

    Useful for identifying facts that may need review or re-verification.

    Args:
        status: Filter by source status: "deleted", "modified", or "all"
        limit: Max results (default 50, max 100)

    Returns:
        List of facts with stale sources
    """
    integrity = get_integrity_manager()
    limit = validate_limit(limit, default=50)

    facts = []

    if status in ("deleted", "all"):
        deleted_facts = integrity.get_facts_with_deleted_sources(limit=limit)
        facts.extend(deleted_facts)

    if status in ("modified", "all"):
        remaining = limit - len(facts) if status == "all" else limit
        modified_facts = integrity.get_facts_with_modified_sources(limit=remaining)
        facts.extend(modified_facts)

    # Deduplicate (a fact could have both deleted and modified sources)
    seen_ids = set()
    unique_facts = []
    for fact in facts:
        if fact.id not in seen_ids:
            seen_ids.add(fact.id)
            unique_facts.append(fact)

    return [f.to_dict() for f in unique_facts[:limit]]


@mcp.tool()
async def get_source_statistics() -> dict:
    """
    Get statistics about fact source integrity.

    Returns counts of sources by status across all facts.

    Returns:
        Dict with source counts by status and integrity score
    """
    integrity = get_integrity_manager()
    stats = integrity.get_source_statistics()

    # Extract values from the nested structure
    total = stats.get("total_sources", 0)
    by_status = stats.get("by_status", {})
    by_type = stats.get("by_type", {})

    # Calculate integrity score
    active = by_status.get("active", 0)
    integrity_score = active / total if total > 0 else 1.0

    return {
        "total_sources": total,
        "by_status": by_status,
        "by_type": by_type,
        "integrity_score": round(integrity_score, 4),
        "needs_attention": by_status.get("deleted", 0) + by_status.get("modified", 0),
    }


@mcp.tool()
async def check_fact_integrity(fact_id: str) -> dict:
    """
    Check integrity of a specific fact's sources.

    Returns breakdown of source statuses for the fact.

    Args:
        fact_id: Fact UUID string

    Returns:
        IntegrityCheckResult with source status breakdown
    """
    integrity = get_integrity_manager()

    try:
        uuid = UUID(fact_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {fact_id}")

    result = integrity.check_fact_integrity(uuid)
    return result.to_dict()


@mcp.tool()
async def revalidate_fact_sources(
    source_id: str | None = None,
    source_type: str | None = None,
) -> dict:
    """
    Reset modified/deleted sources back to active after re-verification.

    Use this after manually verifying that sources are still valid.

    Args:
        source_id: Optional source UUID to revalidate (e.g., a note or glossary entry UUID)
        source_type: Optional source type filter ("note", "glossary", "document")

    Returns:
        Count of sources revalidated
    """
    integrity = get_integrity_manager()

    uuid = None
    if source_id:
        try:
            uuid = UUID(source_id)
        except ValueError:
            return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {source_id}")

    src_type = None
    if source_type:
        try:
            src_type = SourceType(source_type)
        except ValueError:
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid source_type: {source_type}. "
                "Must be one of: note, glossary, document, manual"
            )

    count = integrity.revalidate_sources(
        source_type=src_type,
        source_id=uuid,
    )

    return {
        "revalidated_count": count,
        "source_type": source_type,
        "source_id": source_id,
    }
