"""Glossary management operations.

Tools:
- add_glossary_entry: Add a new glossary entry
- lookup_term: Exact lookup by term or alias
- search_glossary: Semantic search for glossary entries
- list_glossary: List all glossary entries
- update_glossary_entry: Update an existing entry
- delete_glossary_entry: Delete an entry
"""

import logging
import warnings

from pydantic.json_schema import PydanticJsonSchemaWarning

# Suppress warning for UNSET sentinel (intentionally non-JSON-serializable)
warnings.filterwarnings(
    "ignore",
    category=PydanticJsonSchemaWarning,
    message=".*UNSET.*",
)

from vector_core import UNSET, UnsetType, is_set, validate_limit
from vector_core.errors import ErrorCode, error_response
from vector_core.glossary import GlossaryNotFoundError, TermExistsError

from mcp_notes.app import mcp
from mcp_notes.singletons import (
    get_glossary_indexer,
    get_glossary_store,
    get_integrity_manager,
    get_search,
)

logger = logging.getLogger(__name__)


def _require_text(value: str, field: str) -> dict | None:
    """Return an INVALID_INPUT error dict if a text field is blank, else None.

    These tools call the GlossaryStore directly (not through vector-core's
    GlossaryToolHelper), so the store-facing validation has to live here.
    """
    if not isinstance(value, str) or not value.strip():
        return error_response(
            ErrorCode.INVALID_INPUT, f"{field} must be a non-empty string"
        )
    return None


def _require_alias_texts(aliases: list[str]) -> dict | None:
    """Return an INVALID_INPUT error dict if any alias is blank or a duplicate
    after normalization (strip + case-fold), else None.

    Duplicates must be caught before the store touches the database: alias
    replacement deletes the old aliases first, so a UNIQUE-constraint failure
    mid-insert would leave the entry partially mutated.
    """
    seen: set[str] = set()
    for alias in aliases:
        if not isinstance(alias, str) or not alias.strip():
            return error_response(
                ErrorCode.INVALID_INPUT, "aliases must not contain empty strings"
            )
        normalized = alias.strip().lower()
        if normalized in seen:
            return error_response(
                ErrorCode.INVALID_INPUT, f"duplicate alias: {alias.strip()}"
            )
        seen.add(normalized)
    return None


def _validate_update_inputs(
    term: str | None,
    expansion: str | None,
    definition: str | None,
    aliases: list[str] | None | UnsetType,
) -> dict | None:
    """Validate update_glossary_entry inputs; return an error dict or None.

    None means "leave unchanged" for term/expansion/definition, so only
    explicit values are checked; aliases use UNSET for "unchanged" and
    None/[] for "clear", which are valid and stay untouched.
    """
    fields = ((term, "term"), (expansion, "expansion"), (definition, "definition"))
    for value, field in fields:
        if value is not None:
            err = _require_text(value, field)
            if err:
                return err
    if is_set(aliases) and aliases is not None:
        return _require_alias_texts(aliases)
    return None


@mcp.tool()
async def add_glossary_entry(
    term: str,
    expansion: str,
    definition: str,
    domain: str | None = None,
    aliases: list[str] | None = None,
) -> dict:
    """
    Add a new glossary entry.

    Args:
        term: Canonical term (e.g., "USAF")
        expansion: Full expansion (e.g., "United States Air Force")
        definition: Detailed definition
        domain: Optional category (e.g., "military", "tech", "finance")
        aliases: Optional alternative terms that point to this entry

    Returns:
        Created entry as dict, or error dict for blank/duplicate input
    """
    for value, field in ((term, "term"), (expansion, "expansion"), (definition, "definition")):
        err = _require_text(value, field)
        if err:
            return err
    if aliases is not None:
        err = _require_alias_texts(aliases)
        if err:
            return err

    term = term.strip()
    expansion = expansion.strip()
    definition = definition.strip()
    # A blank domain means "no domain", matching update_glossary_entry's
    # documented "" semantics.
    domain = domain.strip() if isinstance(domain, str) and domain.strip() else None
    aliases = [a.strip() for a in aliases] if aliases is not None else None

    store = get_glossary_store()

    try:
        entry = store.create(
            term=term,
            expansion=expansion,
            definition=definition,
            domain=domain,
            aliases=aliases,
        )

        # Index for search
        try:
            indexer = await get_glossary_indexer()
            await indexer.index_entry(entry.id)
        except Exception as e:
            logger.warning(f"Failed to index glossary entry: {e}")

        return entry.to_dict()

    except TermExistsError as e:
        return error_response(ErrorCode.DUPLICATE, f"Term '{e.term}' already exists")


@mcp.tool()
async def lookup_term(term: str) -> dict:
    """
    Exact lookup by term or alias (case-insensitive).

    Args:
        term: Term to look up (e.g., "usaf", "USAF", "US Air Force")

    Returns:
        Glossary entry if found, or error dict
    """
    store = get_glossary_store()
    entry = store.lookup(term)

    if entry:
        return entry.to_dict()
    return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Term not found: {term}")


@mcp.tool()
async def search_glossary(
    query: str,
    domain: str | None = None,
    limit: int = 10,
) -> list[dict] | dict:
    """
    Semantic search for glossary entries.

    Args:
        query: Natural language search query
        domain: Optional domain filter
        limit: Max results (default 10, max 100)

    Returns:
        List of matching entries with relevance scores, or error dict
    """
    try:
        search = await get_search()
        limit = validate_limit(limit, default=10)

        # Search with glossary type filter
        results = await search.search(
            query=query,
            limit=limit,
            type_filter="glossary",
            domain=domain,
        )

        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.warning(f"Glossary search failed: {e}")
        return error_response(ErrorCode.SERVICE_UNAVAILABLE, f"Search unavailable: {e}")


@mcp.tool()
async def list_glossary(
    domain: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    List all glossary entries with optional domain filter.

    Args:
        domain: Optional domain filter
        limit: Max results (default 50, max 100)

    Returns:
        List of glossary entry summaries
    """
    store = get_glossary_store()
    limit = validate_limit(limit, default=50)

    summaries = store.list_all(domain=domain, limit=limit)
    return [s.to_dict() for s in summaries]


@mcp.tool()
async def update_glossary_entry(
    term_or_id: str,
    term: str | None = None,
    expansion: str | None = None,
    definition: str | None = None,
    domain: str | None | UnsetType = UNSET,
    aliases: list[str] | None | UnsetType = UNSET,
) -> dict:
    """
    Update an existing glossary entry. Only provided fields are updated.

    Args:
        term_or_id: Term (case-insensitive) or UUID to identify the entry
        term: New canonical term (optional)
        expansion: New expansion (optional)
        definition: New definition (optional)
        domain: New domain (optional, pass "" or null to clear)
        aliases: New aliases (optional, pass [] to clear)

    Returns:
        Updated entry as dict
    """
    # Validate provided fields before touching the store.
    err = _validate_update_inputs(term, expansion, definition, aliases)
    if err:
        return err
    term = term.strip() if term is not None else None
    expansion = expansion.strip() if expansion is not None else None
    definition = definition.strip() if definition is not None else None
    if is_set(domain) and domain is not None:
        # "" is documented as "clear"; store it as a real NULL.
        domain = domain.strip() or None
    if is_set(aliases) and aliases is not None:
        aliases = [a.strip() for a in aliases]

    store = get_glossary_store()

    # Find entry by term or ID
    entry = store.find_by_term_or_id(term_or_id)
    if not entry:
        return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Entry not found: {term_or_id}")

    try:
        updated = store.update(
            entry_id=entry.id,
            term=term,
            expansion=expansion,
            definition=definition,
            domain=domain,  # UNSET means not provided, None means clear
            aliases=aliases,  # UNSET means not provided, [] means clear
        )

        # Re-index
        try:
            indexer = await get_glossary_indexer()
            await indexer.index_entry(updated.id)
        except Exception as e:
            logger.warning(f"Failed to re-index glossary entry: {e}")

        # Mark fact sources as modified
        integrity = get_integrity_manager()
        try:
            sources_marked = integrity.mark_glossary_modified(updated.id)
            if sources_marked > 0:
                logger.info(
                    f"Marked {sources_marked} fact sources as modified "
                    f"for glossary {updated.id}"
                )
        except Exception as e:
            logger.warning(f"Failed to mark fact sources as modified: {e}")

        return updated.to_dict()

    except TermExistsError as e:
        return error_response(ErrorCode.DUPLICATE, f"Term '{e.term}' already exists")
    except GlossaryNotFoundError:
        return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Entry not found: {term_or_id}")


@mcp.tool()
async def delete_glossary_entry(term_or_id: str) -> dict:
    """
    Delete a glossary entry by term or UUID.

    Args:
        term_or_id: Term (case-insensitive) or UUID string

    Returns:
        Success status or error
    """
    store = get_glossary_store()
    integrity = get_integrity_manager()

    # Find entry by term or ID
    entry = store.find_by_term_or_id(term_or_id)
    if not entry:
        return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Entry not found: {term_or_id}")

    term = entry.term
    entry_id = entry.id

    # Delete
    store.delete(entry_id)

    # Remove from index
    try:
        indexer = await get_glossary_indexer()
        await indexer.delete_entry_index(entry_id)
    except Exception as e:
        logger.warning(f"Failed to remove glossary entry from index: {e}")

    # Mark fact sources as deleted
    try:
        sources_marked = integrity.mark_glossary_deleted(entry_id)
        if sources_marked > 0:
            logger.info(f"Marked {sources_marked} fact sources as deleted for glossary {entry_id}")
    except Exception as e:
        logger.warning(f"Failed to mark fact sources as deleted: {e}")

    return {"success": True, "deleted_term": term, "deleted_id": str(entry_id)}
