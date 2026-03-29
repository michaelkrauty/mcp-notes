"""Fact graph operations.

Tools:
- add_fact: Add a new fact (subject-predicate-object triple)
- add_facts_batch: Batch import multiple facts
- update_fact: Update an existing fact's metadata
- delete_fact: Delete a fact
- query_facts: Query facts by various criteria
- get_entity: Get all facts involving an entity
- list_facts: List facts as summaries
- search_facts: Semantic search across indexed facts
- index_facts: Index all facts for semantic search
- find_connections: Find connections between entities
- get_neighbors: Get immediate neighbors of an entity
"""

from uuid import UUID

from vector_core import UNSET, UnsetType, validate_limit
from vector_core.errors import ErrorCode, error_response

from mcp_notes.app import mcp
from mcp_notes.facts import (
    DuplicateFactError,
    FactNotFoundError,
    FactSource,
    SourceType,
)
from mcp_notes.singletons import get_fact_indexer, get_fact_store, get_search

# Input validation limits (DoS protection)
MAX_ENTITY_NAME_LENGTH = 1000  # Maximum characters in entity name


def _validate_entity_name(name: str, field: str = "entity") -> str | dict:
    """
    Validate entity name for security/DoS protection.

    Args:
        name: Entity name to validate
        field: Field name for error message

    Returns:
        Validated name (stripped) or error dict
    """
    if not name or not name.strip():
        return error_response(ErrorCode.INVALID_INPUT, f"{field} cannot be empty")
    if len(name) > MAX_ENTITY_NAME_LENGTH:
        return error_response(
            ErrorCode.INVALID_INPUT,
            f"{field} exceeds maximum length of {MAX_ENTITY_NAME_LENGTH} characters"
        )
    return name.strip()


@mcp.tool()
async def add_fact(
    subject: str,
    predicate: str,
    object: str,
    subject_type: str = "entity",
    object_type: str = "entity",
    context: str | None = None,
    confidence: float = 1.0,
    valid_from: str | None = None,
    valid_to: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    source_path: str | None = None,
    source_hash: str | None = None,
    source_location: str | None = None,
) -> dict:
    """
    Add a new fact (subject-predicate-object triple).

    Args:
        subject: Subject entity name (e.g., "John Smith")
        predicate: Relationship type (e.g., "works_at", "served_in")
        object: Object entity name (e.g., "Acme Corp")
        subject_type: Type of subject (e.g., "person", "organization")
        object_type: Type of object (e.g., "organization", "military_unit")
        context: Optional context description (e.g., "as squad leader")
        confidence: Confidence level 0.0-1.0 (1.0 = verified/manual)
        valid_from: Start date of validity (ISO format: YYYY-MM-DD)
        valid_to: End date of validity (ISO format: YYYY-MM-DD)
        source_type: Source type ("note", "document", "glossary", "manual")
        source_id: Source UUID (for notes/glossary)
        source_path: Source path (for documents)
        source_hash: Content hash (for documents)
        source_location: Location within source (e.g., "page 3")

    Returns:
        Created fact as dict
    """
    from datetime import date as date_type

    # Validate entity names
    for field, value in [("subject", subject), ("predicate", predicate), ("object", object)]:
        validated = _validate_entity_name(value, field)
        if isinstance(validated, dict):
            return validated  # Error response

    store = get_fact_store()

    # Parse dates
    parsed_valid_from = None
    parsed_valid_to = None
    if valid_from:
        try:
            parsed_valid_from = date_type.fromisoformat(valid_from)
        except ValueError:
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid valid_from date format: {valid_from}. Use YYYY-MM-DD",
            )
    if valid_to:
        try:
            parsed_valid_to = date_type.fromisoformat(valid_to)
        except ValueError:
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid valid_to date format: {valid_to}. Use YYYY-MM-DD",
            )

    # Build source if provided
    source = None
    if source_type:
        try:
            src_type = SourceType(source_type)
        except ValueError:
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid source_type: {source_type}. "
                "Must be one of: note, document, glossary, manual"
            )

        try:
            source_id_uuid = UUID(source_id) if source_id else None
        except ValueError:
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid source_id UUID: {source_id}"
            )

        source = FactSource(
            source_type=src_type,
            source_id=source_id_uuid,
            source_path=source_path,
            content_hash=source_hash,
            location=source_location,
        )

    try:
        fact = store.create(
            subject=subject,
            predicate=predicate,
            object_value=object,
            subject_type=subject_type,
            object_type=object_type,
            context=context,
            confidence=confidence,
            valid_from=parsed_valid_from,
            valid_to=parsed_valid_to,
            source=source,
        )
        return fact.to_dict()

    except DuplicateFactError as e:
        return error_response(
            ErrorCode.DUPLICATE,
            f"Fact already exists with same subject/predicate/object (id={e.existing_id})"
        )


@mcp.tool()
async def add_facts_batch(facts: list[dict]) -> dict:
    """
    Batch import multiple facts in a single transaction.

    Args:
        facts: List of fact dicts, each with same fields as add_fact:
            - subject, predicate, object (required)
            - subject_type, object_type, context, confidence, valid_from, valid_to
            - source_type, source_id, source_path, source_hash, source_location

    Returns:
        Summary dict with added count, duplicates count, and any errors
    """
    from datetime import date as date_type

    store = get_fact_store()

    added = 0
    duplicates = 0
    errors: list[dict] = []

    for i, fact_data in enumerate(facts):
        # Validate required fields
        if not all(k in fact_data for k in ("subject", "predicate", "object")):
            errors.append({
                "index": i,
                "error": "Missing required field(s): subject, predicate, object",
            })
            continue

        # Validate entity names
        validation_failed = False
        for field in ("subject", "predicate", "object"):
            validated = _validate_entity_name(fact_data[field], field)
            if isinstance(validated, dict):
                errors.append({"index": i, "error": validated.get("error", f"Invalid {field}")})
                validation_failed = True
                break
        if validation_failed:
            continue

        # Parse dates
        parsed_valid_from = None
        parsed_valid_to = None
        if fact_data.get("valid_from"):
            try:
                parsed_valid_from = date_type.fromisoformat(fact_data["valid_from"])
            except ValueError:
                errors.append({"index": i, "error": "Invalid valid_from date"})
                continue
        if fact_data.get("valid_to"):
            try:
                parsed_valid_to = date_type.fromisoformat(fact_data["valid_to"])
            except ValueError:
                errors.append({"index": i, "error": "Invalid valid_to date"})
                continue

        # Build source if provided
        source = None
        if fact_data.get("source_type"):
            try:
                src_type = SourceType(fact_data["source_type"])
            except ValueError:
                errors.append({
                    "index": i,
                    "error": f"Invalid source_type: {fact_data['source_type']}",
                })
                continue

            try:
                source_id_uuid = (
                    UUID(fact_data["source_id"]) if fact_data.get("source_id") else None
                )
            except ValueError:
                errors.append({
                    "index": i,
                    "error": f"Invalid source_id UUID: {fact_data['source_id']}",
                })
                continue

            source = FactSource(
                source_type=src_type,
                source_id=source_id_uuid,
                source_path=fact_data.get("source_path"),
                content_hash=fact_data.get("source_hash"),
                location=fact_data.get("source_location"),
            )

        try:
            store.create(
                subject=fact_data["subject"],
                predicate=fact_data["predicate"],
                object_value=fact_data["object"],
                subject_type=fact_data.get("subject_type", "entity"),
                object_type=fact_data.get("object_type", "entity"),
                context=fact_data.get("context"),
                confidence=fact_data.get("confidence", 1.0),
                valid_from=parsed_valid_from,
                valid_to=parsed_valid_to,
                source=source,
            )
            added += 1

        except DuplicateFactError:
            duplicates += 1

    return {
        "added": added,
        "duplicates": duplicates,
        "errors": errors,
        "total_processed": len(facts),
    }


@mcp.tool()
async def update_fact(
    fact_id: str,
    context: str | None = None,
    confidence: float | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> dict:
    """
    Update an existing fact's metadata.

    Note: Subject, predicate, and object are immutable (delete and recreate if needed).

    Args:
        fact_id: Fact UUID string
        context: New context description (pass null to clear)
        confidence: New confidence level 0.0-1.0
        valid_from: New start date (ISO format, pass null to clear)
        valid_to: New end date (ISO format, pass null to clear)

    Returns:
        Updated fact as dict
    """
    from datetime import date as date_type

    store = get_fact_store()

    try:
        uuid = UUID(fact_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {fact_id}")

    # Parse dates if provided - UNSET means not provided
    parsed_valid_from: date_type | None | UnsetType = UNSET
    parsed_valid_to: date_type | None | UnsetType = UNSET

    if valid_from is not None:
        if valid_from == "":
            parsed_valid_from = None  # Clear
        else:
            try:
                parsed_valid_from = date_type.fromisoformat(valid_from)
            except ValueError:
                return error_response(
                    ErrorCode.INVALID_INPUT,
                    f"Invalid valid_from date format: {valid_from}. Use YYYY-MM-DD",
                )

    if valid_to is not None:
        if valid_to == "":
            parsed_valid_to = None  # Clear
        else:
            try:
                parsed_valid_to = date_type.fromisoformat(valid_to)
            except ValueError:
                return error_response(
                    ErrorCode.INVALID_INPUT,
                    f"Invalid valid_to date format: {valid_to}. Use YYYY-MM-DD",
                )

    try:
        # Use UNSET as sentinel for "not provided" - update() handles this
        fact = store.update(
            fact_id=uuid,
            context=context if context is not None else UNSET,
            confidence=confidence,
            valid_from=parsed_valid_from,
            valid_to=parsed_valid_to,
        )
        return fact.to_dict()

    except FactNotFoundError:
        return error_response(ErrorCode.FACT_NOT_FOUND, f"Fact not found: {fact_id}")


@mcp.tool()
async def delete_fact(fact_id: str) -> dict:
    """
    Delete a fact and its sources.

    Args:
        fact_id: Fact UUID string

    Returns:
        Success status
    """
    store = get_fact_store()

    try:
        uuid = UUID(fact_id)
    except ValueError:
        return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {fact_id}")

    deleted = store.delete(uuid)
    if deleted:
        return {"success": True, "deleted_id": fact_id}
    else:
        return error_response(ErrorCode.FACT_NOT_FOUND, f"Fact not found: {fact_id}")


@mcp.tool()
async def query_facts(
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,
    subject_type: str | None = None,
    object_type: str | None = None,
    min_confidence: float | None = None,
    valid_at: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Query facts by various criteria.

    Args:
        subject: Filter by subject (case-insensitive)
        predicate: Filter by predicate (case-insensitive)
        object: Filter by object (case-insensitive)
        subject_type: Filter by subject type
        object_type: Filter by object type
        min_confidence: Minimum confidence threshold
        valid_at: Filter by validity date (ISO format: YYYY-MM-DD)
        limit: Maximum results (1-100, default 50)

    Returns:
        List of matching facts
    """
    from datetime import date as date_type

    store = get_fact_store()

    # Parse valid_at date
    parsed_valid_at = None
    if valid_at:
        try:
            parsed_valid_at = date_type.fromisoformat(valid_at)
        except ValueError:
            return [error_response(
                ErrorCode.INVALID_INPUT,
                f"Invalid valid_at date format: {valid_at}. Use YYYY-MM-DD",
            )]

    limit = validate_limit(limit, 50)

    facts = store.query(
        subject=subject,
        predicate=predicate,
        object_value=object,
        subject_type=subject_type,
        object_type=object_type,
        min_confidence=min_confidence,
        valid_at=parsed_valid_at,
        limit=limit,
    )

    return [f.to_dict() for f in facts]


@mcp.tool()
async def get_entity(name: str, entity_type: str | None = None) -> dict:
    """
    Get all facts involving an entity (as subject or object).

    Args:
        name: Entity name (case-insensitive)
        entity_type: Optional entity type filter

    Returns:
        Dict with entity info and related facts
    """
    # Validate entity name
    validated = _validate_entity_name(name, "name")
    if isinstance(validated, dict):
        return validated  # Error response
    name = validated

    store = get_fact_store()

    facts = store.get_entity_facts(name, entity_type)

    # Categorize facts
    as_subject = []
    as_object = []
    for fact in facts:
        if fact.subject.lower() == name.lower():
            as_subject.append(fact.to_dict())
        if fact.object_value.lower() == name.lower():
            as_object.append(fact.to_dict())

    return {
        "entity": name,
        "entity_type": entity_type,
        "as_subject": as_subject,
        "as_object": as_object,
        "total_facts": len(facts),
    }


@mcp.tool()
async def list_facts(
    limit: int = 50,
    subject_type: str | None = None,
    object_type: str | None = None,
    predicate: str | None = None,
) -> list[dict]:
    """
    List facts as lightweight summaries.

    Args:
        limit: Maximum results (1-100, default 50)
        subject_type: Filter by subject type
        object_type: Filter by object type
        predicate: Filter by predicate

    Returns:
        List of fact summaries
    """
    store = get_fact_store()

    limit = validate_limit(limit, 50)

    summaries = store.list_summaries(
        subject_type=subject_type,
        object_type=object_type,
        predicate=predicate,
        limit=limit,
    )

    return [s.to_dict() for s in summaries]


@mcp.tool()
async def search_facts(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Semantic search across indexed facts.

    Args:
        query: Natural language search query
        limit: Maximum results (1-100, default 10)

    Returns:
        List of matching facts with scores
    """
    limit = validate_limit(limit, 10)

    # Use search engine with type_filter="fact"
    engine = await get_search()
    results = await engine.search(
        query=query,
        type_filter="fact",
        limit=limit,
    )

    # Convert SearchResult to fact-friendly format
    output = []
    for r in results:
        output.append({
            "fact_id": str(r.note.id),
            "title": r.note.title,
            "excerpt": r.note.excerpt,
            "score": r.score,
            "types": r.note.tags,  # subject_type and object_type
            "highlights": r.highlights,
        })

    return output


@mcp.tool()
async def index_facts(
    force: bool = False,
) -> dict:
    """
    Index all facts for semantic search.

    Args:
        force: If True, reindex all facts. If False, only index new facts.

    Returns:
        Indexing result with counts
    """
    indexer = await get_fact_indexer()
    result = await indexer.index_all(force=force)
    return result


@mcp.tool()
async def find_connections(
    source_entity: str,
    target_entity: str | None = None,
    source_type: str | None = None,
    target_type: str | None = None,
    max_depth: int = 3,
    limit: int = 10,
) -> list[dict]:
    """
    Find connections between entities using BFS graph traversal.

    Discovers how entities are related through chains of facts.

    Args:
        source_entity: Starting entity name
        target_entity: Target entity to find path to (optional)
                      If None, returns all reachable entities up to max_depth
        source_type: Type of source entity (optional, for disambiguation)
        target_type: Type of target entity (optional, for disambiguation)
        max_depth: Maximum path length (1-10, default 3)
        limit: Maximum paths to return (1-100, default 10)

    Returns:
        List of connection paths, each containing:
        - path: List of facts connecting the entities
        - entities: List of entity names in the path
    """
    # Validate entity names
    validated_source = _validate_entity_name(source_entity, "source_entity")
    if isinstance(validated_source, dict):
        return [validated_source]  # Error response
    source_entity = validated_source

    if target_entity is not None:
        validated_target = _validate_entity_name(target_entity, "target_entity")
        if isinstance(validated_target, dict):
            return [validated_target]  # Error response
        target_entity = validated_target

    store = get_fact_store()

    max_depth = max(1, min(10, max_depth))
    limit = validate_limit(limit, 10)

    paths = store.find_connections(
        source_entity=source_entity,
        target_entity=target_entity,
        source_type=source_type,
        target_type=target_type,
        max_depth=max_depth,
        limit=limit,
    )

    # Convert to serializable format
    result = []
    for path in paths:
        # Extract entity chain from path
        entities: list[str] = []
        for fact in path:
            if not entities:
                entities.append(fact.subject)
            if fact.object_value not in entities:
                entities.append(fact.object_value)

        result.append({
            "path": [
                {
                    "id": str(f.id),
                    "subject": f.subject,
                    "subject_type": f.subject_type,
                    "predicate": f.predicate,
                    "object": f.object_value,
                    "object_type": f.object_type,
                }
                for f in path
            ],
            "entities": entities,
            "length": len(path),
        })

    return result


@mcp.tool()
async def get_neighbors(
    entity: str,
    entity_type: str | None = None,
) -> list[dict]:
    """
    Get immediate neighbors of an entity in the fact graph.

    Returns all entities directly connected by a single fact.

    Args:
        entity: Entity name to get neighbors for
        entity_type: Type filter (optional, for disambiguation)

    Returns:
        List of neighbors with:
        - entity: Neighbor entity name
        - type: Neighbor entity type
        - predicate: Relationship predicate
        - direction: 'outgoing' (entity is subject) or 'incoming' (entity is object)
        - fact_id: UUID of connecting fact
    """
    # Validate entity name
    validated = _validate_entity_name(entity, "entity")
    if isinstance(validated, dict):
        return [validated]  # Error response
    entity = validated

    store = get_fact_store()
    return store.get_neighbors(entity=entity, entity_type=entity_type)
