"""Query filter parsing for notes search."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from qdrant_client.models import FieldCondition, MatchValue

from mcp_notes.constants import normalize_tag


@dataclass
class SearchFilters:
    """Parsed search filters from query string."""

    # Semantic query (after removing filters)
    query: str = ""

    # Tag filters
    tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)

    # Category filter
    category: str | None = None

    # Date filters
    after: datetime | None = None
    before: datetime | None = None

    # Title filter
    title_contains: str | None = None

    def add_tags(self, tags: list[str] | None) -> None:
        """Add explicit tag filters, normalized to stored form and de-duplicated.

        Mirrors how ``tag:`` query syntax is parsed, so a caller passing
        ``["Work"]`` filters on the stored ``"work"`` instead of matching
        nothing. ``None`` and tags that normalize to empty are ignored.
        """
        for raw in tags or []:
            tag = normalize_tag(raw)
            if tag and tag not in self.tags:
                self.tags.append(tag)


# Filter patterns
# Use negative lookbehind to prevent TAG_PATTERN from matching inside -tag:
TAG_PATTERN = re.compile(r"(?<!-)tag:(\S+)")
EXCLUDE_TAG_PATTERN = re.compile(r"-tag:(\S+)")
CATEGORY_PATTERN = re.compile(r"category:(\S+)")
AFTER_PATTERN = re.compile(r"after:(\d{4}-\d{2}-\d{2})")
BEFORE_PATTERN = re.compile(r"before:(\d{4}-\d{2}-\d{2})")
TITLE_PATTERN = re.compile(r"title:(\S+)")


def parse_search_query(query: str) -> SearchFilters:
    """
    Parse search query into semantic query and filters.

    Supported syntax:
    - tag:tagname - Filter by tag
    - -tag:tagname - Exclude tag
    - category:path - Filter by category
    - after:YYYY-MM-DD - Created after date
    - before:YYYY-MM-DD - Created before date
    - title:text - Title contains text

    Example:
        "meeting notes about auth tag:work after:2024-01-01"
        -> query="meeting notes about auth", tags=["work"], after=2024-01-01

    Args:
        query: Raw search query

    Returns:
        SearchFilters with parsed values
    """
    filters = SearchFilters()

    # Extract tags (normalized to their stored form so filters match)
    for match in TAG_PATTERN.finditer(query):
        tag = normalize_tag(match.group(1))
        if tag and tag not in filters.tags:
            filters.tags.append(tag)
    query = TAG_PATTERN.sub("", query)

    # Extract excluded tags
    for match in EXCLUDE_TAG_PATTERN.finditer(query):
        tag = normalize_tag(match.group(1))
        if tag and tag not in filters.exclude_tags:
            filters.exclude_tags.append(tag)
    query = EXCLUDE_TAG_PATTERN.sub("", query)

    # Extract category
    if cat_match := CATEGORY_PATTERN.search(query):
        filters.category = cat_match.group(1).strip()
    query = CATEGORY_PATTERN.sub("", query)

    # Extract after date
    if after_match := AFTER_PATTERN.search(query):
        try:
            filters.after = datetime.strptime(
                after_match.group(1), "%Y-%m-%d"
            ).replace(tzinfo=UTC)
        except ValueError:
            pass
    query = AFTER_PATTERN.sub("", query)

    # Extract before date
    if before_match := BEFORE_PATTERN.search(query):
        try:
            filters.before = datetime.strptime(
                before_match.group(1), "%Y-%m-%d"
            ).replace(tzinfo=UTC)
        except ValueError:
            pass
    query = BEFORE_PATTERN.sub("", query)

    # Extract title filter
    if title_match := TITLE_PATTERN.search(query):
        filters.title_contains = title_match.group(1).strip()
    query = TITLE_PATTERN.sub("", query)

    # Clean up remaining query
    filters.query = " ".join(query.split())

    return filters


def filters_to_qdrant(filters: SearchFilters) -> list:
    """
    Convert SearchFilters to Qdrant filter conditions.

    Args:
        filters: Parsed search filters

    Returns:
        List of Qdrant FieldCondition objects
    """
    conditions = []

    # Tag filters (must have all specified tags). Skip empty tags defensively:
    # a MatchValue("") matches no stored tag and would filter out every result.
    for tag in filters.tags:
        if not tag:
            continue
        conditions.append(
            FieldCondition(key="tags", match=MatchValue(value=tag))
        )

    # Category filter (exact match — prefix matching is done in list_notes)
    if filters.category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=filters.category))
        )

    # Note: Date filters and exclude_tags need post-filtering
    # Qdrant doesn't support all filter types natively

    return conditions


def apply_post_filters(
    results: list,
    filters: SearchFilters,
) -> list:
    """
    Apply filters that can't be done in Qdrant.

    Args:
        results: Search results with payload
        filters: Parsed filters

    Returns:
        Filtered results
    """
    filtered = []

    for result in results:
        payload = result.get("payload", {})

        # Check excluded tags
        if filters.exclude_tags:
            note_tags = payload.get("tags", [])
            if any(tag in note_tags for tag in filters.exclude_tags):
                continue

        # Check date filters
        if filters.after or filters.before:
            created_str = payload.get("created", "")
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    if filters.after and created < filters.after:
                        continue
                    if filters.before and created > filters.before:
                        continue
                except ValueError:
                    pass

        # Check title filter
        if filters.title_contains:
            title = payload.get("title", "")
            if filters.title_contains.lower() not in title.lower():
                continue

        filtered.append(result)

    return filtered
