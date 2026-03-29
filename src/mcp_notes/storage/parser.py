"""Markdown + YAML frontmatter parsing."""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import yaml

logger = logging.getLogger(__name__)

# Maximum frontmatter size in bytes (prevents memory exhaustion DoS)
# Default: 64KB - should be more than enough for any reasonable frontmatter
MAX_FRONTMATTER_SIZE = 64 * 1024


class FrontmatterTooLargeError(ValueError):
    """Raised when frontmatter exceeds maximum allowed size."""

    def __init__(self, size: int, max_size: int = MAX_FRONTMATTER_SIZE):
        self.size = size
        self.max_size = max_size
        super().__init__(
            f"Frontmatter too large: {size:,} bytes (max {max_size:,} bytes). "
            f"This limit prevents memory exhaustion attacks."
        )


@dataclass
class ParsedNote:
    """Parsed note with frontmatter and content."""

    id: UUID
    title: str
    content: str
    tags: list[str]
    category: str | None
    links: list[UUID]
    created: datetime
    modified: datetime

    # Raw content for re-serialization
    raw_frontmatter: dict
    body: str  # Content without frontmatter


FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def parse_note(file_content: str) -> ParsedNote:  # noqa: PLR0912
    """
    Parse a note file with YAML frontmatter.

    Args:
        file_content: Full file content

    Returns:
        ParsedNote with parsed fields

    Raises:
        ValueError: If frontmatter is invalid or missing required fields
    """
    match = FRONTMATTER_PATTERN.match(file_content)
    if not match:
        raise ValueError("Note missing YAML frontmatter (---...---)")

    frontmatter_str = match.group(1)
    body = file_content[match.end():]

    # Check frontmatter size before parsing (prevents memory exhaustion DoS)
    frontmatter_bytes = len(frontmatter_str.encode("utf-8"))
    if frontmatter_bytes > MAX_FRONTMATTER_SIZE:
        raise FrontmatterTooLargeError(frontmatter_bytes)

    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
        raise ValueError("Frontmatter must be a YAML mapping")

    # Required fields
    if "id" not in frontmatter:
        raise ValueError("Frontmatter missing required field: id")
    if "title" not in frontmatter:
        raise ValueError("Frontmatter missing required field: title")
    if "created" not in frontmatter:
        raise ValueError("Frontmatter missing required field: created")
    if "modified" not in frontmatter:
        raise ValueError("Frontmatter missing required field: modified")

    # Parse UUID
    try:
        note_id = UUID(str(frontmatter["id"]))
    except ValueError as e:
        raise ValueError(f"Invalid UUID in frontmatter: {e}") from e

    # Parse timestamps
    created = _parse_datetime(frontmatter["created"])
    modified = _parse_datetime(frontmatter["modified"])

    # Parse optional fields
    tags = frontmatter.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags] if tags else []
    tags = [str(t).lower().strip() for t in tags]

    category = frontmatter.get("category")
    if category:
        category = str(category).strip()

    # Parse links (list of UUIDs)
    raw_links = frontmatter.get("links", [])
    if not isinstance(raw_links, list):
        raw_links = [raw_links] if raw_links else []

    links = []
    for link in raw_links:
        try:
            links.append(UUID(str(link)))
        except ValueError:
            logger.debug(f"Skipping invalid UUID in note links: {link}")

    return ParsedNote(
        id=note_id,
        title=str(frontmatter["title"]),
        content=file_content,
        tags=tags,
        category=category,
        links=links,
        created=created,
        modified=modified,
        raw_frontmatter=frontmatter,
        body=body.strip(),
    )


def _parse_datetime(value: str | datetime | None) -> datetime:
    """Parse datetime from various formats."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    if isinstance(value, str):
        # Try ISO format
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            pass

        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ]:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.replace(tzinfo=UTC)
            except ValueError:
                continue

    raise ValueError(f"Cannot parse datetime: {value}")


def serialize_note(
    note_id: UUID,
    title: str,
    body: str,
    tags: list[str] | None = None,
    category: str | None = None,  # Kept for API compat, but NOT written to frontmatter
    links: list[UUID] | None = None,
    created: datetime | None = None,
    modified: datetime | None = None,
) -> str:
    """
    Serialize a note to markdown with YAML frontmatter.

    Note: Category is NOT written to frontmatter - it's derived from folder path.
    The category parameter is kept for API compatibility but is ignored.

    Args:
        note_id: Note UUID
        title: Note title
        body: Note body content (without frontmatter)
        tags: Optional list of tags
        category: IGNORED - kept for API compatibility only
        links: Optional list of linked UUIDs
        created: Creation timestamp (defaults to now)
        modified: Modification timestamp (defaults to now)

    Returns:
        Complete note file content
    """
    # Silence unused parameter warning
    _ = category

    now = datetime.now(UTC)
    created = created or now
    modified = modified or now

    frontmatter: dict[str, str | list[str]] = {
        "id": str(note_id),
        "title": title,
        "created": created.isoformat(),
        "modified": modified.isoformat(),
    }

    if tags:
        frontmatter["tags"] = tags

    # Note: category is NOT written - it's derived from folder path

    if links:
        frontmatter["links"] = [str(link) for link in links]

    # Serialize YAML
    yaml_str = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    return f"---\n{yaml_str}---\n\n{body}"


def extract_inline_links(content: str) -> list[UUID]:
    """
    Extract [[uuid]] links from note content.

    Args:
        content: Note content

    Returns:
        List of UUIDs found in [[...]] syntax
    """
    pattern = re.compile(r"\[\[([0-9a-fA-F-]{36})\]\]")
    links = []

    for match in pattern.finditer(content):
        try:
            links.append(UUID(match.group(1)))
        except ValueError:
            pass

    return links
