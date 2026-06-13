"""Constants and validators for mcp-notes."""

import re

from mcp_notes.settings import settings

# Tag validation pattern (alphanumeric with hyphens, cannot start with hyphen)
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def normalize_tag(tag: str) -> str:
    """
    Normalize a tag to its stored form: lowercase, stripped, spaces to hyphens.

    This is the single source of truth for how a raw tag maps to what is
    persisted on a note. Tag *filters* (search/list) must normalize through
    this same function, or a filter like ``"My Tag"`` would never match the
    stored ``"my-tag"`` and silently return nothing.
    """
    return tag.lower().strip().replace(" ", "-")


def validate_tag(tag: str) -> tuple[str, str | None]:
    """
    Validate and normalize a tag name.

    Args:
        tag: Raw tag input

    Returns:
        Tuple of (normalized_tag, error_message or None)
    """
    normalized = normalize_tag(tag)

    if not normalized:
        return "", "Tag cannot be empty"

    if len(normalized) > settings.max_tag_length:
        return "", f"Tag exceeds maximum length of {settings.max_tag_length} characters"

    if not TAG_PATTERN.match(normalized):
        return "", (
            "Tag must contain only lowercase letters, numbers, and hyphens "
            "(cannot start with hyphen)"
        )

    return normalized, None
