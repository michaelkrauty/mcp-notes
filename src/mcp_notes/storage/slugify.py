"""Slug generation utilities for note filenames."""

import re
import unicodedata
from uuid import UUID

# Default max length for slug portion of filename
DEFAULT_SLUG_MAX_LENGTH = 50


def generate_slug(title: str, max_length: int = DEFAULT_SLUG_MAX_LENGTH) -> str:
    """
    Generate a URL-friendly slug from a title.

    Process:
    1. NFD normalize (separate base chars from diacritics)
    2. Strip diacritics (combining characters)
    3. Lowercase
    4. Replace spaces/underscores with hyphens
    5. Remove non-alphanumeric (except hyphens)
    6. Collapse multiple hyphens
    7. Strip leading/trailing hyphens
    8. Truncate at word boundary if needed

    Args:
        title: The note title to slugify
        max_length: Maximum slug length (default 50)

    Returns:
        URL-friendly slug string
    """
    if not title:
        return "untitled"

    # NFD normalize and strip diacritics
    normalized = unicodedata.normalize("NFD", title)
    stripped = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    # Lowercase
    slug = stripped.lower()

    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)

    # Remove non-alphanumeric except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)

    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)

    # Strip leading/trailing hyphens
    slug = slug.strip("-")

    # Handle empty result
    if not slug:
        return "untitled"

    # Truncate at word boundary if needed
    if len(slug) > max_length:
        truncated = slug[:max_length]
        # Try to break at hyphen (word boundary)
        last_hyphen = truncated.rfind("-")
        if last_hyphen > max_length * 0.6:  # Don't cut too short
            truncated = truncated[:last_hyphen]
        slug = truncated.rstrip("-")

    return slug or "untitled"


def slugify_path_segment(segment: str) -> str:
    """
    Slugify a single path segment (category name).

    Args:
        segment: Category name like "Work & Projects"

    Returns:
        Slugified segment like "work-projects"
    """
    return generate_slug(segment, max_length=100)  # Longer allowed for paths


def slugify_category_path(category: str) -> str:
    """
    Slugify a full category path.

    Args:
        category: Category path like "Work & Projects/Client X"

    Returns:
        Slugified path like "work-projects/client-x"
    """
    if not category:
        return ""

    segments = category.split("/")
    slugified = [slugify_path_segment(seg) for seg in segments if seg.strip()]
    return "/".join(slugified)


def build_filename(
    title: str, note_id: UUID, max_slug_length: int = DEFAULT_SLUG_MAX_LENGTH
) -> str:
    """
    Build a note filename from title and UUID.

    Args:
        title: Note title
        note_id: Note UUID
        max_slug_length: Maximum length for slug portion

    Returns:
        Filename like "my-project-plan-550e8400-e29b-41d4-a716-446655440000.md"
    """
    slug = generate_slug(title, max_length=max_slug_length)
    return f"{slug}-{note_id}.md"


def extract_uuid_from_filename(filename: str) -> UUID | None:
    """
    Extract UUID from a note filename.

    Args:
        filename: Filename like "my-note-550e8400-e29b-41d4-a716-446655440000.md"

    Returns:
        UUID if found, None otherwise
    """
    # UUID pattern at end of filename before .md
    pattern = r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.md$"
    match = re.search(pattern, filename)
    if match:
        try:
            return UUID(match.group(1))
        except ValueError:
            return None
    return None
