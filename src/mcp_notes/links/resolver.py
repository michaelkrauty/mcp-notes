"""Link resolution for [[uuid]] syntax."""

import logging
from uuid import UUID

from mcp_notes.models import BrokenLink, NoteLinks, NoteSummary
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import (
    NoteNotFoundError,
    NoteStore,
    NoteTooLargeError,
    PathTraversalError,
)
from mcp_notes.storage.parser import ParsedNote, extract_inline_links, parse_note

logger = logging.getLogger(__name__)


def _parsed_to_summary(parsed: ParsedNote, category: str | None) -> NoteSummary:
    """Convert ParsedNote to NoteSummary without re-reading file.

    This avoids the N+1 query problem where we already have parsed data
    but would otherwise call get_summary() which re-reads the file.

    Args:
        parsed: ParsedNote object
        category: Category derived from path (not frontmatter)
    """
    excerpt = parsed.body[:settings.excerpt_length]
    if len(parsed.body) > settings.excerpt_length:
        # Try to break at word boundary
        last_space = excerpt.rfind(" ")
        if last_space > settings.excerpt_length * 0.7:
            excerpt = excerpt[:last_space] + "..."
        else:
            excerpt += "..."

    return NoteSummary(
        id=parsed.id,
        title=parsed.title,
        tags=parsed.tags,
        category=category,  # From path, not frontmatter
        created=parsed.created,
        modified=parsed.modified,
        excerpt=excerpt,
    )


class LinkResolver:
    """
    Resolves and tracks [[uuid]] links between notes.

    Provides:
    - Forward links (outgoing): Notes this note links to
    - Backlinks (incoming): Notes that link to this note
    - Broken link detection
    """

    def __init__(self, note_store: NoteStore | None = None):
        """
        Initialize link resolver.

        Args:
            note_store: NoteStore instance
        """
        self.note_store = note_store or NoteStore()

    def get_note_links(self, note_id: UUID) -> NoteLinks:
        """
        Get all links for a note.

        Args:
            note_id: Note UUID

        Returns:
            NoteLinks with outgoing, incoming (backlinks), and broken links
        """
        try:
            note = self.note_store.read(note_id)
            parsed = parse_note(note.content)
        except (
            NoteNotFoundError,
            ValueError,
            PathTraversalError,
            NoteTooLargeError,
            OSError,
        ):
            # The source note is missing, or present-but-corrupt (unparseable
            # frontmatter), oversized, or unreadable. It has no resolvable links,
            # so return an empty result instead of crashing the whole links view,
            # mirroring the per-file tolerance of NoteStore.iter_all/list_all.
            return NoteLinks()

        # Get outgoing links (from frontmatter + inline)
        outgoing_ids = set(parsed.links)
        inline_links = extract_inline_links(parsed.body)
        outgoing_ids.update(inline_links)

        # Resolve outgoing links
        outgoing = []
        broken = []
        for target_id in outgoing_ids:
            try:
                summary = self.note_store.get_summary(target_id)
                outgoing.append(summary)
            except (
                NoteNotFoundError,
                ValueError,
                PathTraversalError,
                NoteTooLargeError,
                OSError,
            ):
                # The target is missing, or present-but-corrupt (unparseable
                # frontmatter), oversized, or unreadable. In every case it is not
                # a usable link, so report it as broken rather than letting one
                # bad note abort the entire links view.
                logger.debug("Treating link target %s as broken", target_id)
                broken.append(target_id)

        # Find incoming links (backlinks)
        incoming = self._find_backlinks(note_id)

        return NoteLinks(
            outgoing=outgoing,
            incoming=incoming,
            broken=broken,
        )

    def _find_backlinks(self, note_id: UUID) -> list[NoteSummary]:
        """Find all notes that link to this note.

        Uses already-parsed data to avoid N+1 query pattern - we don't
        re-read files we've already parsed during iteration.
        """
        backlinks = []

        for parsed, category in self.note_store.iter_all():
            if parsed.id == note_id:
                continue

            # Check frontmatter links first
            if note_id in parsed.links:
                # Use already-parsed data instead of re-reading file
                backlinks.append(_parsed_to_summary(parsed, category))
                continue

            # Check inline links
            inline = extract_inline_links(parsed.body)
            if note_id in inline:
                # Use already-parsed data instead of re-reading file
                backlinks.append(_parsed_to_summary(parsed, category))

        return backlinks

    def get_all_broken_links(self) -> list[BrokenLink]:
        """
        Find all broken links across all notes.

        Returns:
            List of BrokenLink objects
        """
        broken_links = []

        for parsed, _category in self.note_store.iter_all():
            # Collect all outgoing links
            outgoing_ids = set(parsed.links)
            inline_links = extract_inline_links(parsed.body)
            outgoing_ids.update(inline_links)

            # Check each link
            for target_id in outgoing_ids:
                if not self.note_store.exists(target_id):
                    broken_links.append(
                        BrokenLink(
                            source_note_id=parsed.id,
                            source_note_title=parsed.title,
                            broken_target_id=target_id,
                        )
                    )

        return broken_links

    def get_orphan_notes(self) -> list[NoteSummary]:
        """
        Find notes with no incoming links.

        Uses single-pass iteration to avoid N+1 query pattern.

        Returns:
            List of orphan note summaries
        """
        # Build set of all linked-to note IDs and collect all notes in one pass
        linked_to: set[UUID] = set()
        all_parsed: list[tuple[ParsedNote, str | None]] = []

        for parsed, category in self.note_store.iter_all():
            all_parsed.append((parsed, category))
            # A note's links to itself are not incoming links, so exclude its
            # own id before recording the targets (mirrors _find_backlinks,
            # which skips parsed.id == note_id).
            outgoing = set(parsed.links) | set(extract_inline_links(parsed.body))
            outgoing.discard(parsed.id)
            linked_to.update(outgoing)

        # Find notes not in linked_to set using already-parsed data
        orphans = [
            _parsed_to_summary(parsed, category)
            for parsed, category in all_parsed
            if parsed.id not in linked_to
        ]

        return orphans

    def update_note_links(self, note_id: UUID) -> list[UUID]:
        """
        Update a note's frontmatter links based on [[uuid]] syntax in content.

        This is called after note updates to keep frontmatter in sync.

        Args:
            note_id: Note UUID

        Returns:
            List of link UUIDs found
        """
        try:
            note = self.note_store.read(note_id)
        except NoteNotFoundError:
            return []

        parsed = parse_note(note.content)
        inline_links = extract_inline_links(parsed.body)

        return inline_links

    def validate_link(self, target_id: UUID) -> bool:
        """
        Check if a link target exists.

        Args:
            target_id: Target note UUID

        Returns:
            True if target exists
        """
        return self.note_store.exists(target_id)
