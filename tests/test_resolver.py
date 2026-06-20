"""Tests for LinkResolver (link resolution and tracking)."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from mcp_notes.links.resolver import LinkResolver
from mcp_notes.models import BrokenLink, Note, NoteSummary
from mcp_notes.storage.filesystem import NoteNotFoundError
from mcp_notes.storage.parser import ParsedNote


def make_mock_note_store():
    """Create a mock NoteStore."""
    return MagicMock()


def make_parsed_note(
    note_id=None,
    title="Test Note",
    body="Body content",
    links=None,
) -> ParsedNote:
    """Create a ParsedNote for testing."""
    now = datetime.now(UTC)
    return ParsedNote(
        id=note_id or uuid4(),
        title=title,
        content=f"---\n...\n---\n\n{body}",
        tags=[],
        category=None,
        links=links or [],
        created=now,
        modified=now,
        raw_frontmatter={},
        body=body,
    )


def make_note(note_id=None, title="Test Note", body="Content", links=None, content=None) -> Note:
    """Create a Note model for testing with valid YAML frontmatter.

    If content is provided, uses it directly. Otherwise generates YAML frontmatter.
    """
    nid = note_id or uuid4()
    now = datetime.now(UTC)
    link_list = links or []

    if content is None:
        # Build YAML frontmatter
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        frontmatter_lines = [
            "---",
            f"id: {nid}",
            f"title: {title}",
            f"created: {now_str}",
            f"modified: {now_str}",
        ]
        if link_list:
            frontmatter_lines.append("links:")
            for link in link_list:
                frontmatter_lines.append(f"  - {link}")
        frontmatter_lines.append("---")
        frontmatter_lines.append("")
        frontmatter_lines.append(body)
        content = "\n".join(frontmatter_lines)

    return Note(
        id=nid,
        title=title,
        content=content,
        tags=[],
        category=None,
        links=link_list,
        created=now,
        modified=now,
    )


def make_note_summary(note_id=None, title="Test Note") -> NoteSummary:
    """Create a NoteSummary for testing."""
    now = datetime.now(UTC)
    return NoteSummary(
        id=note_id or uuid4(),
        title=title,
        tags=[],
        category=None,
        created=now,
        modified=now,
    )


class TestLinkResolverInit:
    """Tests for LinkResolver initialization."""

    def test_with_note_store(self):
        """Resolver accepts note_store parameter."""
        mock_store = make_mock_note_store()
        resolver = LinkResolver(note_store=mock_store)

        assert resolver.note_store is mock_store

    def test_without_note_store_creates_default(self):
        """Resolver creates default NoteStore if not provided."""
        with patch("mcp_notes.links.resolver.NoteStore") as MockStore:
            LinkResolver()

        MockStore.assert_called_once()


class TestGetNoteLinks:
    """Tests for get_note_links method."""

    def test_note_not_found(self):
        """Returns empty NoteLinks when note not found."""
        mock_store = make_mock_note_store()
        mock_store.read.side_effect = NoteNotFoundError("Not found")

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(uuid4())

        assert result.outgoing == []
        assert result.incoming == []
        assert result.broken == []

    def test_note_with_frontmatter_links(self):
        """Resolves outgoing links from frontmatter."""
        note_id = uuid4()
        target_id = uuid4()

        mock_store = make_mock_note_store()

        # Note content with frontmatter link
        note_content = f"""---
id: {note_id}
title: Test
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
links:
  - {target_id}
---

Body content"""

        mock_note = make_note(note_id=note_id, content=note_content)
        mock_store.read.return_value = mock_note

        target_summary = make_note_summary(note_id=target_id, title="Target Note")
        mock_store.get_summary.return_value = target_summary
        mock_store.iter_all.return_value = iter([])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(note_id)

        assert len(result.outgoing) == 1
        assert result.outgoing[0].id == target_id

    def test_note_with_inline_links(self):
        """Resolves outgoing links from [[uuid]] syntax."""
        note_id = uuid4()
        target_id = uuid4()

        mock_store = make_mock_note_store()

        note_content = f"""---
id: {note_id}
title: Test
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---

Link to [[{target_id}]] here."""

        mock_note = make_note(note_id=note_id, content=note_content)
        mock_store.read.return_value = mock_note

        target_summary = make_note_summary(note_id=target_id, title="Target Note")
        mock_store.get_summary.return_value = target_summary
        mock_store.iter_all.return_value = iter([])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(note_id)

        assert len(result.outgoing) == 1
        assert result.outgoing[0].id == target_id

    def test_broken_link_detected(self):
        """Broken links (non-existent targets) are detected."""
        note_id = uuid4()
        target_id = uuid4()

        mock_store = make_mock_note_store()

        note_content = f"""---
id: {note_id}
title: Test
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
links:
  - {target_id}
---

Body"""

        mock_note = make_note(note_id=note_id, content=note_content)
        mock_store.read.return_value = mock_note
        mock_store.get_summary.side_effect = NoteNotFoundError("Not found")
        mock_store.iter_all.return_value = iter([])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(note_id)

        assert len(result.broken) == 1
        assert result.broken[0] == target_id

    def test_combined_frontmatter_and_inline_links(self):
        """Both frontmatter and inline links are resolved."""
        note_id = uuid4()
        fm_target = uuid4()
        inline_target = uuid4()

        mock_store = make_mock_note_store()

        note_content = f"""---
id: {note_id}
title: Test
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
links:
  - {fm_target}
---

Link to [[{inline_target}]]"""

        mock_note = make_note(note_id=note_id, content=note_content)
        mock_store.read.return_value = mock_note

        def get_summary_impl(target_id):
            return make_note_summary(note_id=target_id)

        mock_store.get_summary.side_effect = get_summary_impl
        mock_store.iter_all.return_value = iter([])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(note_id)

        assert len(result.outgoing) == 2


class TestFindBacklinks:
    """Tests for backlink finding."""

    def test_finds_backlinks_from_frontmatter(self):
        """Finds notes that link via frontmatter."""
        target_id = uuid4()
        source_id = uuid4()

        mock_store = make_mock_note_store()

        # Target note (the one we're finding backlinks for)
        target_content = f"""---
id: {target_id}
title: Target
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---

Target body"""

        mock_note = make_note(note_id=target_id, content=target_content)
        mock_store.read.return_value = mock_note

        # Source note that links to target
        source_parsed = make_parsed_note(
            note_id=source_id,
            title="Source",
            links=[target_id],
        )

        mock_store.iter_all.return_value = iter([(source_parsed, None)])
        mock_store.get_summary.return_value = make_note_summary(
            note_id=source_id, title="Source"
        )

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(target_id)

        assert len(result.incoming) == 1
        assert result.incoming[0].id == source_id

    def test_finds_backlinks_from_inline(self):
        """Finds notes that link via [[uuid]] syntax."""
        target_id = uuid4()
        source_id = uuid4()

        mock_store = make_mock_note_store()

        target_content = f"""---
id: {target_id}
title: Target
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---

Target body"""

        mock_note = make_note(note_id=target_id, content=target_content)
        mock_store.read.return_value = mock_note

        source_parsed = make_parsed_note(
            note_id=source_id,
            title="Source",
            body=f"Links to [[{target_id}]]",
        )

        mock_store.iter_all.return_value = iter([(source_parsed, None)])
        mock_store.get_summary.return_value = make_note_summary(
            note_id=source_id, title="Source"
        )

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(target_id)

        assert len(result.incoming) == 1


class TestGetAllBrokenLinks:
    """Tests for get_all_broken_links method."""

    def test_empty_when_no_broken_links(self):
        """Returns empty list when all links are valid."""
        mock_store = make_mock_note_store()
        mock_store.iter_all.return_value = iter([])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_all_broken_links()

        assert result == []

    def test_finds_broken_links(self):
        """Finds all broken links across notes."""
        source_id = uuid4()
        broken_target = uuid4()

        mock_store = make_mock_note_store()

        source_parsed = make_parsed_note(
            note_id=source_id,
            title="Source",
            links=[broken_target],
        )

        mock_store.iter_all.return_value = iter([(source_parsed, None)])
        mock_store.exists.return_value = False  # Target doesn't exist

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_all_broken_links()

        assert len(result) == 1
        assert isinstance(result[0], BrokenLink)
        assert result[0].source_note_id == source_id
        assert result[0].broken_target_id == broken_target


class TestGetOrphanNotes:
    """Tests for get_orphan_notes method."""

    def test_empty_when_all_linked(self):
        """Returns empty when all notes have incoming links."""
        note1 = uuid4()
        note2 = uuid4()

        mock_store = make_mock_note_store()

        parsed1 = make_parsed_note(note_id=note1, links=[note2])
        parsed2 = make_parsed_note(note_id=note2, links=[note1])

        mock_store.iter_all.return_value = iter([(parsed1, None), (parsed2, None)])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_orphan_notes()

        assert result == []

    def test_finds_orphans(self):
        """Finds notes with no incoming links."""
        orphan_id = uuid4()
        linked_id = uuid4()

        mock_store = make_mock_note_store()

        orphan = make_parsed_note(note_id=orphan_id, links=[linked_id])
        linked = make_parsed_note(note_id=linked_id, links=[])

        # Two passes: first to collect links, second to find orphans
        mock_store.iter_all.side_effect = [
            iter([(orphan, None), (linked, None)]),
            iter([(orphan, None), (linked, None)]),
        ]
        mock_store.get_summary.return_value = make_note_summary(
            note_id=orphan_id, title="Orphan"
        )

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_orphan_notes()

        # orphan_id links to linked_id, so linked_id is not orphan
        # orphan_id has no incoming links, so it's an orphan
        assert len(result) == 1
        assert result[0].id == orphan_id

    def test_self_link_in_body_is_not_an_incoming_link(self):
        """A note linking to itself in its body is still an orphan when nothing
        else links to it. A self-reference is not a meaningful incoming link."""
        note_id = uuid4()
        mock_store = make_mock_note_store()
        self_note = make_parsed_note(note_id=note_id, body=f"see [[{note_id}]]")
        mock_store.iter_all.return_value = iter([(self_note, None)])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_orphan_notes()

        assert len(result) == 1
        assert result[0].id == note_id

    def test_self_link_in_frontmatter_is_not_an_incoming_link(self):
        """A note whose frontmatter links list contains its own id is still an
        orphan when nothing else links to it."""
        note_id = uuid4()
        mock_store = make_mock_note_store()
        self_note = make_parsed_note(note_id=note_id, links=[note_id])
        mock_store.iter_all.return_value = iter([(self_note, None)])

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_orphan_notes()

        assert len(result) == 1
        assert result[0].id == note_id


class TestValidateLink:
    """Tests for validate_link method."""

    def test_valid_link(self):
        """Returns True for existing target."""
        mock_store = make_mock_note_store()
        mock_store.exists.return_value = True

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.validate_link(uuid4())

        assert result is True

    def test_invalid_link(self):
        """Returns False for non-existing target."""
        mock_store = make_mock_note_store()
        mock_store.exists.return_value = False

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.validate_link(uuid4())

        assert result is False


class TestUpdateNoteLinks:
    """Tests for update_note_links method."""

    def test_note_not_found(self):
        """Returns empty list when note not found."""
        mock_store = make_mock_note_store()
        mock_store.read.side_effect = NoteNotFoundError("Not found")

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.update_note_links(uuid4())

        assert result == []

    def test_extracts_inline_links(self):
        """Extracts [[uuid]] links from content."""
        note_id = uuid4()
        link1 = uuid4()
        link2 = uuid4()

        mock_store = make_mock_note_store()

        note_content = f"""---
id: {note_id}
title: Test
created: 2024-01-01T00:00:00Z
modified: 2024-01-01T00:00:00Z
---

Links: [[{link1}]] and [[{link2}]]"""

        mock_note = make_note(note_id=note_id, content=note_content)
        mock_store.read.return_value = mock_note

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.update_note_links(note_id)

        assert len(result) == 2
        assert link1 in result
        assert link2 in result


class TestBacklinksExceptionHandling:
    """Tests for exception handling in backlinks and orphan detection."""

    def test_backlinks_uses_parsed_data_directly_frontmatter(self):
        """_find_backlinks uses parsed data directly (no N+1 queries) for frontmatter links."""
        target_id = uuid4()
        source_id = uuid4()

        mock_store = make_mock_note_store()

        # Source note links to target via frontmatter
        source_parsed = make_parsed_note(
            note_id=source_id,
            title="Source Note",
            links=[target_id],
        )
        # Target note exists (so get_note_links is called)
        target_parsed = make_parsed_note(
            note_id=target_id,
            title="Target Note",
        )
        target_note = make_note(note_id=target_id, title="Target Note")
        mock_store.read.return_value = target_note
        mock_store.iter_all.return_value = [(source_parsed, None), (target_parsed, None)]

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(target_id)

        # Backlinks are found using parsed data directly (no get_summary call)
        assert len(result.incoming) == 1
        assert result.incoming[0].id == source_id
        assert result.incoming[0].title == "Source Note"
        # get_summary should NOT be called for backlinks (N+1 fix)
        # It's only called for outgoing links

    def test_backlinks_uses_parsed_data_directly_inline(self):
        """_find_backlinks uses parsed data directly (no N+1 queries) for inline links."""
        target_id = uuid4()
        source_id = uuid4()

        mock_store = make_mock_note_store()

        # Source note links to target via inline [[uuid]] syntax
        source_parsed = make_parsed_note(
            note_id=source_id,
            title="Source Note",
            body=f"Check out [[{target_id}]]",
            links=[],  # No frontmatter links
        )
        # Target note exists
        target_parsed = make_parsed_note(
            note_id=target_id,
            title="Target Note",
        )
        target_note = make_note(note_id=target_id, title="Target Note")
        mock_store.read.return_value = target_note
        mock_store.iter_all.return_value = [(source_parsed, None), (target_parsed, None)]

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(target_id)

        # Backlinks are found using parsed data directly
        assert len(result.incoming) == 1
        assert result.incoming[0].id == source_id
        assert result.incoming[0].title == "Source Note"

    def test_orphan_notes_uses_parsed_data_directly(self):
        """get_orphan_notes uses parsed data directly (no N+1 queries)."""
        orphan_id = uuid4()

        mock_store = make_mock_note_store()

        # Orphan note with no links
        orphan_parsed = make_parsed_note(
            note_id=orphan_id,
            title="Orphan Note",
            body="No links here",
        )
        mock_store.iter_all.return_value = [(orphan_parsed, None)]

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_orphan_notes()

        # Orphan found using parsed data directly (no get_summary call)
        assert len(result) == 1
        assert result[0].id == orphan_id
        assert result[0].title == "Orphan Note"

    def test_backlinks_finds_all_linking_notes(self):
        """_find_backlinks finds all notes that link to target using parsed data."""
        target_id = uuid4()
        source1_id = uuid4()
        source2_id = uuid4()

        mock_store = make_mock_note_store()

        # Two notes link to target via frontmatter
        source1_parsed = make_parsed_note(
            note_id=source1_id,
            title="Source 1",
            links=[target_id],
        )
        source2_parsed = make_parsed_note(
            note_id=source2_id,
            title="Source 2",
            links=[target_id],
        )
        target_parsed = make_parsed_note(
            note_id=target_id,
            title="Target",
        )
        target_note = make_note(note_id=target_id, title="Target")
        mock_store.read.return_value = target_note
        mock_store.iter_all.return_value = [(source1_parsed, None), (source2_parsed, None), (target_parsed, None)]

        resolver = LinkResolver(note_store=mock_store)
        result = resolver.get_note_links(target_id)

        # Both backlinks found using parsed data directly (no get_summary calls)
        assert len(result.incoming) == 2
        ids = {s.id for s in result.incoming}
        assert source1_id in ids
        assert source2_id in ids
