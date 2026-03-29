"""Tests for facts source integrity."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from vector_core.facts import (
    FactSource,
    FactStore,
    IntegrityCheckResult,
    SourceIntegrityManager,
    SourceStatus,
    SourceType,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir: Path):
    """Create a temporary FactStore."""
    db_path = temp_dir / "facts.db"
    return FactStore(db_path=db_path)


@pytest.fixture
def integrity_manager(store: FactStore):
    """Create a SourceIntegrityManager."""
    return SourceIntegrityManager(fact_store=store)


class TestIntegrityCheckResult:
    """Tests for IntegrityCheckResult."""

    def test_to_dict(self) -> None:
        """to_dict returns correct structure."""
        result = IntegrityCheckResult(
            total_sources=10,
            active_sources=8,
            deleted_sources=1,
            modified_sources=1,
            relocated_sources=0,
        )

        d = result.to_dict()

        assert d["total_sources"] == 10
        assert d["active_sources"] == 8
        assert d["deleted_sources"] == 1
        assert d["modified_sources"] == 1
        assert d["relocated_sources"] == 0
        assert d["integrity_score"] == 0.8

    def test_to_dict_zero_total(self) -> None:
        """to_dict handles zero total sources."""
        result = IntegrityCheckResult(
            total_sources=0,
            active_sources=0,
            deleted_sources=0,
            modified_sources=0,
            relocated_sources=0,
        )

        d = result.to_dict()

        assert d["integrity_score"] == 1.0


class TestMarkNoteDeleted:
    """Tests for mark_note_deleted."""

    def test_mark_note_deleted_no_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marking nonexistent note sources returns 0."""
        note_id = uuid4()
        count = integrity_manager.mark_note_deleted(note_id)
        assert count == 0

    def test_mark_note_deleted_with_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marks all note sources as deleted."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Mark as deleted
        count = integrity_manager.mark_note_deleted(note_id)

        assert count == 1

        # Verify status changed
        updated = store.read(fact.id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0].status == SourceStatus.DELETED


class TestMarkNoteModified:
    """Tests for mark_note_modified."""

    def test_mark_note_modified_no_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marking nonexistent note sources returns 0."""
        note_id = uuid4()
        count = integrity_manager.mark_note_modified(note_id)
        assert count == 0

    def test_mark_note_modified_with_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marks all note sources as modified."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Mark as modified
        count = integrity_manager.mark_note_modified(note_id)

        assert count == 1

        # Verify status changed
        updated = store.read(fact.id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0].status == SourceStatus.MODIFIED


class TestMarkGlossaryDeleted:
    """Tests for mark_glossary_deleted."""

    def test_mark_glossary_deleted_no_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marking nonexistent glossary sources returns 0."""
        entry_id = uuid4()
        count = integrity_manager.mark_glossary_deleted(entry_id)
        assert count == 0

    def test_mark_glossary_deleted_with_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marks all glossary sources as deleted."""
        entry_id = uuid4()

        # Create fact with glossary source
        source = FactSource(
            source_type=SourceType.GLOSSARY,
            source_id=entry_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Glossary",
            source=source,
        )

        # Mark as deleted
        count = integrity_manager.mark_glossary_deleted(entry_id)

        assert count == 1

        # Verify status changed
        updated = store.read(fact.id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0].status == SourceStatus.DELETED


class TestMarkDocumentModified:
    """Tests for mark_document_modified."""

    def test_mark_document_modified_no_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marking nonexistent document hash returns 0."""
        count = integrity_manager.mark_document_modified("nonexistent_hash")
        assert count == 0

    def test_mark_document_modified_with_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marks all document sources by hash as modified."""
        content_hash = "abc123def456"

        # Create fact with document source
        source = FactSource(
            source_type=SourceType.DOCUMENT,
            source_path="/path/to/doc.pdf",
            content_hash=content_hash,
        )
        fact = store.create(
            subject="Test",
            predicate="from",
            object_value="Document",
            source=source,
        )

        # Mark as modified
        count = integrity_manager.mark_document_modified(content_hash)

        assert count == 1

        # Verify status changed
        updated = store.read(fact.id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0].status == SourceStatus.MODIFIED


class TestMarkDocumentDeleted:
    """Tests for mark_document_deleted."""

    def test_mark_document_deleted_with_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marks all document sources by hash as deleted."""
        content_hash = "xyz789"

        # Create fact with document source
        source = FactSource(
            source_type=SourceType.DOCUMENT,
            source_path="/path/to/doc.pdf",
            content_hash=content_hash,
        )
        fact = store.create(
            subject="Test",
            predicate="from",
            object_value="Document",
            source=source,
        )

        # Mark as deleted
        count = integrity_manager.mark_document_deleted(content_hash)

        assert count == 1

        # Verify status changed
        updated = store.read(fact.id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0].status == SourceStatus.DELETED


class TestGetFactsWithDeletedSources:
    """Tests for get_facts_with_deleted_sources."""

    def test_get_facts_with_deleted_sources_empty(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns empty list when no deleted sources."""
        # Create fact with active source
        source = FactSource(source_type=SourceType.MANUAL)
        store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        facts = integrity_manager.get_facts_with_deleted_sources()
        assert facts == []

    def test_get_facts_with_deleted_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns facts with deleted sources."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Mark as deleted
        integrity_manager.mark_note_deleted(note_id)

        # Get facts with deleted sources
        facts = integrity_manager.get_facts_with_deleted_sources()

        assert len(facts) == 1
        assert facts[0].id == fact.id


class TestGetFactsWithModifiedSources:
    """Tests for get_facts_with_modified_sources."""

    def test_get_facts_with_modified_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns facts with modified sources."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Mark as modified
        integrity_manager.mark_note_modified(note_id)

        # Get facts with modified sources
        facts = integrity_manager.get_facts_with_modified_sources()

        assert len(facts) == 1
        assert facts[0].id == fact.id


class TestCheckFactIntegrity:
    """Tests for check_fact_integrity."""

    def test_check_nonexistent_fact(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns zero counts for nonexistent fact."""
        result = integrity_manager.check_fact_integrity(uuid4())

        assert result.total_sources == 0
        assert result.active_sources == 0

    def test_check_fact_all_active(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns correct counts for all active sources."""
        source = FactSource(source_type=SourceType.MANUAL)
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        result = integrity_manager.check_fact_integrity(fact.id)

        assert result.total_sources == 1
        assert result.active_sources == 1
        assert result.deleted_sources == 0
        assert result.modified_sources == 0

    def test_check_fact_mixed_status(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns correct counts for mixed source statuses."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Add another source (manual)
        store.add_source(fact.id, FactSource(source_type=SourceType.MANUAL))

        # Mark note source as deleted
        integrity_manager.mark_note_deleted(note_id)

        result = integrity_manager.check_fact_integrity(fact.id)

        assert result.total_sources == 2
        assert result.active_sources == 1
        assert result.deleted_sources == 1


class TestGetSourceStatistics:
    """Tests for get_source_statistics."""

    def test_get_source_statistics_empty(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns empty stats when no sources."""
        stats = integrity_manager.get_source_statistics()

        assert stats["total_sources"] == 0
        assert stats["by_status"] == {}
        assert stats["by_type"] == {}

    def test_get_source_statistics(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Returns correct statistics."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        store.create(
            subject="Test1",
            predicate="has",
            object_value="Source1",
            source=source,
        )

        # Create fact with manual source
        store.create(
            subject="Test2",
            predicate="has",
            object_value="Source2",
            source=FactSource(source_type=SourceType.MANUAL),
        )

        # Mark note source as deleted
        integrity_manager.mark_note_deleted(note_id)

        stats = integrity_manager.get_source_statistics()

        assert stats["total_sources"] == 2
        assert stats["by_status"].get("active", 0) == 1
        assert stats["by_status"].get("deleted", 0) == 1
        assert stats["by_type"].get("note", 0) == 1
        assert stats["by_type"].get("manual", 0) == 1


class TestRevalidateSources:
    """Tests for revalidate_sources."""

    def test_revalidate_sources(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Resets sources back to active."""
        note_id = uuid4()

        # Create fact with note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )
        fact = store.create(
            subject="Test",
            predicate="has",
            object_value="Source",
            source=source,
        )

        # Mark as deleted
        integrity_manager.mark_note_deleted(note_id)

        # Verify deleted
        updated = store.read(fact.id)
        assert updated.sources[0].status == SourceStatus.DELETED

        # Revalidate
        count = integrity_manager.revalidate_sources(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )

        assert count == 1

        # Verify back to active
        revalidated = store.read(fact.id)
        assert revalidated.sources[0].status == SourceStatus.ACTIVE


class TestMultipleFactsWithSameSource:
    """Tests for marking sources across multiple facts."""

    def test_mark_affects_all_facts(
        self, store: FactStore, integrity_manager: SourceIntegrityManager
    ) -> None:
        """Marking a source affects all facts referencing it."""
        note_id = uuid4()

        # Create multiple facts with same note source
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
        )

        fact1 = store.create(
            subject="Test1",
            predicate="from",
            object_value="Note",
            source=source,
        )

        fact2 = store.create(
            subject="Test2",
            predicate="from",
            object_value="Note",
            source=source,
        )

        # Mark note as deleted
        count = integrity_manager.mark_note_deleted(note_id)

        assert count == 2

        # Verify both facts' sources are deleted
        updated1 = store.read(fact1.id)
        updated2 = store.read(fact2.id)

        assert updated1.sources[0].status == SourceStatus.DELETED
        assert updated2.sources[0].status == SourceStatus.DELETED
