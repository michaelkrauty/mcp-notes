"""Tests for facts subsystem."""

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mcp_notes.facts import (
    DuplicateFactError,
    Fact,
    FactNotFoundError,
    FactSource,
    FactStore,
    FactSummary,
    SourceStatus,
    SourceType,
    compute_spo_hash,
)


class TestComputeSPOHash:
    """Tests for compute_spo_hash function."""

    def test_deterministic(self):
        """Same input produces same output."""
        hash1 = compute_spo_hash("John", "person", "works_at", "Acme", "organization")
        hash2 = compute_spo_hash("John", "person", "works_at", "Acme", "organization")
        assert hash1 == hash2

    def test_case_insensitive(self):
        """Hash is case-insensitive."""
        hash1 = compute_spo_hash("John", "person", "works_at", "Acme", "organization")
        hash2 = compute_spo_hash("JOHN", "PERSON", "WORKS_AT", "ACME", "ORGANIZATION")
        assert hash1 == hash2

    def test_different_values_different_hash(self):
        """Different values produce different hashes."""
        hash1 = compute_spo_hash("John", "person", "works_at", "Acme", "organization")
        hash2 = compute_spo_hash("Jane", "person", "works_at", "Acme", "organization")
        assert hash1 != hash2

    def test_type_matters(self):
        """Different types produce different hashes."""
        hash1 = compute_spo_hash("Python", "language", "created_by", "Guido", "person")
        hash2 = compute_spo_hash("Python", "snake", "created_by", "Guido", "person")
        assert hash1 != hash2


class TestFactSource:
    """Tests for FactSource model."""

    def test_create_note_source(self):
        """Create note source."""
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=uuid4(),
            location="section: History",
        )
        assert source.source_type == SourceType.NOTE
        assert source.status == SourceStatus.ACTIVE

    def test_create_document_source(self):
        """Create document source with hash."""
        source = FactSource(
            source_type=SourceType.DOCUMENT,
            source_path="/docs/manual.pdf",
            content_hash="abc123",
            location="page 3",
        )
        assert source.source_type == SourceType.DOCUMENT
        assert source.content_hash == "abc123"

    def test_to_dict(self):
        """Convert source to dict."""
        source_id = uuid4()
        now = datetime.now(UTC)
        source = FactSource(
            source_type=SourceType.GLOSSARY,
            source_id=source_id,
            extracted_at=now,
        )
        d = source.to_dict()
        assert d["source_type"] == "glossary"
        assert d["source_id"] == str(source_id)


class TestFact:
    """Tests for Fact model."""

    def test_to_dict(self):
        """Convert fact to dict."""
        fact_id = uuid4()
        now = datetime.now(UTC)
        fact = Fact(
            id=fact_id,
            subject="John",
            subject_type="person",
            predicate="works_at",
            object_value="Acme",
            object_type="organization",
            context="as engineer",
            confidence=0.9,
            valid_from=date(2020, 1, 1),
            valid_to=None,
            spo_hash="abc",
            created=now,
            modified=now,
        )
        d = fact.to_dict()
        assert d["id"] == str(fact_id)
        assert d["subject"] == "John"
        assert d["object"] == "Acme"  # Note: object_value maps to "object" in dict
        assert d["confidence"] == 0.9
        assert d["valid_from"] == "2020-01-01"


class TestFactStoreInit:
    """Tests for FactStore initialization."""

    def test_init_creates_db(self, tmp_path):
        """Creates database on init."""
        db_path = tmp_path / "test_facts.db"
        store = FactStore(db_path=db_path)
        assert db_path.exists()
        store.close()

    def test_init_creates_tables(self, tmp_path):
        """Creates required tables."""
        db_path = tmp_path / "test_facts.db"
        store = FactStore(db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "facts" in tables
        assert "fact_sources" in tables
        assert "entity_adjacency" in tables
        store.close()


class TestFactStoreCreate:
    """Tests for FactStore.create method."""

    def test_create_basic_fact(self, tmp_path):
        """Create basic fact."""
        store = FactStore(db_path=tmp_path / "facts.db")
        fact = store.create(
            subject="John Smith",
            predicate="works_at",
            object_value="Acme Corp",
        )

        assert fact.subject == "John Smith"
        assert fact.predicate == "works_at"
        assert fact.object_value == "Acme Corp"
        assert fact.subject_type == "entity"  # Default
        assert fact.object_type == "entity"  # Default
        assert fact.confidence == 1.0  # Default
        assert fact.spo_hash is not None
        store.close()

    def test_create_fact_with_types(self, tmp_path):
        """Create fact with custom types."""
        store = FactStore(db_path=tmp_path / "facts.db")
        fact = store.create(
            subject="John Smith",
            subject_type="person",
            predicate="served_in",
            object_value="101st Airborne",
            object_type="military_unit",
        )

        assert fact.subject_type == "person"
        assert fact.object_type == "military_unit"
        store.close()

    def test_create_fact_with_metadata(self, tmp_path):
        """Create fact with all metadata."""
        store = FactStore(db_path=tmp_path / "facts.db")
        fact = store.create(
            subject="John Smith",
            predicate="served_in",
            object_value="101st Airborne",
            context="as squad leader",
            confidence=0.9,
            valid_from=date(1968, 3, 15),
            valid_to=date(1970, 9, 1),
        )

        assert fact.context == "as squad leader"
        assert fact.confidence == 0.9
        assert fact.valid_from == date(1968, 3, 15)
        assert fact.valid_to == date(1970, 9, 1)
        store.close()

    def test_create_fact_with_source(self, tmp_path):
        """Create fact with source."""
        store = FactStore(db_path=tmp_path / "facts.db")
        note_id = uuid4()
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=note_id,
            location="paragraph 3",
        )
        fact = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
            source=source,
        )

        assert len(fact.sources) == 1
        assert fact.sources[0].source_type == SourceType.NOTE
        assert fact.sources[0].source_id == note_id
        store.close()

    def test_create_duplicate_raises_error(self, tmp_path):
        """Creating duplicate fact raises error."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        with pytest.raises(DuplicateFactError):
            store.create(
                subject="John",
                predicate="knows",
                object_value="Jane",
            )
        store.close()

    def test_create_duplicate_case_insensitive(self, tmp_path):
        """Duplicate detection is case-insensitive."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        with pytest.raises(DuplicateFactError):
            store.create(
                subject="JOHN",
                predicate="KNOWS",
                object_value="JANE",
            )
        store.close()

    def test_create_populates_adjacency(self, tmp_path):
        """Create populates entity adjacency table."""
        store = FactStore(db_path=tmp_path / "facts.db")
        fact = store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )

        # Query adjacency directly
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "facts.db"))
        cursor = conn.execute(
            "SELECT entity_name, entity_type, role FROM entity_adjacency WHERE fact_id = ?",
            (str(fact.id),),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 2
        entities = {(row[0], row[2]) for row in rows}
        assert ("john", "subject") in entities
        assert ("jane", "object") in entities
        store.close()


class TestFactStoreRead:
    """Tests for FactStore.read method."""

    def test_read_existing(self, tmp_path):
        """Read existing fact."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        fact = store.read(created.id)
        assert fact.id == created.id
        assert fact.subject == "John"
        store.close()

    def test_read_not_found(self, tmp_path):
        """Read non-existent fact raises error."""
        store = FactStore(db_path=tmp_path / "facts.db")

        with pytest.raises(FactNotFoundError):
            store.read(uuid4())
        store.close()

    def test_read_with_sources(self, tmp_path):
        """Read fact includes sources."""
        store = FactStore(db_path=tmp_path / "facts.db")
        source = FactSource(
            source_type=SourceType.NOTE,
            source_id=uuid4(),
        )
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
            source=source,
        )

        fact = store.read(created.id)
        assert len(fact.sources) == 1
        assert fact.sources[0].source_type == SourceType.NOTE
        store.close()


class TestFactStoreUpdate:
    """Tests for FactStore.update method."""

    def test_update_context(self, tmp_path):
        """Update context field."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="works_at",
            object_value="Acme",
        )

        updated = store.update(created.id, context="as engineer")
        assert updated.context == "as engineer"

        # Verify persistence
        fact = store.read(created.id)
        assert fact.context == "as engineer"
        store.close()

    def test_update_confidence(self, tmp_path):
        """Update confidence field."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="works_at",
            object_value="Acme",
            confidence=0.5,
        )

        updated = store.update(created.id, confidence=0.9)
        assert updated.confidence == 0.9
        store.close()

    def test_update_validity_dates(self, tmp_path):
        """Update validity dates."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="works_at",
            object_value="Acme",
        )

        updated = store.update(
            created.id,
            valid_from=date(2020, 1, 1),
            valid_to=date(2023, 12, 31),
        )
        assert updated.valid_from == date(2020, 1, 1)
        assert updated.valid_to == date(2023, 12, 31)
        store.close()

    def test_update_clears_field(self, tmp_path):
        """Update with None clears field."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="works_at",
            object_value="Acme",
            context="original context",
        )

        updated = store.update(created.id, context=None)
        assert updated.context is None
        store.close()

    def test_update_preserves_unspecified(self, tmp_path):
        """Update preserves unspecified fields."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="works_at",
            object_value="Acme",
            context="original",
            confidence=0.8,
        )

        # Only update confidence
        updated = store.update(created.id, confidence=0.9)
        assert updated.context == "original"  # Preserved
        assert updated.confidence == 0.9  # Updated
        store.close()

    def test_update_not_found(self, tmp_path):
        """Update non-existent fact raises error."""
        store = FactStore(db_path=tmp_path / "facts.db")

        with pytest.raises(FactNotFoundError):
            store.update(uuid4(), context="test")
        store.close()


class TestFactStoreDelete:
    """Tests for FactStore.delete method."""

    def test_delete_existing(self, tmp_path):
        """Delete existing fact."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        result = store.delete(created.id)
        assert result is True

        with pytest.raises(FactNotFoundError):
            store.read(created.id)
        store.close()

    def test_delete_not_found(self, tmp_path):
        """Delete non-existent fact returns False."""
        store = FactStore(db_path=tmp_path / "facts.db")
        result = store.delete(uuid4())
        assert result is False
        store.close()

    def test_delete_cascades_sources(self, tmp_path):
        """Delete cascades to sources."""
        store = FactStore(db_path=tmp_path / "facts.db")
        source = FactSource(source_type=SourceType.NOTE, source_id=uuid4())
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
            source=source,
        )

        store.delete(created.id)

        # Check sources are deleted
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "facts.db"))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM fact_sources WHERE fact_id = ?",
            (str(created.id),),
        )
        assert cursor.fetchone()[0] == 0
        conn.close()
        store.close()

    def test_delete_cleans_adjacency(self, tmp_path):
        """Delete cleans entity adjacency."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        store.delete(created.id)

        # Check adjacency is cleaned
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "facts.db"))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM entity_adjacency WHERE fact_id = ?",
            (str(created.id),),
        )
        assert cursor.fetchone()[0] == 0
        conn.close()
        store.close()


class TestFactStoreAddSource:
    """Tests for FactStore.add_source method."""

    def test_add_source(self, tmp_path):
        """Add source to existing fact."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
        )

        source = FactSource(
            source_type=SourceType.DOCUMENT,
            source_path="/docs/report.pdf",
            content_hash="abc123",
        )
        updated = store.add_source(created.id, source)

        assert len(updated.sources) == 1
        assert updated.sources[0].source_type == SourceType.DOCUMENT
        store.close()

    def test_add_multiple_sources(self, tmp_path):
        """Add multiple sources to fact."""
        store = FactStore(db_path=tmp_path / "facts.db")
        source1 = FactSource(source_type=SourceType.NOTE, source_id=uuid4())
        created = store.create(
            subject="John",
            predicate="knows",
            object_value="Jane",
            source=source1,
        )

        source2 = FactSource(source_type=SourceType.DOCUMENT, content_hash="abc")
        updated = store.add_source(created.id, source2)

        assert len(updated.sources) == 2
        store.close()


class TestFactStoreQuery:
    """Tests for FactStore.query method."""

    def test_query_by_subject(self, tmp_path):
        """Query by subject."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="John", predicate="works_with", object_value="Bob")
        store.create(subject="Alice", predicate="knows", object_value="Bob")

        results = store.query(subject="John")
        assert len(results) == 2
        assert all(f.subject == "John" for f in results)
        store.close()

    def test_query_by_predicate(self, tmp_path):
        """Query by predicate."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="John", predicate="works_with", object_value="Bob")
        store.create(subject="Alice", predicate="knows", object_value="Bob")

        results = store.query(predicate="knows")
        assert len(results) == 2
        store.close()

    def test_query_by_object(self, tmp_path):
        """Query by object."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Bob")
        store.create(subject="Alice", predicate="knows", object_value="Bob")
        store.create(subject="Alice", predicate="knows", object_value="Jane")

        results = store.query(object_value="Bob")
        assert len(results) == 2
        store.close()

    def test_query_by_type(self, tmp_path):
        """Query by subject/object type."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John", subject_type="person",
            predicate="works_at",
            object_value="Acme", object_type="organization",
        )
        store.create(
            subject="Jane", subject_type="person",
            predicate="works_at",
            object_value="Globex", object_type="organization",
        )
        store.create(
            subject="Python", subject_type="language",
            predicate="created_by",
            object_value="Guido", object_type="person",
        )

        results = store.query(subject_type="person")
        assert len(results) == 2
        store.close()

    def test_query_by_min_confidence(self, tmp_path):
        """Query by minimum confidence."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="A", predicate="p", object_value="B", confidence=0.5)
        store.create(subject="C", predicate="p", object_value="D", confidence=0.9)
        store.create(subject="E", predicate="p", object_value="F", confidence=0.7)

        results = store.query(min_confidence=0.8)
        assert len(results) == 1
        assert results[0].confidence == 0.9
        store.close()

    def test_query_by_valid_at(self, tmp_path):
        """Query by validity date."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John", predicate="worked_at", object_value="Acme",
            valid_from=date(2020, 1, 1), valid_to=date(2022, 12, 31),
        )
        store.create(
            subject="John", predicate="works_at", object_value="Globex",
            valid_from=date(2023, 1, 1), valid_to=None,
        )
        store.create(
            subject="Jane", predicate="works_at", object_value="Corp",
            valid_from=None, valid_to=None,  # Always valid
        )

        # Query for 2021
        results = store.query(valid_at=date(2021, 6, 1))
        subjects = {f.subject for f in results}
        assert "John" in subjects  # First John fact is valid
        assert "Jane" in subjects  # Always valid

        # Query for 2024
        results = store.query(valid_at=date(2024, 1, 1))
        assert len(results) == 2  # John at Globex + Jane
        store.close()

    def test_query_case_insensitive(self, tmp_path):
        """Query is case-insensitive for subject/predicate/object."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John Smith", predicate="KNOWS", object_value="Jane")

        results = store.query(subject="john smith")
        assert len(results) == 1

        results = store.query(predicate="knows")
        assert len(results) == 1
        store.close()

    def test_query_combined_filters(self, tmp_path):
        """Query with multiple filters."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John", subject_type="person",
            predicate="works_at",
            object_value="Acme", object_type="organization",
            confidence=0.9,
        )
        store.create(
            subject="Jane", subject_type="person",
            predicate="works_at",
            object_value="Acme", object_type="organization",
            confidence=0.5,
        )

        results = store.query(
            subject_type="person",
            predicate="works_at",
            min_confidence=0.8,
        )
        assert len(results) == 1
        assert results[0].subject == "John"
        store.close()

    def test_query_limit(self, tmp_path):
        """Query respects limit."""
        store = FactStore(db_path=tmp_path / "facts.db")
        for i in range(10):
            store.create(subject=f"Entity{i}", predicate="p", object_value="B")

        results = store.query(limit=5)
        assert len(results) == 5
        store.close()


class TestFactStoreGetEntityFacts:
    """Tests for FactStore.get_entity_facts method."""

    def test_get_entity_as_subject(self, tmp_path):
        """Get facts where entity is subject."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="John", predicate="works_at", object_value="Acme")
        store.create(subject="Alice", predicate="knows", object_value="John")

        results = store.get_entity_facts("John")
        assert len(results) == 3  # Subject in 2, object in 1
        store.close()

    def test_get_entity_as_object(self, tmp_path):
        """Get facts where entity is object."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="Alice", predicate="knows", object_value="Bob")
        store.create(subject="Carol", predicate="knows", object_value="Bob")

        results = store.get_entity_facts("Bob")
        assert len(results) == 2
        store.close()

    def test_get_entity_case_insensitive(self, tmp_path):
        """Entity lookup is case-insensitive."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John Smith", predicate="knows", object_value="Jane")

        results = store.get_entity_facts("john smith")
        assert len(results) == 1
        store.close()

    def test_get_entity_with_type(self, tmp_path):
        """Filter by entity type."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="Python", subject_type="language",
            predicate="created_by",
            object_value="Guido", object_type="person",
        )
        store.create(
            subject="Python", subject_type="snake",
            predicate="is_a",
            object_value="Reptile", object_type="class",
        )

        # Get only language facts
        results = store.get_entity_facts("Python", entity_type="language")
        assert len(results) == 1
        assert results[0].subject_type == "language"
        store.close()


class TestFactStoreListSummaries:
    """Tests for FactStore.list_summaries method."""

    def test_list_summaries(self, tmp_path):
        """List facts as summaries."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="Alice", predicate="works_at", object_value="Acme")

        summaries = store.list_summaries()
        assert len(summaries) == 2
        assert all(isinstance(s, FactSummary) for s in summaries)
        store.close()

    def test_list_summaries_includes_source_count(self, tmp_path):
        """Summaries include source count."""
        store = FactStore(db_path=tmp_path / "facts.db")
        source = FactSource(source_type=SourceType.NOTE, source_id=uuid4())
        fact = store.create(
            subject="John", predicate="knows", object_value="Jane",
            source=source,
        )
        store.add_source(fact.id, FactSource(source_type=SourceType.DOCUMENT, content_hash="x"))

        summaries = store.list_summaries()
        assert summaries[0].source_count == 2
        store.close()

    def test_list_summaries_with_filters(self, tmp_path):
        """List summaries with filters."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="John", subject_type="person",
            predicate="works_at",
            object_value="Acme", object_type="organization",
        )
        store.create(
            subject="Python", subject_type="language",
            predicate="created_by",
            object_value="Guido", object_type="person",
        )

        summaries = store.list_summaries(subject_type="person")
        assert len(summaries) == 1
        assert summaries[0].subject == "John"
        store.close()


class TestFactStoreFindBySPOHash:
    """Tests for FactStore.find_by_spo_hash method."""

    def test_find_existing(self, tmp_path):
        """Find fact by SPO hash."""
        store = FactStore(db_path=tmp_path / "facts.db")
        created = store.create(
            subject="John",
            subject_type="person",
            predicate="knows",
            object_value="Jane",
            object_type="person",
        )

        spo_hash = compute_spo_hash("John", "person", "knows", "Jane", "person")
        found = store.find_by_spo_hash(spo_hash)
        assert found is not None
        assert found.id == created.id
        store.close()

    def test_find_not_found(self, tmp_path):
        """Find returns None if not found."""
        store = FactStore(db_path=tmp_path / "facts.db")
        spo_hash = compute_spo_hash("X", "x", "y", "Z", "z")
        found = store.find_by_spo_hash(spo_hash)
        assert found is None
        store.close()


class TestFactStoreGetFactsBySource:
    """Tests for FactStore.get_facts_by_source method."""

    def test_get_by_source_type(self, tmp_path):
        """Get facts by source type."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="A", predicate="p", object_value="B",
            source=FactSource(source_type=SourceType.NOTE, source_id=uuid4()),
        )
        store.create(
            subject="C", predicate="p", object_value="D",
            source=FactSource(source_type=SourceType.DOCUMENT, content_hash="x"),
        )

        results = store.get_facts_by_source(source_type=SourceType.NOTE)
        assert len(results) == 1
        assert results[0].subject == "A"
        store.close()

    def test_get_by_source_id(self, tmp_path):
        """Get facts by source ID."""
        store = FactStore(db_path=tmp_path / "facts.db")
        note_id = uuid4()
        store.create(
            subject="A", predicate="p", object_value="B",
            source=FactSource(source_type=SourceType.NOTE, source_id=note_id),
        )
        store.create(
            subject="C", predicate="p", object_value="D",
            source=FactSource(source_type=SourceType.NOTE, source_id=uuid4()),
        )

        results = store.get_facts_by_source(source_id=note_id)
        assert len(results) == 1
        assert results[0].subject == "A"
        store.close()

    def test_get_by_content_hash(self, tmp_path):
        """Get facts by content hash."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="A", predicate="p", object_value="B",
            source=FactSource(source_type=SourceType.DOCUMENT, content_hash="abc123"),
        )
        store.create(
            subject="C", predicate="p", object_value="D",
            source=FactSource(source_type=SourceType.DOCUMENT, content_hash="xyz789"),
        )

        results = store.get_facts_by_source(content_hash="abc123")
        assert len(results) == 1
        store.close()


class TestFactStoreUpdateSourceStatus:
    """Tests for FactStore.update_source_status method."""

    def test_update_status(self, tmp_path):
        """Update source status."""
        store = FactStore(db_path=tmp_path / "facts.db")
        note_id = uuid4()
        fact = store.create(
            subject="A", predicate="p", object_value="B",
            source=FactSource(source_type=SourceType.NOTE, source_id=note_id),
        )

        count = store.update_source_status(
            source_type=SourceType.NOTE,
            source_id=note_id,
            new_status=SourceStatus.DELETED,
        )
        assert count == 1

        # Verify
        updated = store.read(fact.id)
        assert updated.sources[0].status == SourceStatus.DELETED
        store.close()

    def test_update_by_content_hash(self, tmp_path):
        """Update status by content hash."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="A", predicate="p", object_value="B",
            source=FactSource(source_type=SourceType.DOCUMENT, content_hash="abc"),
        )
        store.create(
            subject="C", predicate="p", object_value="D",
            source=FactSource(source_type=SourceType.DOCUMENT, content_hash="abc"),
        )

        count = store.update_source_status(
            source_type=SourceType.DOCUMENT,
            content_hash="abc",
            new_status=SourceStatus.MODIFIED,
        )
        assert count == 2
        store.close()


class TestFactStoreCount:
    """Tests for FactStore.count method."""

    def test_count_empty(self, tmp_path):
        """Count empty store."""
        store = FactStore(db_path=tmp_path / "facts.db")
        assert store.count() == 0
        store.close()

    def test_count_with_facts(self, tmp_path):
        """Count with facts."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="A", predicate="p", object_value="B")
        store.create(subject="C", predicate="p", object_value="D")
        assert store.count() == 2
        store.close()


class TestFactStoreIterAll:
    """Tests for FactStore.iter_all method."""

    def test_iter_all(self, tmp_path):
        """Iterate all facts."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="A", predicate="p", object_value="B")
        store.create(subject="C", predicate="p", object_value="D")

        facts = list(store.iter_all())
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)
        store.close()


class TestFactStoreContextManager:
    """Tests for FactStore context manager."""

    def test_context_manager(self, tmp_path):
        """Use store as context manager."""
        with FactStore(db_path=tmp_path / "facts.db") as store:
            store.create(subject="A", predicate="p", object_value="B")
            assert store.count() == 1


class TestFactStoreConcurrency:
    """Tests for thread safety."""

    def test_concurrent_reads(self, tmp_path):
        """Concurrent reads work correctly."""
        import threading

        store = FactStore(db_path=tmp_path / "facts.db")
        fact = store.create(subject="A", predicate="p", object_value="B")

        results = []
        errors = []

        def read_fact():
            try:
                f = store.read(fact.id)
                results.append(f)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_fact) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        store.close()


class TestFactStoreFindConnections:
    """Tests for FactStore.find_connections BFS graph traversal."""

    def test_find_direct_connection(self, tmp_path):
        """Find direct single-hop connection."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")

        paths = store.find_connections("John", "Jane")
        assert len(paths) == 1
        assert len(paths[0]) == 1  # Single fact path
        assert paths[0][0].subject == "John"
        assert paths[0][0].object_value == "Jane"
        store.close()

    def test_find_two_hop_connection(self, tmp_path):
        """Find 2-hop connection through intermediate entity."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Alice")
        store.create(subject="Alice", predicate="knows", object_value="Bob")

        paths = store.find_connections("John", "Bob", max_depth=2)
        assert len(paths) >= 1
        assert len(paths[0]) == 2  # Two facts: John->Alice, Alice->Bob
        store.close()

    def test_find_no_connection(self, tmp_path):
        """No connection exists between entities."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="Alice", predicate="knows", object_value="Bob")

        paths = store.find_connections("John", "Bob", max_depth=5)
        assert len(paths) == 0
        store.close()

    def test_find_all_reachable(self, tmp_path):
        """Find all reachable entities when no target specified."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Alice")
        store.create(subject="John", predicate="works_with", object_value="Bob")

        paths = store.find_connections("John", target_entity=None, max_depth=1)
        assert len(paths) == 2  # Two neighbors
        store.close()

    def test_find_connection_case_insensitive(self, tmp_path):
        """Entity names are case-insensitive."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John Smith", predicate="knows", object_value="Jane Doe")

        paths = store.find_connections("john smith", "jane doe")
        assert len(paths) == 1
        store.close()

    def test_find_connection_with_type_filter(self, tmp_path):
        """Filter connections by entity type."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="Python", subject_type="language",
            predicate="created_by",
            object_value="Guido", object_type="person",
        )
        store.create(
            subject="Python", subject_type="snake",
            predicate="is_a",
            object_value="Reptile", object_type="class",
        )

        # Find connection from language Python
        paths = store.find_connections(
            "Python", "Guido",
            source_type="language", target_type="person",
        )
        assert len(paths) == 1
        assert paths[0][0].subject_type == "language"
        store.close()

    def test_find_connection_respects_limit(self, tmp_path):
        """Limit parameter restricts number of paths."""
        store = FactStore(db_path=tmp_path / "facts.db")
        # Create fan-out from John
        for i in range(10):
            store.create(subject="John", predicate="knows", object_value=f"Person{i}")

        paths = store.find_connections("John", target_entity=None, limit=3)
        assert len(paths) == 3
        store.close()


class TestFactStoreGetNeighbors:
    """Tests for FactStore.get_neighbors method."""

    def test_get_outgoing_neighbors(self, tmp_path):
        """Get neighbors where entity is subject."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="John", predicate="works_at", object_value="Acme")

        neighbors = store.get_neighbors("John")
        assert len(neighbors) == 2
        assert all(n["direction"] == "outgoing" for n in neighbors)
        store.close()

    def test_get_incoming_neighbors(self, tmp_path):
        """Get neighbors where entity is object."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="Alice", predicate="knows", object_value="Bob")
        store.create(subject="Carol", predicate="knows", object_value="Bob")

        neighbors = store.get_neighbors("Bob")
        assert len(neighbors) == 2
        assert all(n["direction"] == "incoming" for n in neighbors)
        store.close()

    def test_get_neighbors_mixed_direction(self, tmp_path):
        """Get both incoming and outgoing neighbors."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")
        store.create(subject="Alice", predicate="knows", object_value="John")

        neighbors = store.get_neighbors("John")
        assert len(neighbors) == 2

        directions = {n["direction"] for n in neighbors}
        assert "outgoing" in directions
        assert "incoming" in directions
        store.close()

    def test_get_neighbors_case_insensitive(self, tmp_path):
        """Neighbor lookup is case-insensitive."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John Smith", predicate="knows", object_value="Jane")

        neighbors = store.get_neighbors("john smith")
        assert len(neighbors) == 1
        store.close()

    def test_get_neighbors_with_type_filter(self, tmp_path):
        """Filter neighbors by entity type."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(
            subject="Python", subject_type="language",
            predicate="created_by",
            object_value="Guido", object_type="person",
        )
        store.create(
            subject="Python", subject_type="snake",
            predicate="is_a",
            object_value="Reptile", object_type="class",
        )

        neighbors = store.get_neighbors("Python", entity_type="language")
        assert len(neighbors) == 1
        assert neighbors[0]["entity"] == "Guido"
        store.close()

    def test_get_neighbors_includes_predicate(self, tmp_path):
        """Neighbors include predicate info."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="works_at", object_value="Acme")

        neighbors = store.get_neighbors("John")
        assert len(neighbors) == 1
        assert neighbors[0]["predicate"] == "works_at"
        store.close()

    def test_get_neighbors_empty(self, tmp_path):
        """No neighbors for isolated entity."""
        store = FactStore(db_path=tmp_path / "facts.db")
        store.create(subject="John", predicate="knows", object_value="Jane")

        neighbors = store.get_neighbors("Alice")  # Not in any fact
        assert len(neighbors) == 0
        store.close()
