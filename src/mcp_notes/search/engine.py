"""Search engine for notes using hybrid search."""

import logging
import re
from typing import Literal
from uuid import UUID

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)
from vector_core import (
    EmbeddingClient,
    QdrantStorage,
    generate_collection_name,
    generate_point_id,
    parse_iso_datetime,
)
from vector_core.embeddings.client import CircuitBreakerOpenError
from vector_core.embeddings.global_vocab import GlobalVocabulary

from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID
from mcp_notes.models import SearchResult
from mcp_notes.search.converters import convert_payload
from mcp_notes.search.filters import (
    SearchFilters,
    apply_post_filters,
    filters_to_qdrant,
    parse_search_query,
)
from mcp_notes.settings import settings
from mcp_notes.storage.filesystem import NoteStore

logger = logging.getLogger(__name__)


class NoteSearchEngine:
    """
    Search engine for notes using hybrid semantic + keyword search.

    Uses Qdrant's built-in RRF fusion for hybrid search.
    """

    def __init__(
        self,
        note_store: NoteStore | None = None,
        storage: QdrantStorage | None = None,
        embedder: EmbeddingClient | None = None,
        global_vocab: GlobalVocabulary | None = None,
    ):
        """
        Initialize search engine.

        Args:
            note_store: NoteStore instance
            storage: QdrantStorage instance
            embedder: EmbeddingClient instance
            global_vocab: GlobalVocabulary instance (uses singleton if not provided)
        """
        self.note_store = note_store or NoteStore()
        self.storage = storage or QdrantStorage()
        self.embedder = embedder or EmbeddingClient()
        self._global_vocab = global_vocab  # Use singleton if None
        self._collection_name: str | None = None

    @property
    def global_vocab(self) -> GlobalVocabulary:
        """Get GlobalVocabulary instance.

        Returns the instance passed to __init__, or the singleton after
        _ensure_global_vocab() is called.

        Raises:
            RuntimeError: If accessed before async initialization.
        """
        if self._global_vocab is None:
            raise RuntimeError(
                "GlobalVocabulary not initialized. Call await _ensure_global_vocab() first, "
                "or pass an instance to __init__."
            )
        return self._global_vocab

    async def _ensure_global_vocab(self) -> None:
        """Ensure GlobalVocabulary is initialized using singleton."""
        if self._global_vocab is None:
            self._global_vocab = GlobalVocabulary.get_instance()

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        if self._collection_name is None:
            self._collection_name = generate_collection_name(
                str(self.note_store.base_dir),
                prefix=settings.collection_prefix,
            )
        return self._collection_name

    def _ensure_vocabulary_registered(self) -> bool:
        """Check if GlobalVocabulary has notes codebase registered."""
        if self.global_vocab.get_codebase_doc_count(NOTES_CODEBASE_ID) > 0:
            return True
        logger.warning("GlobalVocabulary not registered for notes, sparse search may be limited")
        return False

    async def search(
        self,
        query: str,
        mode: Literal["note", "chunk", "both"] = "both",
        limit: int | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
        after: str | None = None,
        before: str | None = None,
        type_filter: Literal["note", "chunk", "glossary", "fact", "all"] | None = None,
        domain: str | None = None,
    ) -> list[SearchResult]:
        """
        Search notes and glossary with hybrid semantic + keyword search.

        Args:
            query: Search query (supports filter syntax)
            mode: Search mode - "note" for file-level, "chunk" for sections, "both"
            limit: Max results (default from settings)
            tags: Additional tag filters
            category: Additional category filter
            after: Created after date (ISO format)
            before: Created before date (ISO format)
            type_filter: Filter by content type - "note", "chunk", "glossary", or "all"
            domain: Filter glossary entries by domain

        Returns:
            List of SearchResult objects
        """
        await self._ensure_global_vocab()
        limit = limit or settings.search_limit_default
        self._ensure_vocabulary_registered()

        # Parse query filters
        filters = parse_search_query(query)

        # Merge explicit filters with query filters. Explicit tags are
        # normalized to their stored form (lowercase, hyphenated) — the same
        # as tag: query syntax — or a caller-supplied "Work"/"my tag" would
        # silently match nothing.
        filters.add_tags(tags)
        if category:
            filters.category = category
        if after:
            filters.after = parse_iso_datetime(after)
        if before:
            filters.before = parse_iso_datetime(before)

        # If no semantic query, fall back to listing
        if not filters.query.strip():
            return await self._filter_only_search(filters, mode, limit, type_filter, domain)

        # Build Qdrant filters first (shared between hybrid and sparse-only)
        qdrant_filters = filters_to_qdrant(filters)

        # Add type filter - type_filter takes precedence over mode
        if type_filter == "glossary":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="glossary"))
            )
        elif type_filter == "fact":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="fact"))
            )
        elif type_filter == "note":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="note"))
            )
        elif type_filter == "chunk":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="chunk"))
            )
        elif type_filter is None or type_filter == "all":
            # Use mode for notes/chunks filtering
            if mode == "note":
                qdrant_filters.append(
                    FieldCondition(key="type", match=MatchValue(value="note"))
                )
            elif mode == "chunk":
                qdrant_filters.append(
                    FieldCondition(key="type", match=MatchValue(value="chunk"))
                )

        # Add domain filter for glossary
        if domain:
            qdrant_filters.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )

        query_filter = Filter(must=qdrant_filters) if qdrant_filters else None

        # Get Qdrant client and set up search parameters
        client = await self.storage.get_client()
        prefetch_limit = settings.rrf_prefetch_limit

        # Fetch extra results to account for post-filtering (date ranges, exclude_tags)
        # Use 3x multiplier + buffer to handle cases where many results are filtered
        fetch_limit = max(limit * 3, limit + 20)

        # Try hybrid search with graceful degradation to sparse-only
        degraded = False
        sparse_vector = self.global_vocab.vectorize_query(filters.query)

        try:
            # Get dense embeddings for hybrid search
            dense_vector = await self.embedder.embed_single_cached(filters.query)

            # Perform hybrid search with RRF fusion
            points = await client.query_points(
                self.collection_name,
                prefetch=[
                    Prefetch(
                        query=QdrantSparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=fetch_limit,
            )
        except CircuitBreakerOpenError as e:
            # Embedding service unavailable - fall back to sparse-only search
            logger.warning(
                "Embedding service unavailable, falling back to sparse-only search: %s", e
            )
            degraded = True

            points = await client.query_points(
                self.collection_name,
                query=QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using="sparse",
                limit=fetch_limit,
                query_filter=query_filter,
            )

        # Convert to results with post-filtering
        results = []
        for point in points.points:
            payload = point.payload or {}
            point_type = payload.get("type", "note")

            # Apply post-filters (glossary/fact use domain filter, not tag/date filters)
            if point_type not in ("glossary", "fact"):
                result_dict = {"payload": payload, "score": point.score}
                if not apply_post_filters([result_dict], filters):
                    continue

            # Extract highlights from searchable content
            highlights = []
            content = payload.get("content") or payload.get("definition", "")
            if content and filters.query:
                highlights = self._extract_highlights(content, filters.query)

            # Convert payload to SearchResult using unified converter
            result = convert_payload(payload, point.score or 0.0, highlights, degraded)
            if result is None:
                continue

            results.append(result)

            if len(results) >= limit:
                break

        return results

    async def _filter_only_search(
        self,
        filters: SearchFilters,
        mode: Literal["note", "chunk", "both"],
        limit: int,
        type_filter: Literal["note", "chunk", "glossary", "fact", "all"] | None = None,
        domain: str | None = None,
    ) -> list[SearchResult]:
        """Search using filters only (no semantic query)."""
        qdrant_filters = filters_to_qdrant(filters)

        # Add type filter
        if type_filter == "glossary":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="glossary"))
            )
        elif type_filter == "fact":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="fact"))
            )
        elif type_filter == "note":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="note"))
            )
        elif type_filter == "chunk":
            qdrant_filters.append(
                FieldCondition(key="type", match=MatchValue(value="chunk"))
            )
        elif type_filter is None or type_filter == "all":
            # Default to note-level for filter-only
            if mode in ("note", "both"):
                qdrant_filters.append(
                    FieldCondition(key="type", match=MatchValue(value="note"))
                )

        # Add domain filter for glossary
        if domain:
            qdrant_filters.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )

        # Fetch extra results to account for post-filtering
        fetch_limit = max(limit * 3, limit + 20)

        points = await self.storage.scroll_points(
            self.collection_name,
            filter_conditions=qdrant_filters if qdrant_filters else None,
            limit=fetch_limit,
        )

        # Apply post-filters and convert
        results = []
        for payload in points:
            point_type = payload.get("type", "note")

            # Apply post-filters only for notes (glossary/fact use domain filter)
            if point_type not in ("glossary", "fact"):
                result_dict = {"payload": payload}
                if not apply_post_filters([result_dict], filters):
                    continue

            # Convert using unified converter (score=1.0 for filter-only matches)
            result = convert_payload(payload, score=1.0)
            if result is None:
                continue

            results.append(result)

            if len(results) >= limit:
                break

        return results

    def _extract_highlights(self, content: str, query: str, max_highlights: int = 3) -> list[str]:
        """
        Extract highlight snippets around query terms.

        Args:
            content: Full content
            query: Query string
            max_highlights: Max snippets to return

        Returns:
            List of highlight snippets
        """
        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()

        for term in query_terms:
            if len(term) < 3:
                continue

            # Find occurrences
            for match in re.finditer(re.escape(term), content_lower):
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)

                # Expand to word boundaries
                while start > 0 and content[start - 1] not in " \n":
                    start -= 1
                while end < len(content) and content[end] not in " \n":
                    end += 1

                snippet = content[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."

                if snippet not in highlights:
                    highlights.append(snippet)

                if len(highlights) >= max_highlights:
                    break

            if len(highlights) >= max_highlights:
                break

        return highlights

    async def find_similar(
        self,
        note_id: UUID,
        limit: int = 5,
    ) -> list[SearchResult]:
        """
        Find notes similar to a given note.

        Args:
            note_id: Source note UUID
            limit: Max results

        Returns:
            List of similar notes (excluding the source)
        """
        # Get the source note's embedding
        client = await self.storage.get_client()

        # Find the note's vector using deterministic point ID
        point_key = f"note:{note_id}"
        point_id = generate_point_id(point_key)

        try:
            points = await client.retrieve(
                self.collection_name,
                ids=[point_id],
                with_vectors=True,
            )
        except Exception as e:
            logger.debug(f"Failed to retrieve vector for similar notes lookup: {e}")
            return []

        if not points or not points[0].vector:
            return []

        vector = points[0].vector
        if not isinstance(vector, dict):
            return []
        dense_vector = vector.get("dense", [])
        if not dense_vector:
            return []

        # Search for similar notes
        response = await client.query_points(
            self.collection_name,
            query=dense_vector,
            using="dense",
            limit=limit + 1,  # +1 to exclude self
            query_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="note")),
                ],
            ),
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            result_note_id = payload.get("note_id", "")

            # Skip self
            if result_note_id == str(note_id):
                continue

            # Convert using unified converter
            result = convert_payload(payload, point.score or 0.0)
            if result is None:
                continue

            results.append(result)

            if len(results) >= limit:
                break

        return results

    async def close(self) -> None:
        """Close connections safely."""
        if self.storage is not None:
            await self.storage.close()
        if self.embedder is not None:
            await self.embedder.close()
