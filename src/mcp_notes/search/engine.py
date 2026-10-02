"""Search engine for notes using hybrid search."""

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Literal, cast
from uuid import UUID

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    Prefetch,
    Range,
    ScoredPoint,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)
from vector_core import (
    EmbeddingClient,
    QdrantStorage,
    generate_collection_name,
    parse_iso_datetime,
)
from vector_core.embeddings.client import CircuitBreakerOpenError
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.sparse import SparseVector
from vector_core.search.rank_fusion import reciprocal_rank_fusion
from vector_core.storage.embedding_fragments import fragment_marker
from vector_core.storage.embedding_migration import active_embedding_collection
from vector_core.storage.hybrid import HybridSearcher
from vector_core.storage.hybrid import SearchResult as HybridResult

from mcp_notes.indexing.indexer import NOTES_CODEBASE_ID, NoteIndexer
from mcp_notes.indexing.migration import ensure_notes_collection
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
            self._collection_name = settings.collection_name or generate_collection_name(
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

    async def _ready_collection(self) -> str:
        """Resolve once before embedding, retaining a physical query target."""
        await self._ensure_global_vocab()
        indexer = NoteIndexer(
            note_store=self.note_store,
            storage=self.storage,
            embedder=self.embedder,
            global_vocab=self.global_vocab,
        )
        generation = await ensure_notes_collection(indexer)
        return generation.physical_name

    async def _readable_collection(self) -> tuple[str, CircuitBreakerOpenError | None]:
        try:
            return await self._ready_collection(), None
        except CircuitBreakerOpenError as error:
            # Only an unavailable identity service permits sparse/filter reads
            # of the retained target. Migration/source/schema errors still fail.
            collection = await active_embedding_collection(self.storage, self.collection_name)
            if collection is None:
                raise
            return collection, error

    async def _embed_query(
        self, query: str, readiness_error: CircuitBreakerOpenError | None
    ) -> list[float]:
        if readiness_error is not None:
            raise readiness_error
        return await self.embedder.embed_single_cached(query, role="query")

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
        collection, readiness_error = await self._readable_collection()
        limit = limit or settings.search_limit_default
        self._ensure_vocabulary_registered()

        # Parse query filters
        filters = parse_search_query(query)

        # Merge explicit filters with query filters. Explicit tags and the
        # category are normalized to their stored form (the same as the
        # tag:/category: query syntax), or a caller-supplied "Work"/"my tag"
        # would silently match nothing.
        filters.add_tags(tags)
        if category:
            filters.set_category(category)
        if after:
            filters.after = parse_iso_datetime(after)
        if before:
            filters.before = parse_iso_datetime(before)

        # If no semantic query, fall back to listing
        if not filters.query.strip():
            return await self._filter_only_search(
                filters, mode, limit, type_filter, domain, collection=collection
            )

        # Build Qdrant filters first (shared between hybrid and sparse-only)
        qdrant_filters = filters_to_qdrant(filters)
        rollup_notes = type_filter == "note" or (mode == "note" and type_filter in {None, "all"})
        group_by = (
            "note_id"
            if rollup_notes
            else {"fact": "fact_id", "glossary": "glossary_id"}.get(type_filter or "")
        )
        if type_filter == "all" and mode == "both":
            group_by = "embedding_fragment.parent_id"

        qdrant_filters.extend(self._semantic_type_filters(mode, type_filter))

        # Add domain filter for glossary
        if domain:
            qdrant_filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        # Fetch extra results to account for post-filtering (date ranges, exclude_tags)
        # Use 3x multiplier + buffer to handle cases where many results are filtered
        fetch_limit = max(limit * 3, limit + 20)
        point_list, degraded = await self._query_hits(
            collection, filters.query, qdrant_filters, fetch_limit, group_by, readiness_error
        )
        winners = self._select_winners(point_list, filters, limit)
        display_payloads = await self._entity_display_payloads(collection, winners)

        results = []
        for point in winners:
            payload = point.payload or {}
            content = (
                payload.get("content")
                or payload.get("embedding_text")
                or payload.get("definition", "")
            )
            highlights = self._extract_highlights(content, filters.query) if content else []
            if rollup_notes:
                payload = {**payload, "type": "note"}
            payload = display_payloads.get(point.id, payload)
            result = convert_payload(payload, point.score or 0.0, highlights, degraded)
            if result is None:
                continue
            results.append(result)
        return results

    @staticmethod
    def _select_winners(
        points: Sequence[ScoredPoint | HybridResult], filters: SearchFilters, limit: int
    ) -> list[ScoredPoint | HybridResult]:
        winners = []
        for point in points:
            payload = point.payload or {}
            if payload.get("type", "note") not in {"fact", "glossary"} and not apply_post_filters(
                [{"payload": payload, "score": point.score}], filters
            ):
                continue
            if convert_payload(payload, point.score or 0.0) is not None:
                winners.append(point)
            if len(winners) >= limit:
                break
        return winners

    @staticmethod
    def _semantic_type_filters(mode: str, type_filter: str | None) -> list[FieldCondition]:
        if type_filter is not None and type_filter in {"glossary", "fact", "chunk"}:
            return [FieldCondition(key="type", match=MatchValue(value=type_filter))]
        if type_filter == "note" or mode == "note":
            return [FieldCondition(key="type", match=MatchAny(any=["note", "chunk"]))]
        if mode == "chunk":
            return [FieldCondition(key="type", match=MatchValue(value="chunk"))]
        if type_filter is None:
            return [FieldCondition(key="type", match=MatchAny(any=["note", "chunk"]))]
        return []

    async def _query_hits(
        self,
        collection: str,
        query: str,
        conditions: list[FieldCondition],
        fetch_limit: int,
        group_by: str | None,
        readiness_error: CircuitBreakerOpenError | None,
    ) -> tuple[Sequence[ScoredPoint | HybridResult], bool]:
        """Query complete passages, grouping distinct entities before fusion when requested."""
        query_filter = Filter(must=list(conditions)) if conditions else None
        client = await self.storage.get_client()

        # The per-modality prefetch pool must be at least as large as the
        # post-fusion fetch_limit. Otherwise a large limit is silently capped at
        # the fixed rrf_prefetch_limit candidates per modality, so RRF fusion is
        # starved and returns fewer results than requested even when more notes
        # genuinely match.
        prefetch_limit = max(settings.rrf_prefetch_limit, fetch_limit)

        # Try hybrid search with graceful degradation to sparse-only
        degraded = False
        sparse_vector = self.global_vocab.vectorize_query(query)
        point_list: Sequence[ScoredPoint | HybridResult]

        try:
            # Get dense embeddings for hybrid search
            dense_vector = await self._embed_query(query, readiness_error)

            # Perform hybrid search with RRF fusion
            if group_by:
                if group_by == "embedding_fragment.parent_id":
                    return await self._source_query_hits(
                        collection, conditions, sparse_vector, dense_vector, fetch_limit
                    ), False
                point_list = await HybridSearcher(
                    self.storage, dense_weight=1.0, sparse_weight=1.0
                ).search(
                    collection=collection,
                    dense_query=dense_vector,
                    sparse_query=sparse_vector,
                    limit=fetch_limit,
                    filter_conditions=conditions,
                    group_by=group_by,
                )
            else:
                points = await client.query_points(
                    collection,
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
                point_list = points.points
        except CircuitBreakerOpenError as e:
            # Embedding service unavailable - fall back to sparse-only search
            logger.warning(
                "Embedding service unavailable, falling back to sparse-only search: %s", e
            )
            degraded = True

            if group_by:
                if group_by == "embedding_fragment.parent_id":
                    return await self._source_query_hits(
                        collection, conditions, sparse_vector, None, fetch_limit
                    ), True
                grouped_response = await client.query_points_groups(
                    collection,
                    query=QdrantSparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using="sparse",
                    group_by=group_by,
                    group_size=1,
                    limit=fetch_limit,
                    query_filter=query_filter,
                )
                point_list = [hit for group in grouped_response.groups for hit in group.hits]
            else:
                points = await client.query_points(
                    collection,
                    query=QdrantSparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    limit=fetch_limit,
                    query_filter=query_filter,
                )
                point_list = points.points

        return point_list, degraded

    async def _source_query_hits(
        self,
        collection: str,
        conditions: list[FieldCondition],
        sparse_vector: SparseVector,
        dense_vector: list[float] | None,
        limit: int,
    ) -> list[HybridResult]:
        """Group modern source fragments while keeping markerless legacy records searchable."""
        client = await self.storage.get_client()
        missing_parent = IsEmptyCondition(is_empty=PayloadField(key="embedding_fragment.parent_id"))

        async def modality(
            query: list[float] | QdrantSparseVector, using: str
        ) -> list[ScoredPoint]:
            grouped, legacy = await asyncio.gather(
                client.query_points_groups(
                    collection,
                    query=query,
                    using=using,
                    group_by="embedding_fragment.parent_id",
                    group_size=1,
                    limit=limit,
                    query_filter=Filter(must=list(conditions)),
                ),
                client.query_points(
                    collection,
                    query=query,
                    using=using,
                    limit=limit,
                    query_filter=Filter(must=[*conditions, missing_parent]),
                ),
            )
            hits = [hit for group in grouped.groups for hit in group.hits]
            return sorted([*hits, *legacy.points], key=lambda hit: hit.score, reverse=True)[:limit]

        queries = []
        if dense_vector is not None:
            queries.append(modality(dense_vector, "dense"))
        queries.append(
            modality(
                QdrantSparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                "sparse",
            )
        )
        ranked = await asyncio.gather(*queries)

        def source_id(point: ScoredPoint) -> int | str | UUID:
            marker = fragment_marker(point.payload or {})
            return marker["parent_id"] if marker else point.id

        if dense_vector is None:
            return [
                HybridResult(id=cast(int | str, hit.id), score=hit.score, payload=hit.payload or {})
                for hit in ranked[0]
            ]
        fused = reciprocal_rank_fusion(ranked, key=source_id, limit=limit)
        return [
            HybridResult(
                id=cast(int | str, result.item.id),
                score=result.score,
                payload=result.item.payload or {},
            )
            for result in fused
        ]

    async def _entity_display_payloads(
        self, collection: str, winners: Sequence[ScoredPoint | HybridResult]
    ) -> dict[int | str | UUID, dict]:
        """Hydrate canonical entity fields once while retaining each winning dense snippet."""
        parents = {}
        for point in winners:
            if (point.payload or {}).get("type") not in {"fact", "glossary"}:
                continue
            marker = fragment_marker(point.payload or {})
            if marker is not None and marker["index"] > 0:
                parents[point.id] = marker["parent_id"]
        if not parents:
            return {}
        client = await self.storage.get_client()
        records = await client.retrieve(
            collection,
            ids=list(set(parents.values())),
            with_vectors=False,
            with_payload=[
                "type",
                "fact_id",
                "subject",
                "predicate",
                "object",
                "subject_type",
                "object_type",
                "context",
                "created",
                "modified",
                "glossary_id",
                "term",
                "expansion",
                "definition",
                "domain",
                "embedding_fragment",
            ],
        )
        canonical = {record.id: record.payload or {} for record in records}
        hydrated = {}
        for point in winners:
            if point.id not in parents:
                continue
            parent = canonical.get(parents[point.id])
            if parent is None:
                raise ValueError("Winning embedding fragment has no canonical entity")
            child = point.payload or {}
            marker = fragment_marker(child)
            parent_marker = fragment_marker(parent)
            entity_key = f"{child['type']}_id"
            if (
                marker is None
                or parent_marker is None
                or parent_marker["index"] != 0
                or parent_marker["parent_id"] != parents[point.id]
                or parent_marker["source_hash"] != marker["source_hash"]
                or parent.get("type") != child["type"]
                or not child.get(entity_key)
                or parent.get(entity_key) != child[entity_key]
            ):
                raise ValueError("Winning embedding fragment has no matching canonical source")
            display = {key: value for key, value in parent.items() if key != "embedding_fragment"}
            hydrated[point.id] = {**child, **display}
        return hydrated

    async def _filter_only_search(
        self,
        filters: SearchFilters,
        mode: Literal["note", "chunk", "both"],
        limit: int,
        type_filter: Literal["note", "chunk", "glossary", "fact", "all"] | None = None,
        domain: str | None = None,
        *,
        collection: str | None = None,
    ) -> list[SearchResult]:
        """Search using filters only (no semantic query)."""
        collection = collection or (await self._readable_collection())[0]
        qdrant_filters = filters_to_qdrant(filters)

        # Add type filter
        if type_filter == "glossary":
            qdrant_filters.append(FieldCondition(key="type", match=MatchValue(value="glossary")))
        elif type_filter == "fact":
            qdrant_filters.append(FieldCondition(key="type", match=MatchValue(value="fact")))
        elif type_filter == "note":
            qdrant_filters.append(FieldCondition(key="type", match=MatchValue(value="note")))
        elif type_filter == "chunk":
            qdrant_filters.append(FieldCondition(key="type", match=MatchValue(value="chunk")))
        elif type_filter is None or type_filter == "all":
            # Default to note-level for filter-only
            if mode in ("note", "both"):
                qdrant_filters.append(FieldCondition(key="type", match=MatchValue(value="note")))

        # Add domain filter for glossary
        if domain:
            qdrant_filters.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        # Fetch extra results to account for post-filtering
        fetch_limit = max(limit * 3, limit + 20)

        points = await self.storage.scroll_points(
            collection,
            filter_conditions=[
                *qdrant_filters,
                Filter(
                    must_not=[
                        FieldCondition(key="embedding_fragment.index", range=Range(gt=0)),
                    ]
                ),
            ],
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
        collection, _ = await self._readable_collection()
        client = await self.storage.get_client()
        source_filter = Filter(
            must=[
                FieldCondition(key="note_id", match=MatchValue(value=str(note_id))),
                FieldCondition(key="type", match=MatchValue(value="chunk")),
            ]
        )
        target_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="chunk"))],
            must_not=[FieldCondition(key="note_id", match=MatchValue(value=str(note_id)))],
        )
        best: dict[int | str, tuple[int | str | UUID, float]] = {}
        offset = None
        while True:
            sources, offset = await client.scroll(
                collection,
                scroll_filter=source_filter,
                offset=offset,
                limit=128,
                with_vectors=["dense"],
                with_payload=False,
            )
            for source in sources:
                vector = source.vector
                if not isinstance(vector, dict) or not vector.get("dense"):
                    raise ValueError("Indexed note chunk has no dense vector")
                response = await client.query_points_groups(
                    collection,
                    query=vector["dense"],
                    using="dense",
                    group_by="note_id",
                    group_size=1,
                    limit=limit,
                    query_filter=target_filter,
                    with_payload=False,
                    with_vectors=False,
                )
                for group in response.groups:
                    for hit in group.hits:
                        if group.id not in best or hit.score > best[group.id][1]:
                            best[group.id] = (hit.id, hit.score)
            if offset is None:
                break
        winners = sorted(best.values(), key=lambda hit: hit[1], reverse=True)[:limit]
        if not winners:
            return []
        records = await client.retrieve(
            collection,
            ids=[point_id for point_id, _ in winners],
            with_payload=True,
            with_vectors=False,
        )
        payloads = {record.id: record.payload or {} for record in records}
        results = []
        for point_id, score in winners:
            result = convert_payload({**payloads.get(point_id, {}), "type": "note"}, score)
            if result is not None:
                results.append(result)
        return results

    async def close(self) -> None:
        """Close connections safely."""
        if self.storage is not None:
            await self.storage.close()
        if self.embedder is not None:
            await self.embedder.close()
