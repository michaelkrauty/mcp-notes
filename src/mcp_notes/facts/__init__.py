"""Facts subsystem for structured knowledge storage.

All components are now provided by vector-core for sharing across MCP servers.
This module re-exports everything for backward compatibility.
"""

# Re-export everything from vector_core.facts
from vector_core.facts import (
    FACTS_CODEBASE_ID,
    DuplicateFactError,
    Fact,
    FactError,
    FactIndexer,
    FactNotFoundError,
    FactSource,
    FactStore,
    FactSummary,
    IntegrityCheckResult,
    SourceIntegrityManager,
    SourceStatus,
    SourceType,
    compute_spo_hash,
    generate_fact_text,
)

__all__ = [
    # All from vector_core.facts
    "FactStore",
    "FactIndexer",
    "FACTS_CODEBASE_ID",
    "generate_fact_text",
    "Fact",
    "FactSource",
    "FactSummary",
    "FactError",
    "FactNotFoundError",
    "DuplicateFactError",
    "SourceType",
    "SourceStatus",
    "compute_spo_hash",
    "SourceIntegrityManager",
    "IntegrityCheckResult",
]
