# Changelog

## [1.0.7] - 2026-05-31

### Fixed

- **`revalidate_fact_sources` no longer resets every fact source when called with no arguments.** With neither `source_id` nor `source_type`, it issued an unfiltered `UPDATE` that flipped *all* modified/deleted sources back to `active`, silently wiping the integrity-tracking state the tool exists to manage. It now fails fast with `INVALID_INPUT`, requiring at least one filter.
- **`add_fact`, `update_fact`, and `add_facts_batch` now reject out-of-range `confidence`.** The value was passed straight through with no bounds check despite the documented `0.0–1.0` range, so a value like `42.0` was stored and then corrupted `min_confidence` filtering. Out-of-range (or non-numeric, in the batch path) values now return a clear `INVALID_INPUT` error (per-item in the batch). A `None` confidence on `update_fact` still means "unchanged."
- **`rename_tag` and `merge_tags` now match a space-containing source tag.** Tags are stored normalized (lowercased, spaces → hyphens), but these tools normalized the *source* tag with only `lower().strip()`, so `rename_tag(old_tag="my tag", ...)` never matched the stored `my-tag` and silently reported `0` updates. The source tag is now normalized the same way it is stored.

## [1.0.6] - 2026-05-30

### Fixed

- **Orphaned chunks are now actually pruned when a note is re-indexed to fewer chunks.** `_delete_orphan_chunks` called `QdrantStorage.delete_points(...)`, which did not exist in `vector-core` — the resulting `AttributeError` was swallowed by the best-effort `except`, so a note that shrank from N to M chunks left its chunks `M..N-1` behind as stale, still-searchable results. `vector-core` `v1.2.0` adds `delete_points`, and the bump makes the existing cleanup path work. Additionally, the bulk `index_all` incremental path now prunes orphans after each re-index (previously only the single-note `update_note` path did), so a note edited outside the server and picked up by startup/auto-index is cleaned up too. Covered by new regression tests (including a `spec`-checked mock that fails if the method ever disappears again).
- **`index_all` no longer reports success it didn't achieve.** It hard-coded `indexed_notes = total_notes` and `index_healthy = True` even when notes raised during indexing, so `reindex_notes` and startup logs always claimed a perfect index. It now reports the count actually indexed (already-current notes plus this run's successes) and marks the index unhealthy when any note that needed indexing failed.

### Changed

- Bumped the `vector-core` dependency to `v1.2.0` (adds `QdrantStorage.delete_points`).

## [1.0.5] - 2026-05-30

### Fixed

- **`list_notes` now rejects an unsupported `sort_by`** instead of silently returning unsorted results. An invalid value (e.g. `sort_by="date"`) previously matched none of the sort branches and left the results in arbitrary order; it now returns a clear `invalid_input` error listing the valid fields (`modified`, `created`, `title`), matching how `list_documents` validates its enum filters in `mcp-docs`.
- **`get_facts_with_stale_sources` now rejects an unsupported `status`** instead of silently returning an empty list. An invalid value (e.g. `status="active"`) matched neither the `deleted` nor `modified` branch and produced no facts with no explanation; it now returns a clear `invalid_input` error listing the valid values (`deleted`, `modified`, `all`).

## [1.0.4] - 2026-05-30

### Changed

- Bumped the `vector-core` dependency to `v1.1.0`, which adds nested ignore-file support to `FileDiscovery`.

## [1.0.3] - 2026-05-27

### Fixed

- Aligned the runtime package `__version__` constant, project metadata, lockfile package entry, and version regression test.
- Bumped the `vector-core` dependency to `v1.0.5`, where `vector_core.__version__` matches package metadata.

## [1.0.2] - 2026-05-25

### Changed

- Bumped `vector-core` dependency to the reachable `v1.0.4` tag, aligning with corrected Vector Core release metadata.

## [1.0.1] - 2026-05-23

### Changed

- Tagged the first reproducible consumer release after pinning `vector-core` to `v1.0.3`.

## [1.0.0] - 2026-03-20

Initial public release.

### Features

- **38 MCP tools** for note CRUD, search, tagging, linking, glossary, and fact management
- **7 MCP resources** for browsable note and category metadata
- **Markdown notes** with YAML frontmatter, stored as flat files with git version control
- **Hybrid vector search** with dense embeddings + TF-IDF sparse vectors (RRF fusion)
- **Structured fact store** with source tracking, confidence levels, and staleness detection
- **Glossary system** for domain-specific term definitions with vector-indexed search
- **Tag management** with rename, merge, and search operations
- **Note linking** with bidirectional relationship tracking
- **Category organization** with move and rename support
- **Git-backed versioning** with automatic commits and history browsing
- **Incremental indexing** — only re-indexes modified notes
- **POSIX file locking** for safe multi-process access (Linux/macOS)
