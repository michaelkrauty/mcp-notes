# Changelog

## [1.0.11] - 2026-06-12

### Fixed

- **Case-only renames and alias-promoting renames of glossary terms no longer fail.** Bumped the `vector-core` dependency to `v1.2.3`, which fixes a bug in `GlossaryStore.update()`: the term-uniqueness check did not exclude the entry being updated, so renaming a term to a different casing of itself (e.g. `"USAF"` → `"Usaf"`) or to one of the entry's own aliases incorrectly raised `TermExistsError`. This bug was reachable through `update_glossary_entry` — mcp-notes' tool-layer preflight validation (added in `1.0.9`) self-excludes the current entry for *alias* collision checks, but the *term* path is validated by the store itself, so these renames were impossible through mcp-notes until now.

## [1.0.10] - 2026-06-12

### Changed

- Bumped the `vector-core` dependency to `v1.2.2`.
- **Defense-in-depth for glossary alias validation.** vector-core `v1.2.2` makes `GlossaryStore.create()` and `update()` validate aliases (cross-entry collisions and case-normalized intra-list duplicates) *before* any row is written, raising `TermExistsError` with the store left fully unchanged, plus a rollback-on-error backstop. mcp-notes' glossary tools call `GlossaryStore` directly and already preflight these checks in their own tool layer (since `1.0.9`), so no user-reachable bug is fixed here — the store-level validation guards against races between the tool-layer preflight and the store mutation, and against any future code path that hits the store without that preflight.

## [1.0.9] - 2026-06-11

### Fixed

- **Blank or duplicate glossary input is rejected with `INVALID_INPUT`.** `add_glossary_entry` accepted whitespace-only `term`/`expansion`/`definition` and stored junk entries; `update_glossary_entry` likewise accepted blank values. Alias lists that collide after normalization (e.g. `["api", " api "]` or `["API", "api"]`) crashed on the database's UNIQUE constraint — on update, after the old aliases were already deleted. All of these now fail fast with a clear error before anything is written, and accepted values are stripped of surrounding whitespace. A blank `domain` is treated as "no domain" and the documented `""`-clears-domain behavior on update now stores a real NULL instead of an empty string.
- **Corrected the 1.0.8 release notes**, which claimed this validation arrived with vector-core v1.2.1. That release added it to vector-core's shared `GlossaryToolHelper`, but mcp-notes' glossary tools call the `GlossaryStore` directly and never gained it; this release implements the validation in mcp-notes' own tool layer.

## [1.0.8] - 2026-06-11

### Changed

- Bumped the `vector-core` dependency to `v1.2.1`.

### Fixed

Inherited from `vector-core` `v1.2.1`:

- **Fact query and list results are ordered most-recently-modified-first again.** `FactStore` batch reads dropped the `ORDER BY modified DESC` ordering, so `query_facts`, `list_facts`, and related tools returned facts in arbitrary order.
- **Glossary alias-only updates no longer leave a stale change-detection hash.** `GlossaryStore.update()` kept the old `entry_hash` when only aliases changed, so subsequent change detection could miss or misreport the update.

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
