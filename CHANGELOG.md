# Changelog

## [1.0.24] - 2026-06-20

### Fixed

- **`get_orphan_notes` (and the orphan count in the note health report) no longer treats a note's link to itself as an incoming link.** An orphan is a note with no incoming links. When building the set of linked-to note ids, the scan unioned every note's frontmatter links and inline `[[uuid]]` body links without excluding the note's own id, so a note that referenced itself (an inline `[[<own-uuid>]]` or its own id in frontmatter links) added itself to that set and was excluded from the orphan list even when no other note linked to it. The scan now discards each note's own id from its outgoing links before recording them, mirroring `_find_backlinks`, which already skips self-references. Genuine cross-note links are unaffected.

## [1.0.23] - 2026-06-20

### Fixed

- **Restoring the commit that deleted a note now returns a clear `INVALID_INPUT` error instead of a generic `INTERNAL_ERROR`.** `get_note_history` lists every commit that touched a note, including the "Delete note: ..." commit, and presents each as a restorable version. Restoring that delete commit could never succeed: the note's content is absent from the delete commit's tree, so the restore aborted and the tool reported `INTERNAL_ERROR` ("Failed to restore version <sha>"), which was misleading because the version id was valid and had been advertised by the history tool. `restore_note_version` now detects this case through a new `GitManager.is_note_deleted_at` helper, which recognizes a deletion commit structurally (the note's blob is absent from the commit but present in its first parent) rather than by matching the commit message, and returns `INVALID_INPUT` explaining that the version is a deletion and an earlier version should be restored instead. A genuinely unknown or malformed version id still returns `INTERNAL_ERROR`. No content is changed; this only improves the error.

## [1.0.22] - 2026-06-19

### Fixed

- **Tags read from a note file are now normalized to the same canonical form the write path uses, so a hand-edited or imported note stays matchable by tag filters.** The write path stores a tag as `lower().strip()` with spaces replaced by hyphens (`normalize_tag`), but `parse_note` only lowercased and stripped, omitting the space-to-hyphen step. A note file whose frontmatter contained a spaced tag (e.g. `My Tag`), from a manual edit, external tooling, or a `git restore` of older content, was parsed as `my tag` instead of the canonical `my-tag`. Tag filters (`list_notes(tags=...)`, `search_notes(tag:...)`/`tags=...`) all normalize the filter to `my-tag`, so the note silently failed to match, and `list_tags` reported a stray `my tag` entry. `parse_note` now normalizes through `normalize_tag` and drops empty tags, matching the write path.

## [1.0.21] - 2026-06-19

### Changed

- Bumped `vector-core` to `v1.2.8`. Pure dependency hygiene: v1.2.8 fixes `SparseVectorizer.extend_vocab` IDF recomputation and the `limit=0` semantics of two library list methods, plus two docstring corrections. mcp-notes does not exercise any of these paths (it never passes `limit=0` to the glossary store, and does not use the standalone `SparseVectorizer`), so no behavior of this server changes. This keeps the pin current with the shared library.

## [1.0.20] - 2026-06-19

### Fixed

- **`search_notes` no longer returns glossary entries and facts mixed in with note results.** Notes, glossary entries, and facts share a single Qdrant collection, distinguished by a `type` payload. `search_notes` runs the engine with the default `mode="both"` and no `type_filter`, but the semantic-query path only added a `type` restriction for `mode="note"` and `mode="chunk"`, never for `"both"`. With no type condition the query matched every type, so a semantic query could surface `[Glossary]` and `[Fact]` items among the notes (the dedicated `search_glossary` and `search_facts` tools exist for those). The default `mode="both"` case (no `type_filter`) now restricts results to `note` and `chunk` types, matching the scoping the no-query (filter-only) path already applies. An explicit `type_filter="all"` still returns all types unrestricted, and glossary and facts remain searchable through their own tools, which pass an explicit `type_filter`.

## [1.0.19] - 2026-06-15

### Fixed

- **`move_category` now records a category change as a git move, so a note is no longer left in version history at both its old and new paths.** Changing a note's category moves its file on disk (the category is the folder path), but `move_category` committed the change with a plain update that staged only the new path and never removed the old one. The committed tree ended up with the note duplicated at both the old and new category paths, and the working tree was left dirty with an unstaged deletion of the old path. Because a later `delete_note` only removes the note's current path, the stale duplicate survived deletion, and a subsequent `restore_note_version` could resurrect the note at the old category with its pre-move content. `move_category` now detects the path change and commits a git move that stages the removal of the old path, matching how single-note updates have always been committed.

## [1.0.18] - 2026-06-15

### Fixed

- **Category filters now match notes stored under a non-slug category instead of silently returning nothing.** Categories are slugified before being written to disk and into the search index (for example `"Work & Projects"` becomes `"work-projects"` and `"Finance"` becomes `"finance"`), but the filters compared the caller's raw input against the stored slug, so `list_notes(category="Finance")`, `search_notes(query="...", category="Finance")`, and the `category:Finance` query syntax all returned an empty result for a category that plainly contains notes. Each filter site now normalizes the category through the same `slugify_category_path()` transform used on write, the same way tag filters were fixed in 1.0.15. `move_category` normalizes its `old_path` and `new_path` too, so renaming a category by its human-readable name now finds and moves the notes rather than reporting zero updates.

## [1.0.17] - 2026-06-14

### Fixed

- **`query_facts(subject_type=..., object_type=...)` now matches fact types case-insensitively**, and **`add_fact`/`update_fact` reject an inverted validity range (`valid_from` after `valid_to`)**. Both come from bumping the shared `vector-core` library to v1.2.7. Fact types are stored exactly as given and `add_fact` passes them through unmodified, so a fact added with `subject_type="Person"` was previously invisible to `query_facts(subject_type="person")` while `find_connections`/`get_entity` (which normalize) still found it — the type filters now normalize consistently. An inverted validity interval, which silently made a fact unmatchable by any time-scoped query, is now rejected at creation/update with a clear `INVALID_INPUT` error instead of being stored.

## [1.0.16] - 2026-06-13

### Fixed

- **Semantic fact search now covers every fact, not just the 50 most recently modified.** Bumped the shared `vector-core` library to v1.2.6, which fixes `FactIndexer.index_all()` (and `_train_vocabulary()`) to index the complete fact corpus — both previously read facts via `FactStore.list_summaries()`, whose `limit` defaults to 50, so on a store with more than 50 facts the older ones were never embedded into Qdrant and were invisible to semantic fact search (`search_notes(type_filter="fact")`), and the sparse vocabulary was trained on only 50 facts. mcp-notes drives fact indexing (`add_fact`/`reindex_notes`), so this is user-reachable: after a reindex, all facts are searchable and the facts IDF is computed over the full corpus. v1.2.6 also makes incremental fact indexing register the vocabulary from the whole corpus (incremental runs had dropped the facts document count to the size of the batch), reads the corpus before any destructive delete, and skips individual unreadable/malformed facts while letting systemic DB errors fail loud.

## [1.0.15] - 2026-06-13

### Fixed

- **Tag filters passed explicitly to `search_notes(tags=[...])` are now normalized to their stored form, so a mixed-case or spaced tag matches instead of silently returning nothing.** Tags are persisted lowercased, stripped, and with spaces collapsed to hyphens, and the `tag:` query-string syntax already normalized to match — but the explicit `tags` parameter was passed straight through to the Qdrant filter, so `search_notes(query="...", tags=["Work"])` filtered on a `"Work"` that no note carries and returned an empty result set. Both paths now share a single `normalize_tag()` helper. `list_notes(tags=[...])` previously only lowercased (missing the strip / space-to-hyphen step) and is normalized the same way. An empty tag can no longer produce a match-nothing Qdrant condition.

## [1.0.14] - 2026-06-12

### Fixed

- **`find_connections` `source_type`/`target_type` filters are now case-insensitive (vector-core `1.2.5`); passing a type exactly as facts display it (e.g. `"Person"`) previously returned no paths.** The store's `entity_adjacency` rows are stored lowercased, but the filters compared the caller's raw input against them, so any mixed-case type silently matched nothing. mcp-notes passes `source_type`/`target_type` through to the store unmodified, so this was user-reachable from the `find_connections` tool. This was the runner-up item in #13, fixed upstream as vector-core#18.

### Changed

- Bumped the `vector-core` dependency to `v1.2.5`. Besides the `find_connections` fix above, `v1.2.5` also makes `VectorStore.get_metadata()` round-trip symmetrically with `set_metadata()`; mcp-notes does not read collection metadata, so that fix does not affect notes.

## [1.0.13] - 2026-06-12

### Fixed

- **`find_connections` no longer garbles the `entities` chain when a path traverses an edge backwards.** The underlying graph search is undirected, but the tool rebuilt the entity chain assuming every fact was walked subject→object, so paths like Alice→Carol over facts ("Bob", "manages", "Alice") and ("Carol", "mentors", "Bob") returned `entities=["Bob", "Alice"]` — missing the target and out of order. The chain is now rebuilt direction-aware, walking from the requested source entity and taking the far side of each fact (case-insensitively, matching the search), so the same query returns `["Alice", "Bob", "Carol"]`.

- **`get_note_history` now returns history for deleted notes instead of silently returning `[]`.** After deletion the note's path is no longer known, and the git layer fell back to a legacy flat path that never matches the current nested layout, so the only history-discovery tool came up empty — even though `delete_note` explicitly promises recoverability from git history. The fallback now searches git history for the path the note last existed at (the same UUID-based tree search version restore relies on), so create→delete→`get_note_history` returns the pre-deletion commits. `restore_note_version` was also wired up for the deleted-note case: it restores the file to its last-known path (instead of a legacy flat path) and refreshes the note store's UUID index afterwards, so the restored note is immediately readable again — deleted notes are recoverable end to end.

- **`rename_tag` onto a tag the note already has no longer writes duplicate tags.** Renaming "foo" to "bar" on a note tagged `[foo, bar]` persisted `tags: [bar, bar]` to frontmatter and inflated `list_tags` counts. `rename_tag` now dedupes like its sibling `merge_tags`: the note ends up with exactly one "bar", with the remaining tag order preserved.

## [1.0.12] - 2026-06-12

### Fixed

- **Blank `subject_type` / `object_type` are rejected with `INVALID_INPUT`.** `add_fact` and `add_facts_batch` preflighted `subject`/`predicate`/`object` and `confidence` at the tool layer, but not the type fields — a whitespace-only `subject_type` or `object_type` was silently stored as garbage. After the vector-core `v1.2.4` bump (which makes `FactStore.create()` raise `ValueError` on blank type fields), the failure mode would have shifted from silent garbage to a raw `ValueError`: an unfriendly generic tool error in `add_fact`, and worse in `add_facts_batch`, where `store.create` is wrapped in `try/except DuplicateFactError` only, so the `ValueError` aborted the entire batch mid-way (earlier items committed, later items never processed). Both tools now validate the effective `subject_type`/`object_type` alongside the existing entity-name checks and return the structured `invalid_input` error dict; in the batch case the bad item lands in the per-item `errors` list and the remaining items are still processed.

### Changed

- Bumped the `vector-core` dependency to `v1.2.4`. `FactStore.create()` and `update()` now validate subject/predicate/object, the type fields, and the `confidence` range (0.0-1.0) before any database access, raising `ValueError` on bad input. This is defense-in-depth beneath mcp-notes' tool-layer preflight: with the fix above, no user-reachable path hands the store a blank field, but the store-level validation guards any future code path that skips the preflight.

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
