# Changelog

## [1.1.0] - 2026-08-21

### Added

- **The server now supports the stateless MCP `2026-07-28` protocol.** Modern requests carry their protocol version and client capabilities independently, receive server identity in each result, and use `server/discover` instead of a connection-scoped initialization handshake. The MCP Python SDK v2 supplies the wire validation, required `resultType` and cache metadata, cancellation behavior, and stdio framing.
- Protocol tests exercise modern and legacy negotiation, deterministic `tools/list` ordering, resource discovery, and side-effect-free tool dispatch against the real registered surface without reading note data. They also verify the conservative `ttlMs: 0`, `cacheScope: "private"` defaults used for static listings.

### Fixed

- Startup auto-indexing and async resource cleanup now run inside `MCPServer`'s lifespan on the serving event loop, preventing cached clients from being reused after their original loop closes.
- Successful note, tag, category, and version mutations now publish resource-update events for the server's dynamic `notes://` resources to modern subscription streams.

### Changed

- Replaced the removed `FastMCP` API with `MCPServer` and now publish the package version through its public constructor instead of mutating the private low-level server.
- Upgraded to `mcp~=2.0.0` and pinned vector-core v1.4.2 for its SDK v2-compatible registration checks. SDK v2 serves `2025-11-25` and earlier handshake-based clients from the same stdio process, so existing clients remain supported while modern clients use the stateless path.

## [1.0.46] - 2026-07-25

### Fixed

- **The test suite no longer depends on the developer's ambient environment to pass.** Settings read the embedding dimension from the environment once, at import, and left it at zero when unset, so every test that stored a vector failed with `embedding_dim not yet initialized` — 34 failing or erroring tests on a clean checkout. Tests that build their own vectors only need the dimension to be some consistent number, so the suite defaults one. An explicitly exported dimension still wins, because a developer running against a real embedding service needs the dimension that service actually returns.
- **Tests that require a live embedding service are skipped when one is not reachable, rather than failing.** `TestFactIndexerIntegration` documented itself as "skipped if unavailable" while carrying no skip condition at all, and the search and fact-indexing integration modules had none either, so all of them failed at the first embedding call on a checkout without the service. A shared `requires_full_stack` marker now guards them. A clean checkout with no services reports 765 passed and 32 skipped.

### Changed

- **Development dependencies moved from an extra to a PEP 735 dependency group.** `uv sync` installs a dependency group by default but skips an extra unless it is named, so the project environment was left without pytest and `uv run pytest` silently fell through to whatever pytest was on `PATH`. That interpreter brought its own installed `vector-core`, meaning the suite could report on a different copy of the library than the one this project pins. Deployments that want a lean environment can pass `uv sync --no-default-groups`.


## [1.0.45] - 2026-07-25

### Changed

- **Pinned vector-core to v1.4.0.** Every Qdrant request now carries the configured `qdrant_operation_timeout` instead of qdrant-client's five second default, which nothing had been overriding. Any request whose duration scales with the size of the working set was subject to that cap regardless of configuration, and exceeding it raised `ResponseHandlingException(ReadTimeout(''))` — an exception with an empty message, naming neither a timeout nor a duration. Re-indexing the note corpus, where a bulk upsert can exceed five seconds, is where this server was most exposed.

## [1.0.44] - 2026-07-24

### Changed

- **Pinned vector-core to v1.3.1.** `update_codebase_incremental` can now establish a codebase document count instead of silently discarding it, so a corpus maintained purely by deltas no longer reports zero documents while contributing document frequencies. It also caps a removal at what the codebase actually contributed, keeping the shared aggregate equal to the sum of the per-codebase contributions, and floors counts and frequencies at zero: a negative value made the query weighting `log((total + 1) / (df + 1)) + 1` divide by zero or take the logarithm of a negative number, raising out of `vectorize_query` for every codebase sharing the database.

## [1.0.43] - 2026-07-24

### Fixed

- **The per-note lock now actually excludes concurrent writers.** `_file_lock` unlinked its lock file on release, but `flock` is associated with the inode rather than the pathname. A waiter that had already opened the file kept its lock on the now unreachable inode, while the next arrival created a fresh inode and took an uncontended lock on that, so two callers ran the same note's read-modify-write at once. Under an eight-way contention test this was not a rare interleaving: mutual exclusion was violated on the majority of acquisitions, with up to three holders inside the critical section at once. `NoteStore.update` and `NoteStore.delete` rewrite the note file and the UUID index under this lock, so the exposure was lost updates and an index disagreeing with what is on disk. The lock file is now created once and left in place; it is empty, and one accumulates per note UUID ever locked.
- **Startup no longer deletes lock files it judges stale.** Lock files older than an hour were removed on `ensure_directories()`, which is the same defect from the other direction: deleting a file another process is holding leaves that process with an orphaned inode while the next arrival locks a new one. Age carries no information here, because the kernel releases a `flock` as soon as the holder's descriptor closes, including when the process dies, so an old lock file is simply an unheld one.
- **A notes directory that was already a git repository now excludes the lock directory too.** The `.gitignore` covering `.locks/` is only written when this package initialises the repository itself, so an adopted repository never got it. Now that lock files persist, that left them as permanent untracked entries which `git add -A` could commit, and a later checkout or sync replacing a held lock's inode would recreate the very failure persistent lock files prevent. The rule is appended to git's `info/exclude`, which is not version-controlled and leaves any `.gitignore` the user maintains untouched. It goes in the repository's common directory, since that is where git reads `info/exclude` from even for a linked worktree, so the rule is shared with every worktree of the repository. Bare repositories are left alone, having no working tree to exclude anything from.

### Upgrade note

Restart every running server process. Keeping the existing lock file name means an upgraded process and one still running the old code contend on the same path, rather than on two different ones, but a process running the old code still unlinks that file out from under the new one. Only leaving the mixed-version window closes the gap.

### Known limitation

The lock covers the note file and the UUID index, not the Git commit, which `NoteService` performs after `NoteStore` has released it. Two concurrent service-level operations on one note can therefore still interleave around their commits.

## [1.0.42] - 2026-07-10

### Changed

- **Pinned vector-core to v1.3.0.** Brings in the restored MCP tool registration verification, which had been silently skipped against current FastMCP releases and now genuinely checks that every expected tool registered at startup, and the `find_connections()` fix for duplicate paths on a self-referential fact.

## [1.0.41] - 2026-07-10

### Fixed

- **New facts are now immediately available to semantic fact search.** `add_fact` committed facts to SQLite without adding them to the semantic index, and `add_facts_batch` had the same omission, so newly created facts stayed invisible to `search_facts` until a manual `index_facts` run. Single creates now call `index_fact`, while batch creates use one incremental `index_all(force=False)` pass instead of invoking the indexer once per fact. Index maintenance remains best-effort after the database commit: an indexing failure is logged but does not lose the fact, change the successful create response, or abort the batch. The batch tool documentation now also reflects that each newly created fact is committed independently rather than claiming one enclosing transaction.

## [1.0.40] - 2026-07-04

### Fixed

- **`add_facts_batch` no longer crashes the whole batch on a non-string date.** Batch facts arrive as `list[dict]` whose values are not type-coerced by the tool schema, so a `valid_from` or `valid_to` that is not a string (for example a JSON integer `2020`) made `date.fromisoformat` raise `TypeError`, which was not caught (only `ValueError` was), aborting the entire batch after earlier valid facts had already been committed. This is the same partial-commit-abort class that was fixed for a non-string subject, predicate, or object in v1.0.33; the date fields were missed. Date parsing now runs through a shared `_parse_optional_iso_date` helper that reports a non-string date as a per-item error and continues with the rest of the batch. `add_fact` and `update_fact` declare typed `str | None` parameters and never hit the non-string path; `add_fact` now shares the same helper so date parsing lives in one place.

## [1.0.39] - 2026-06-25

### Fixed

- **`get_note_links` no longer crashes when a note on disk has unparseable frontmatter.** The outgoing-link loop caught only `NoteNotFoundError` from `get_summary()`, and the source note's read plus parse was guarded the same narrow way. But a note whose file exists (so it is in the UUID index and `exists()` returns True) can still fail to parse: a missing `---` fence, invalid YAML, non-mapping frontmatter, an oversized frontmatter block, or a missing required field all raise a bare `ValueError` (or `FrontmatterTooLargeError`, a `ValueError` subclass), not `NoteNotFoundError`. That exception propagated out of the tool and aborted the entire link view for the note. Both paths now catch the same set of storage and parse errors that `NoteStore.list_all`/`iter_all` already tolerate per file: a corrupt outgoing target is reported in `broken` and skipped, and a corrupt source note returns empty links. Known limitation: a corrupt target is still not flagged by `get_all_broken_links`, which checks only index-level existence.

## [1.0.38] - 2026-06-20

### Fixed

- **`get_note_links` no longer crashes on a dangling outgoing link when the in-memory UUID index is stale relative to disk.** `NoteStore.get_summary()` resolved the path from the UUID index and read it without checking the file still exists, unlike `read()`/`delete()` which rebuild the index and raise `NoteNotFoundError` when a path is in the index but the file is gone. So if a linked note's file was removed out-of-band (an external delete, or a `git pull`/`checkout`/`stash` in this git-backed markdown store) while its UUID lingered in the not-yet-reindexed index, resolving the link target raised a bare `FileNotFoundError`. The link resolver only catches `NoteNotFoundError`, so the exception propagated and the entire link view for the note failed instead of reporting the target in `broken`. `get_summary()` now applies the same stale-index guard as `read()`, so a dangling target is correctly reported as broken.

## [1.0.37] - 2026-06-20

### Fixed

- **Corrected the `update_fact` documentation for clearing `context`/`valid_from`/`valid_to`.** The docstrings said "pass null to clear", but the tool cannot distinguish an omitted argument from an explicit JSON null (both arrive as `None`), and `None` is treated as "leave unchanged". The actual clear mechanism is an empty string `""`: it clears `valid_from`/`valid_to` to null and sets `context` to blank. Following the old documentation (passing null) silently left the field unchanged. The docstrings now describe the real behavior.

## [1.0.36] - 2026-06-20

### Fixed

- **`create_note`/`update_note` no longer write duplicate tags when distinct raw tags canonicalize to the same stored form.** The two write paths normalized tags with an inline comprehension that did not deduplicate and did not route through `normalize_tag`, so passing tags like `["My Tag", "my-tag"]` stored (and returned) `["my-tag", "my-tag"]`. The read/parse path deduplicates, so the create/update response disagreed with every later read of the same note, and the note's frontmatter held redundant entries. Both write paths now use `normalize_tag` with order-preserving deduplication, matching the read path.

## [1.0.35] - 2026-06-20

### Fixed

- **A note whose frontmatter has an unquoted date (e.g. `created: 2024-01-15`) is no longer silently excluded from every result.** PyYAML loads an unquoted date-only value as a `datetime.date` (not a `datetime` or a string), which the datetime parser did not handle, so parsing the note raised `ValueError`. Through `read_note` this surfaced as an error; through the bulk paths (`list_notes`, `search_notes`, link/backlink resolution, indexing, `list_tags`, `list_categories`) the note was swallowed as a parse error and silently dropped, making a valid note invisible and unsearchable. Unquoted date-only values (the default shape produced by Obsidian, Jekyll/Hugo, Logseq, and hand-edited notes) are now accepted and promoted to midnight UTC.

## [1.0.34] - 2026-06-20

### Fixed

- **Hybrid search no longer silently drops matching results when `limit` is large.** The post-fusion fetch size scales with the requested limit (3x, to survive post-filtering), but the per-modality prefetch pool was a fixed `rrf_prefetch_limit` (50). RRF fusion only ranks the union of the two prefetch candidate lists, so once the limit approached or exceeded 50 the fusion stage was starved of candidates and returned fewer results than requested even though more notes genuinely matched (e.g. `search_notes(query=..., limit=80)` could return ~73). The prefetch pool now scales to at least the post-fusion fetch size, so fusion can return up to the requested limit. `search_notes`, `search_glossary`, and `search_facts` (which share the hybrid path) now also enforce their documented maximum limit of 100, bounding the prefetch fan-out.

## [1.0.33] - 2026-06-20

### Fixed

- **`restore_note_version` no longer raises an unhandled error for a syntactically valid but unknown commit SHA.** Resolving a 40-character hex SHA that names no object in the repository raises a bare `ValueError` from GitPython, which `get_version_content` did not catch (it caught only malformed-ref errors), so the tool surfaced an internal exception instead of the documented clean error response. It now returns the same `INTERNAL_ERROR` ("Failed to restore version ...") for an unknown valid-hex SHA as it already did for a malformed one, matching the sibling history method.
- **`add_facts_batch` no longer crashes on a non-string field, and reports it as a per-item error instead.** Batch facts arrive as a list of dicts whose values are not type-coerced, so a non-string `subject`, `predicate`, `object`, or type field reached the entity-name validator and raised an `AttributeError`, aborting the whole batch after some facts had already been committed and returning no accounting. The validator now rejects non-strings with a clear error, which the batch loop records in `errors[]` and skips, so valid facts in the same batch are still added and the documented summary is returned.

## [1.0.32] - 2026-06-20

### Fixed

- **Updating a note's tags, title, or category no longer drops its frontmatter-only links.** An update rebuilt the note's links from the inline `[[uuid]]` references in the body, so when a note had a link in its frontmatter `links` list with no matching inline reference (a hand-edited or imported note), any update that did not change the body silently removed that link. Because backlink, orphan, and broken-link resolution treat frontmatter links as real edges, a single `rename_tag`, `merge_tags`, or `move_category` could strip those edges across every note carrying the affected tag, and a note could even flip to "orphan". An update that leaves the body unchanged now preserves the existing frontmatter links alongside any inline ones.

## [1.0.31] - 2026-06-20

### Fixed

- **A force reindex (`reindex_notes`) no longer destroys the glossary and fact indexes.** Notes, chunks, glossary entries, and facts share a single Qdrant collection, but a forced note reindex cleared it by deleting and recreating the whole collection, wiping every glossary and fact point as collateral. The glossary has no rebuild path, so `search_glossary` returned nothing for every query after a reindex (the entries still existed in the database and `lookup_term`/`list_glossary` still worked, but semantic glossary search stayed empty until each entry was individually re-added or updated). The forced reindex now clears only the note and chunk points it owns (a scoped delete by point type), preserving the glossary and fact points, mirroring how the facts indexer already scopes its own clear.

## [1.0.30] - 2026-06-20

### Fixed

- **`update_fact` now re-indexes the fact, so semantic fact search reflects the update.** `update_fact` changed the fact in the database but never re-indexed it, so `search_facts` (which reads straight from the indexed payload) kept returning the fact's old context, confidence, and validity until a full `index_facts(force=True)`. It now re-indexes the updated fact, best-effort so an unavailable index does not fail the update. This completes the fact index synchronization started in 1.0.29 (`delete_fact`), and relies on vector-core v1.2.11 making `index_fact` atomic so a transient re-index failure cannot drop the fact from search.

### Changed

- Bump vector-core to v1.2.11 (atomic `index_fact` re-indexing).

## [1.0.29] - 2026-06-20

### Fixed

- **`delete_fact` now removes the fact from the semantic index.** Fact search (`search_facts`) reads straight from the indexed payload with no existence check against the store, but `delete_fact` only removed the fact from the database, so a deleted fact kept being returned by `search_facts` (the orphaned point lingered until a full `index_facts(force=True)`). `delete_fact` now also deletes the fact's index point, best-effort so an unavailable index does not fail the delete (mirroring how note and glossary deletes keep their indexes in sync).

## [1.0.28] - 2026-06-20

### Fixed

- Bump vector-core to v1.2.10, which fixes `find_connections` (the `find_connections` notes tool) duplicating an entity reachable through several facts at the maximum depth and crowding out other distinct reachable entities at the result limit.

## [1.0.27] - 2026-06-20

### Fixed

- **Deleting a note while a concurrent rename moves it no longer leaves the note committed in git history.** `delete()` removes the note from the search index with an `await`, which is a suspension point. It captured the note's path *before* that await and then committed the deletion against it. If a concurrent `update` that changes the note's category or title git-moved the note while `delete` was parked, the deletion was committed against the stale old path, so `git rm` failed (the old path was no longer tracked), the error was swallowed, and no delete commit was recorded. The note was gone from disk and the index but its blob remained committed at the new path (a lost delete and a divergent working tree, from which a `git checkout` could resurrect it). `delete()` now resolves the path after the index await, just before the synchronous delete and commit, so it always commits against the note's current path.

## [1.0.26] - 2026-06-20

### Changed

- Bumped `vector-core` to `v1.2.9`. This improves fuzzy query-token matching: vector-core no longer caps fuzzy match candidates before scoring, so a typo or rare token in a note search reliably finds its closest vocabulary term instead of occasionally missing it on a large vocabulary (`vectorize_query`, which the search path uses, has fuzzy on by default). v1.2.9 also makes `FactStore.update_source_status` refuse an unscoped (no-selector) status reset; mcp-notes already guards this at its `revalidate_fact_sources` tool, so this is defense in depth. The v1.2.9 query field-prefix parser fix is not exercised here.

## [1.0.25] - 2026-06-20

### Fixed

- **A note large enough to be split by headers no longer indexes each section's header line twice.** When a note exceeds `max_chunk_chars` it is split into sections by H1/H2 headers. Each section recorded its start line as the header's own line, so the slice that became the section body kept the raw header, and the chunk builder then re-emitted the section title as a header on top of it. The result was the header appearing twice at the start of the chunk (for example `## First` immediately followed by `# First`), and that duplicated text was embedded and tokenized into the search index. Sections now start at the line after their header, so the header appears once. Notes small enough to fit in a single chunk were never affected.

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
