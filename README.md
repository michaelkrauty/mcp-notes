# mcp-notes

MCP server for personal knowledge management with semantic search, git versioning, and knowledge graph capabilities.

Supports stateless MCP `2026-07-28` requests and legacy MCP clients from the same stdio server through the official Python SDK v2.

## Prerequisites

- **Python 3.12+**
- **Linux or macOS** (uses POSIX file locking via vector-core; not compatible with Windows)
- [Qdrant](https://qdrant.tech/) vector database (default: `localhost:6333`)
- An OpenAI-compatible embedding API (e.g., llama.cpp, Ollama, or any `/v1/embeddings` endpoint; default: `localhost:8080`)
- **git** (must be on PATH for versioning features)

## Installation

Requires [vector-core](https://github.com/michaelkrauty/vector-core).

```bash
pip install git+https://github.com/michaelkrauty/vector-core.git@v1.8.0
pip install git+https://github.com/michaelkrauty/mcp-notes.git
```

Or clone both repos and install locally:

```bash
git clone https://github.com/michaelkrauty/vector-core.git
git clone https://github.com/michaelkrauty/mcp-notes.git
pip install -e vector-core/
pip install -e mcp-notes/
```

## Quick Start

```bash
# Register with Claude Code:
claude mcp add notes -- mcp-notes

# Or add to your MCP client config (e.g., claude_desktop_config.json):
# {
#   "mcpServers": {
#     "notes": {
#       "command": "mcp-notes",
#       "env": {
#         "VECTOR_QDRANT_URL": "http://localhost:6333",
#         "VECTOR_EMBEDDING_URL": "http://localhost:8080",
#         "VECTOR_EMBEDDING_MODEL": "your-model-name"
#       }
#     }
#   }
# }
```

## Features

- **Note Management**: CRUD operations on markdown notes with YAML frontmatter
- **Semantic Search**: Hybrid dense + sparse vector search via Qdrant
- **Git Versioning**: Automatic commits on changes, full history with `--follow`
- **Wiki Linking**: `[[uuid]]` syntax for inter-note references with backlink tracking
- **Tags & Categories**: Flexible organization with tag management tools
- **Glossary**: Shared term definitions with aliases and domains
- **Fact Graph**: Subject-predicate-object triples with source tracking

## Tools (38 total)

### Notes (4)
| Tool | Description |
|------|-------------|
| `create_note` | Create note with auto-generated UUID |
| `read_note` | Read note by UUID |
| `update_note` | Update title, content, tags, or category |
| `delete_note` | Delete note (keeps git history) |

### Search (3)
| Tool | Description |
|------|-------------|
| `search_notes` | Hybrid semantic + keyword search with filters |
| `list_notes` | List notes with tag/category filters |
| `find_similar_notes` | Find semantically similar notes |

### Versioning (2)
| Tool | Description |
|------|-------------|
| `get_note_history` | Git commit history for a note |
| `restore_note_version` | Restore note to previous commit |

### Links (1)
| Tool | Description |
|------|-------------|
| `get_note_links` | Outgoing, incoming (backlinks), and broken links |

### Tags (3)
| Tool | Description |
|------|-------------|
| `list_tags` | All tags with note counts |
| `rename_tag` | Rename tag across all notes |
| `merge_tags` | Merge multiple tags into one |

### Categories (2)
| Tool | Description |
|------|-------------|
| `list_categories` | Category hierarchy with counts |
| `move_category` | Move/rename a category |

### Glossary (6)
| Tool | Description |
|------|-------------|
| `add_glossary_entry` | Add term with expansion, definition, domain |
| `lookup_term` | Exact lookup by term or alias |
| `search_glossary` | Semantic glossary search |
| `list_glossary` | List entries with optional domain filter |
| `update_glossary_entry` | Modify entry metadata |
| `delete_glossary_entry` | Delete entry |

### Facts (11)
| Tool | Description |
|------|-------------|
| `add_fact` | Add SPO triple with source tracking |
| `add_facts_batch` | Batch import facts |
| `update_fact` | Update fact metadata |
| `delete_fact` | Delete fact and sources |
| `query_facts` | Query by subject/predicate/object |
| `get_entity` | All facts involving an entity |
| `list_facts` | List fact summaries |
| `search_facts` | Semantic fact search |
| `index_facts` | Index facts for search |
| `find_connections` | BFS graph traversal between entities |
| `get_neighbors` | Immediate entity connections |

### Integrity (4)
| Tool | Description |
|------|-------------|
| `get_facts_with_stale_sources` | Facts with deleted/modified sources |
| `get_source_statistics` | Source integrity stats |
| `check_fact_integrity` | Check specific fact's sources |
| `revalidate_fact_sources` | Reset sources after verification |

### Health (2)
| Tool | Description |
|------|-------------|
| `reindex_notes` | Force full reindex |
| `check_note_health` | Validate note structure, find errors |

## Data Model

### Note
```yaml
---
id: {uuid}
title: Note Title
created: 2024-01-01T00:00:00Z
modified: 2024-01-05T12:00:00Z
tags: [tag1, tag2]
links: [linked-uuid]
---
Markdown content with [[uuid]] links.
```

### Fact (SPO Triple)
```
subject: "Ada Lovelace" (type: person)
predicate: "works_at"
object: "Babbage Labs" (type: organization)
context: "as lead engineer"
confidence: 1.0
valid_from/to: date range
sources: [{type: note, id: uuid, location: "paragraph 3"}]
```

## Storage

| Data | Location |
|------|----------|
| Notes | `{NOTES_DIR}/notes/` (default `~/notes/notes/`) |
| UUID index | `{NOTES_DIR}/.index/` |
| Vectors | Qdrant collection `notes_{hash}` |
| Git repo | `{NOTES_DIR}/.git/` |
| Facts & glossary | `~/.local/share/vector-core/` (configurable via `VECTOR_SHARED_DATA_DIR`) |

`NOTES_DIR` (default `~/notes`) is the base directory. Actual note files live in the `notes/` subdirectory within it. The base directory also contains `.index/`, `.git/`, and `.locks/`.

Notes stored as `{slug}-{uuid}.md` organized by category subdirectories.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NOTES_DIR` | `~/notes` | Notes storage directory |
| `NOTES_GIT_ENABLED` | `true` | Enable git versioning |
| `NOTES_GIT_USER_NAME` | `Notes MCP` | Git commit author |
| `NOTES_GIT_USER_EMAIL` | `notes@localhost` | Git commit email |
| `NOTES_AUTO_INDEX` | `true` | Auto-index on startup |
| `NOTES_COLLECTION_PREFIX` | `notes` | Qdrant collection prefix |

Plus inherited vector-core settings (`VECTOR_QDRANT_URL`, `VECTOR_EMBEDDING_URL`, etc.).

### Changing embedding models

Update the vector-core embedding configuration and restart the client. The first search or write automatically builds a compatible physical collection, including notes, chunks, glossary entries, facts, and other retained content in a shared collection. This also works with `NOTES_AUTO_INDEX=false`. There is no separate destructive reindex step.

Model, resolved dimension, deployment namespace, and input formatting are part of the embedding identity. Change the deployment namespace whenever an unchanged model alias points to different weights. Reads and writes use a fixed physical generation; an older client that has been superseded must restart before further operations. Stop legacy server versions before changing configuration because those versions do not participate in migration locking.

The original collection is retained. Failed builds remain inactive, and retrying creates a fresh candidate. New records preserve complete embedding input. Retained inputs larger than the configured model capacity use independently searchable fragments rather than truncation or averaged vectors. Legacy records require sufficient retained text or readable original source; missing source and historically truncated payloads can prevent complete reconstruction and must be surfaced. When local source changes are detected, complete note/chunk groups are refreshed without changing unrelated record types. Their vocabulary contribution is reconciled by the next normal full indexing pass. Missing files or unavailable source directories never imply deletion during migration: complete retained groups survive, while legacy metadata-only summaries without a reconstructable body fail explicitly.

Note-level semantic search ranks complete passage matches and returns distinct notes. Similar-note lookup compares all indexed source passages and uses the best passage-pair score, excluding the source note before retrieval. The short note summary is auxiliary; it does not limit body coverage. Chunk text retains exact body spans, including oversized paragraphs and header-only sections, and highlights use full retained searchable text. Limits include repeated metadata and model role formatting. Metadata that cannot fit as repeated context produces an explicit error rather than being silently shortened.

Explicit fact and glossary searches return distinct entities. Default note/chunk and mixed all-types searches return distinct original source records, grouping their fragments while preserving markerless legacy matches. Winning fact and glossary fragments use lineage-validated canonical display fields and keep highlights from the matching snippet. Similar-note queries use bounded concurrency without limiting source coverage; a failed batch is canceled and joined before the error propagates.

For exact tokenizer-based input limits, install the optional extra with `uv sync --extra tokenizer` or `pip install 'mcp-notes[tokenizer]'`, then configure vector-core's tokenizer and token limit. Without that extra, vector-core uses its conservative fallback limit. Query and document formatting is handled by the embedding client; callers should provide ordinary, unprefixed text.

## Search Query Syntax

```
# Filter by tag
tag:project-x

# Exclude a tag
-tag:archived

# Filter by category (exact match)
category:work/projects

# Date filters
after:2024-01-01
before:2024-06-30

# Title search
title:meeting notes

# Combined
project tag:active category:work after:2024-01-01
```

## MCP Resources

Static data endpoints:
- `notes://index` - Full note index
- `notes://tags` - All tags with counts
- `notes://categories` - Category hierarchy
- `notes://recent` - Last 20 modified notes
- `notes://orphans` - Notes with no backlinks
- `notes://broken-links` - Broken reference summary
- `notes://parse-errors` - Notes that failed to parse

## Dependencies

Requires vector-core components:
- EmbeddingClient, GlobalVocabulary (search)
- QdrantStorage (storage)
- GlossaryStore, FactStore (knowledge graph)

External libraries:
- GitPython (versioning)
