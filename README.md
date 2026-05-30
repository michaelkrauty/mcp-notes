# mcp-notes

MCP server for personal knowledge management with semantic search, git versioning, and knowledge graph capabilities.

## Prerequisites

- **Python 3.12+**
- **Linux or macOS** (uses POSIX file locking via vector-core; not compatible with Windows)
- [Qdrant](https://qdrant.tech/) vector database (default: `localhost:6333`)
- An OpenAI-compatible embedding API (e.g., llama.cpp, Ollama, or any `/v1/embeddings` endpoint; default: `localhost:8080`)
- **git** (must be on PATH for versioning features)

## Installation

Requires [vector-core](https://github.com/michaelkrauty/vector-core).

```bash
pip install git+https://github.com/michaelkrauty/vector-core.git@v1.1.0
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
