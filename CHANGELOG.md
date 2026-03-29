# Changelog

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
