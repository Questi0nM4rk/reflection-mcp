# Plan: Migrate to libSQL

## Changes

Replace JSON file storage with libSQL.

### Files to Modify

- `src/reflection_mcp/server.py`:
  - Remove file I/O operations
  - Add libSQL for episodes table
  - Add vector search for retrieve_episodes

### Dependencies

```toml
[project.dependencies]
libsql-experimental = ">=0.0.50"

[project.optional-dependencies]
full = ["sentence-transformers>=2.0.0"]
```

### Database Location

`~/.codeagent/codeagent.db` (shared with other MCPs)

### Schema

```sql
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT UNIQUE NOT NULL,
    task TEXT NOT NULL,
    approach TEXT NOT NULL,
    outcome TEXT NOT NULL,
    feedback TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    reflection TEXT,       -- JSON object
    code_context TEXT,
    file_path TEXT,
    tags TEXT,             -- JSON array
    duration_seconds REAL,
    embedding F32_BLOB(384),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lesson_effectiveness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    led_to_success INTEGER NOT NULL,
    effectiveness_score REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE TABLE IF NOT EXISTS episode_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    lesson_episode_id TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id),
    FOREIGN KEY (lesson_episode_id) REFERENCES episodes(episode_id)
);

CREATE INDEX IF NOT EXISTS libsql_vector_idx_episodes ON episodes(embedding);
CREATE INDEX IF NOT EXISTS idx_episodes_feedback_type ON episodes(feedback_type);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
```

### Embedding Model

Use `all-MiniLM-L6-v2` (384 dimensions) from sentence-transformers.
Embed the task + approach + feedback for similarity search.

### Verification

- `uv run pytest tests/ -v`
- Manual: store_episode, retrieve_episodes, get_common_lessons
