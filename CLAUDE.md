# Reflection MCP

Self-reflection MCP server with persistent episodic memory and lesson effectiveness tracking.

## Structure

| File | Purpose |
|------|---------|
| `src/reflection_mcp/server.py` | MCP server with all tool implementations |
| `src/reflection_mcp/__init__.py` | Package exports |
| `tests/test_server.py` | Server tests |
| `pyproject.toml` | Package config (hatch build) |

## Commands

```bash
# Run tests
uv run pytest tests/ -v

# Format/lint
uv run ruff format .
uv run ruff check .

# Run server directly
uv run reflection-mcp
```

## Dependencies

- `mcp>=1.0.0` - Model Context Protocol SDK
- Python 3.10+
