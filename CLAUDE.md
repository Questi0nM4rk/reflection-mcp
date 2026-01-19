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

## Git Workflow

**IMPORTANT**: Direct push to `main` is disabled. All changes must go through PRs.

1. Create a feature branch: `git checkout -b feat/description`
2. Make changes and commit
3. Push branch: `git push -u origin feat/description`
4. Create PR via `gh pr create` or GitHub UI
5. CodeRabbit provides automated review
6. After addressing comments, **resolve each GitHub conversation**
7. Merge when approved

### CodeRabbit Tips

- Push fixes for review comments
- Click "Resolve conversation" on each addressed comment
- CodeRabbit auto-approves when all resolved and checks pass
