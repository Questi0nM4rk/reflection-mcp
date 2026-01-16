# reflection-mcp

Self-reflection MCP server with persistent episodic memory.
Based on Reflexion research showing +21% improvement on HumanEval.

## Features

- **Persistent Storage**: Episodes saved to `~/.codeagent/data/reflection-episodes/`
- **Lesson Effectiveness Tracking**: Track if applying lessons leads to success
- **Cross-Session Learning**: Aggregate patterns across sessions
- **Learner Skill Integration**: Export lessons for project memory

## MCP Tools

| Tool | Description |
|------|-------------|
| `reflect_on_failure` | Generate structured reflection on failure |
| `store_episode` | Store learning episode in memory |
| `retrieve_episodes` | Find similar past episodes |
| `generate_improved_attempt` | Generate guidance for improvement |
| `get_reflection_history` | View reflection history |
| `get_common_lessons` | Get aggregated lessons by type |
| `clear_episodes` | Clear episodic memory |
| `get_episode_stats` | Get statistics about memory |
| `mark_lesson_effective` | Track lesson effectiveness |
| `export_lessons` | Export for learner skill |
| `link_episode_to_lesson` | Link episode to applied lesson |

## Feedback Types

| Type | Description |
|------|-------------|
| `test_failure` | Test assertion failed |
| `lint_error` | Linter rule violation |
| `build_error` | Compilation/build failure |
| `review_comment` | Code review feedback |
| `runtime_error` | Runtime exception |
| `security_issue` | Security vulnerability |
| `performance_issue` | Performance problem |
| `type_error` | Type mismatch |

## Philosophy

Based on [Reflexion](https://arxiv.org/abs/2303.11366) (NeurIPS 2023) - "Language Agents with Verbal Reinforcement Learning" by Shinn et al.

**Key insight**: +21% accuracy on code tasks through structured failure analysis and episodic learning. Instead of retrying blindly, reflect on what went wrong.

## Installation

```bash
pip install git+https://github.com/Questi0nM4rk/reflection-mcp.git
```

## Usage with Claude Code

```bash
claude mcp add reflection -- python -m reflection_mcp.server
```

Or run standalone:

```bash
python -m reflection_mcp.server
```

## Environment Variables

```bash
CODEAGENT_HOME  # Storage location (default: ~/.codeagent)
```

## Storage

Episodes and lessons are persisted as JSON files in:
```
~/.codeagent/data/reflection-episodes/
├── episodes.json    # All episodes
└── lessons.json     # Aggregated patterns
```
