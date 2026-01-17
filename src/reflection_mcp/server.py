"""
Self-Reflection MCP Server

Implements Reflexion pattern with persistent episodic memory.
Based on NeurIPS 2023 research showing 67% → 88% improvement on HumanEval (+21%).

Features:
- Persistent libSQL storage with optional vector search
- Lesson effectiveness tracking (did applying a lesson lead to success?)
- Cross-session learning aggregation
- Export capability for learner skill integration

Database: ~/.codeagent/codeagent.db

Tools:
- reflect_on_failure: Analyze why output failed and generate insights
- store_episode: Store a learning episode in memory
- retrieve_episodes: Find similar past episodes for learning
- generate_improved_attempt: Generate improved solution using reflection
- get_reflection_history: View reflection history for a task
- get_common_lessons: Get aggregated lessons by feedback type
- clear_episodes: Clear episodic memory
- get_episode_stats: Get statistics about episodic memory
- export_lessons: Export lessons for learner skill
- mark_lesson_effective: Track if applying a lesson led to success
"""

import json
import logging
import os
import struct
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

import libsql_experimental as libsql
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
CODEAGENT_DIR = Path(os.environ.get("CODEAGENT_HOME", Path.home() / ".codeagent"))
CODEAGENT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CODEAGENT_DIR / "codeagent.db"

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Initialize FastMCP server
mcp = FastMCP(
    "reflection",
    instructions="Self-reflection and episodic memory for learning from failures. "
    "Use this when code fails tests or reviews to improve on subsequent attempts.",
)

# Lazy-loaded singleton
_embedding_model = None


def _get_embedding_model():
    """Get or load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except ImportError:
        logger.debug("sentence-transformers not installed, using keyword similarity")
        _embedding_model = None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        _embedding_model = None

    return _embedding_model


class FeedbackType(str, Enum):
    """Types of feedback that trigger reflection."""

    TEST_FAILURE = "test_failure"
    LINT_ERROR = "lint_error"
    BUILD_ERROR = "build_error"
    REVIEW_COMMENT = "review_comment"
    RUNTIME_ERROR = "runtime_error"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    TYPE_ERROR = "type_error"


class OutcomeType(str, Enum):
    """Outcome of an episode."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


@contextmanager
def _get_db() -> Generator[libsql.Connection, None, None]:
    """Get database connection as context manager, creating schema if needed."""
    conn = libsql.connect(str(DB_PATH))
    try:
        _init_schema(conn)
        yield conn
    finally:
        conn.close()


def _init_schema(conn: libsql.Connection) -> None:
    """Initialize database schema."""
    # Enable foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT UNIQUE NOT NULL,
            task TEXT NOT NULL,
            approach TEXT NOT NULL,
            outcome TEXT NOT NULL,
            feedback TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            reflection TEXT,
            code_context TEXT,
            file_path TEXT,
            attempt_number INTEGER DEFAULT 1,
            duration_seconds REAL,
            tags TEXT,
            lesson_applied_from TEXT,
            led_to_success INTEGER,
            effectiveness_score REAL DEFAULT 0.0,
            embedding F32_BLOB({EMBEDDING_DIM}),
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS lesson_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            led_to_success INTEGER NOT NULL,
            effectiveness_score REAL DEFAULT 0.5,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS episode_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            lesson_episode_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE,
            FOREIGN KEY (lesson_episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS task_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_key TEXT UNIQUE NOT NULL,
            attempt_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_feedback_type ON episodes(feedback_type);
        CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
        CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
    """)

    # Create vector index separately
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS libsql_vector_idx_episodes ON episodes(embedding)"
        )
    except Exception as e:
        logger.debug(f"Vector index may already exist: {e}")

    conn.commit()


def _embed(text: str) -> bytes | None:
    """Generate embedding for text, returns F32_BLOB bytes."""
    model = _get_embedding_model()
    if model is None:
        return None

    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return struct.pack(f"<{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def _json_loads(val: str | None) -> dict | list | None:
    """Parse JSON or return None."""
    if val is None:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def _json_dumps(val: dict | list | None) -> str | None:
    """Serialize to JSON or return None."""
    if val is None:
        return None
    return json.dumps(val)


def _get_attempt_number(conn: libsql.Connection, task: str) -> int:
    """Get and increment attempt number for a task using atomic UPSERT."""
    task_key = task[:100]

    # Atomic UPSERT: insert with count=1, or increment on conflict
    cursor = conn.execute(
        """
        INSERT INTO task_attempts (task_key, attempt_count)
        VALUES (?, 1)
        ON CONFLICT(task_key) DO UPDATE SET attempt_count = attempt_count + 1
        RETURNING attempt_count
        """,
        (task_key,),
    )
    row = cursor.fetchone()
    return row[0] if row else 1


def _keyword_similarity(text1: str, text2: str) -> float:
    """Simple keyword-based similarity score."""
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    }

    words1 = set(text1.lower().split()) - stopwords
    words2 = set(text2.lower().split()) - stopwords

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


# ============================================
# MCP Tools
# ============================================


@mcp.tool()
def reflect_on_failure(
    output: str,
    feedback: str,
    feedback_type: str = "test_failure",
    context: str = "",
) -> dict[str, Any]:
    """
    Generate a structured reflection on why output failed.

    Args:
        output: The code or output that failed (can be truncated)
        feedback: The error message or feedback received
        feedback_type: Type of feedback (test_failure, lint_error, build_error, etc.)
        context: Additional context about what was being attempted

    Returns:
        Structured reflection with insights for improvement
    """
    reflection_id = str(uuid4())[:8]

    try:
        fb_type = FeedbackType(feedback_type)
    except ValueError:
        fb_type = FeedbackType.TEST_FAILURE

    return {
        "reflection_id": reflection_id,
        "output_summary": output[:500] if len(output) > 500 else output,
        "feedback": feedback,
        "feedback_type": fb_type.value,
        "context": context,
        "reflection_template": {
            "what_went_wrong": "[Describe the specific failure - what didn't work as expected]",
            "root_cause": "[Identify the underlying cause - why did this approach fail]",
            "what_to_try_next": "[Concrete next steps - what specific changes should be made]",
            "general_lesson": "[Broader lesson - what principle or pattern should be remembered]",
            "confidence": "[0.0-1.0 - how confident are you in this analysis]",
        },
        "guidance": {
            "test_failure": "Focus on what assertion failed and why the code doesn't satisfy the test.",
            "lint_error": "Identify the specific rule violation and the correct pattern.",
            "build_error": "Focus on type mismatches, missing imports, or syntax issues.",
            "review_comment": "Consider the reviewer's perspective and what they're trying to improve.",
            "runtime_error": "Focus on the error type and what state led to the failure.",
            "security_issue": "Identify the vulnerability class and secure alternatives.",
            "performance_issue": "Identify the bottleneck and more efficient approaches.",
            "type_error": "Focus on type mismatches and correct type annotations.",
        }.get(fb_type.value, "Analyze the failure systematically."),
        "instruction": "Complete the reflection_template with specific insights. "
        "Then call store_episode to save this for future learning.",
    }


@mcp.tool()
def store_episode(
    task: str,
    approach: str,
    outcome: str,
    feedback: str,
    feedback_type: str,
    reflection: dict[str, Any],
    code_context: str = "",
    file_path: str | None = None,
    tags: list[str] | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Store a learning episode in episodic memory.

    Args:
        task: Description of the task being attempted
        approach: The approach that was tried
        outcome: Result - 'success', 'partial', or 'failure'
        feedback: The feedback or error message
        feedback_type: Type of feedback
        reflection: Reflection dict with what_went_wrong, root_cause, what_to_try_next, general_lesson
        code_context: Relevant code snippet (truncated if needed)
        file_path: Path to the file being modified
        tags: Optional tags for categorization
        duration_seconds: How long the attempt took

    Returns:
        Stored episode ID and confirmation
    """
    try:
        with _get_db() as conn:
            episode_id = str(uuid4())[:8]

            try:
                outcome_type = OutcomeType(outcome)
            except ValueError:
                outcome_type = OutcomeType.FAILURE

            try:
                fb_type = FeedbackType(feedback_type)
            except ValueError:
                fb_type = FeedbackType.TEST_FAILURE

            attempt_number = _get_attempt_number(conn, task)

            # Generate embedding from task + approach + feedback
            embed_text = f"{task} {approach} {feedback}"
            embedding = _embed(embed_text)

            now = datetime.now().isoformat()

            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, task, approach, outcome, feedback, feedback_type,
                    reflection, code_context, file_path, attempt_number,
                    duration_seconds, tags, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    task,
                    approach,
                    outcome_type.value,
                    feedback[:2000],
                    fb_type.value,
                    _json_dumps(reflection),
                    code_context[:3000] if code_context else None,
                    file_path,
                    attempt_number,
                    duration_seconds,
                    _json_dumps(tags),
                    embedding,
                    now,
                ),
            )
            conn.commit()

            # Count total episodes
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            total = cursor.fetchone()[0]

            return {
                "episode_id": episode_id,
                "task": task[:100],
                "attempt_number": attempt_number,
                "outcome": outcome_type.value,
                "stored": True,
                "total_episodes": total,
                "lesson": reflection.get("general_lesson", ""),
                "storage": str(DB_PATH),
            }

    except Exception as e:
        logger.error(f"Failed to store episode: {e}")
        return {"stored": False, "error": str(e)}


@mcp.tool()
def retrieve_episodes(
    task: str,
    error_pattern: str = "",
    feedback_type: str | None = None,
    top_k: int = 5,
    include_successes: bool = True,
) -> dict[str, Any]:
    """
    Retrieve similar past episodes for learning.

    Args:
        task: Current task description
        error_pattern: Optional error message to match
        feedback_type: Filter by feedback type
        top_k: Maximum episodes to return
        include_successes: Whether to include successful episodes

    Returns:
        Relevant past episodes with their reflections
    """
    try:
        with _get_db() as conn:
            top_k = min(max(top_k, 1), 20)

            results = []

            # Try vector search first
            embed_text = f"{task} {error_pattern}"
            embedding = _embed(embed_text)

            if embedding is not None:
                try:
                    sql = """
                        SELECT episode_id, task, approach, outcome, feedback_type,
                               attempt_number, reflection, created_at,
                               vector_distance_cos(embedding, ?) as distance
                        FROM episodes
                        WHERE embedding IS NOT NULL
                    """
                    params: list[Any] = [embedding]

                    if feedback_type:
                        sql += " AND feedback_type = ?"
                        params.append(feedback_type)

                    if not include_successes:
                        sql += " AND outcome != 'success'"

                    sql += " ORDER BY distance ASC LIMIT ?"
                    params.append(top_k)

                    cursor = conn.execute(sql, params)

                    for row in cursor.fetchall():
                        (
                            ep_id,
                            ep_task,
                            ep_approach,
                            ep_outcome,
                            ep_fb_type,
                            ep_attempt,
                            ep_refl_json,
                            ep_created,
                            distance,
                        ) = row

                        refl = _json_loads(ep_refl_json)
                        score = 1.0 / (1.0 + distance) if distance else 1.0

                        results.append(
                            {
                                "episode_id": ep_id,
                                "task": ep_task[:200],
                                "approach": ep_approach[:200],
                                "outcome": ep_outcome,
                                "feedback_type": ep_fb_type,
                                "attempt_number": ep_attempt,
                                "similarity_score": round(score, 3),
                                "reflection": {
                                    "what_went_wrong": refl.get("what_went_wrong", "")
                                    if refl
                                    else "",
                                    "root_cause": refl.get("root_cause", "")
                                    if refl
                                    else "",
                                    "what_to_try_next": refl.get("what_to_try_next", "")
                                    if refl
                                    else "",
                                    "general_lesson": refl.get("general_lesson", "")
                                    if refl
                                    else "",
                                }
                                if refl
                                else None,
                                "created_at": ep_created,
                            }
                        )

                except Exception as e:
                    logger.debug(f"Vector search failed: {e}")

            # Fallback: keyword similarity
            if not results:
                sql = "SELECT episode_id, task, approach, outcome, feedback, feedback_type, attempt_number, reflection, created_at FROM episodes"
                params = []

                conditions = []
                if feedback_type:
                    conditions.append("feedback_type = ?")
                    params.append(feedback_type)

                if not include_successes:
                    conditions.append("outcome != 'success'")

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

                sql += " LIMIT 100"

                cursor = conn.execute(sql, params)

                scored = []
                for row in cursor.fetchall():
                    (
                        ep_id,
                        ep_task,
                        ep_approach,
                        ep_outcome,
                        ep_feedback,
                        ep_fb_type,
                        ep_attempt,
                        ep_refl_json,
                        ep_created,
                    ) = row

                    task_sim = _keyword_similarity(task, ep_task)
                    error_sim = (
                        _keyword_similarity(error_pattern, ep_feedback)
                        if error_pattern
                        else 0
                    )
                    score = task_sim * 0.7 + error_sim * 0.3

                    refl = _json_loads(ep_refl_json)

                    scored.append(
                        (
                            score,
                            {
                                "episode_id": ep_id,
                                "task": ep_task[:200],
                                "approach": ep_approach[:200],
                                "outcome": ep_outcome,
                                "feedback_type": ep_fb_type,
                                "attempt_number": ep_attempt,
                                "similarity_score": round(score, 3),
                                "reflection": {
                                    "what_went_wrong": refl.get("what_went_wrong", "")
                                    if refl
                                    else "",
                                    "root_cause": refl.get("root_cause", "")
                                    if refl
                                    else "",
                                    "what_to_try_next": refl.get("what_to_try_next", "")
                                    if refl
                                    else "",
                                    "general_lesson": refl.get("general_lesson", "")
                                    if refl
                                    else "",
                                }
                                if refl
                                else None,
                                "created_at": ep_created,
                            },
                        )
                    )

                scored.sort(key=lambda x: x[0], reverse=True)
                results = [ep for _, ep in scored[:top_k]]

            return {
                "query_task": task[:100],
                "query_error": error_pattern[:100] if error_pattern else None,
                "episodes": results,
                "count": len(results),
                "instruction": "Apply lessons from these past episodes to your current approach. "
                "Focus on the general_lesson and what_to_try_next fields.",
            }

    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        return {"query_task": task[:100], "episodes": [], "count": 0, "error": str(e)}


@mcp.tool()
def generate_improved_attempt(
    original_output: str,
    reflection: dict[str, Any],
    similar_episodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Generate guidance for an improved attempt based on reflection and past lessons.

    Args:
        original_output: The code/output that failed
        reflection: Current reflection (from reflect_on_failure)
        similar_episodes: Past episodes to learn from (from retrieve_episodes)

    Returns:
        Guidance for generating an improved solution
    """
    past_lessons = []
    if similar_episodes:
        for ep in similar_episodes:
            if ep.get("reflection") and ep["reflection"].get("general_lesson"):
                past_lessons.append(
                    {
                        "task": ep.get("task", "")[:80],
                        "lesson": ep["reflection"]["general_lesson"],
                        "what_worked": ep["reflection"].get("what_to_try_next", ""),
                    }
                )

    return {
        "original_output_summary": original_output[:300],
        "current_reflection": {
            "root_cause": reflection.get("root_cause", "Unknown"),
            "what_to_try_next": reflection.get("what_to_try_next", ""),
        },
        "past_lessons": past_lessons[:5],
        "improvement_strategy": {
            "1_address_root_cause": f"Fix: {reflection.get('root_cause', 'the identified issue')}",
            "2_apply_lesson": reflection.get(
                "what_to_try_next", "Apply the learned approach"
            ),
            "3_avoid_patterns": "Don't repeat approaches that failed in similar past episodes",
            "4_verify_before_submit": "Ensure the fix addresses the original feedback",
        },
        "instruction": "Generate improved code that: "
        "1) Addresses the root cause identified in reflection, "
        "2) Applies lessons from similar past episodes, "
        "3) Avoids patterns that led to failures before. "
        "Focus on correctness over cleverness.",
    }


@mcp.tool()
def get_reflection_history(
    task: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    View reflection history, optionally filtered by task.

    Args:
        task: Optional task to filter by (substring match)
        limit: Maximum entries to return

    Returns:
        History of reflections with outcomes
    """
    try:
        with _get_db() as conn:
            limit = min(max(limit, 1), 50)

            if task:
                cursor = conn.execute(
                    """
                    SELECT episode_id, task, outcome, feedback_type, attempt_number,
                           reflection, created_at
                    FROM episodes
                    WHERE task LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{task}%", limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT episode_id, task, outcome, feedback_type, attempt_number,
                           reflection, created_at
                    FROM episodes
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            history = []
            for row in cursor.fetchall():
                (
                    ep_id,
                    ep_task,
                    ep_outcome,
                    ep_fb_type,
                    ep_attempt,
                    ep_refl_json,
                    ep_created,
                ) = row
                refl = _json_loads(ep_refl_json)

                history.append(
                    {
                        "episode_id": ep_id,
                        "task": ep_task[:100],
                        "outcome": ep_outcome,
                        "feedback_type": ep_fb_type,
                        "attempt_number": ep_attempt,
                        "lesson": refl.get("general_lesson") if refl else None,
                        "created_at": ep_created,
                    }
                )

            # Calculate stats
            cursor = conn.execute(
                """
                SELECT outcome, COUNT(*) FROM episodes
                WHERE task LIKE ?
                GROUP BY outcome
                """,
                (f"%{task}%" if task else "%",),
            )
            outcome_counts = dict(cursor.fetchall())

            stats = {
                "total": sum(outcome_counts.values()),
                "successes": outcome_counts.get("success", 0),
                "partial": outcome_counts.get("partial", 0),
                "failures": outcome_counts.get("failure", 0),
            }

            return {"history": history, "stats": stats, "filter": task}

    except Exception as e:
        return {"history": [], "stats": {}, "error": str(e)}


@mcp.tool()
def get_common_lessons() -> dict[str, Any]:
    """
    Get aggregated lessons from all episodes, grouped by feedback type.

    Returns:
        Common lessons learned, organized by feedback type
    """
    try:
        with _get_db() as conn:
            cursor = conn.execute(
                "SELECT feedback_type, reflection FROM episodes WHERE reflection IS NOT NULL"
            )

            lessons_by_type: dict[str, list[str]] = {}
            for row in cursor.fetchall():
                fb_type, refl_json = row
                refl = _json_loads(refl_json)
                if refl and refl.get("general_lesson"):
                    if fb_type not in lessons_by_type:
                        lessons_by_type[fb_type] = []
                    lessons_by_type[fb_type].append(refl["general_lesson"])

            # Deduplicate
            formatted = {}
            for fb_type, lessons in lessons_by_type.items():
                unique = list(set(lessons))
                formatted[fb_type] = unique[:10]

            # Get lesson effectiveness
            cursor = conn.execute(
                """
                SELECT reflection, feedback_type,
                       COUNT(*) as occurrences,
                       SUM(CASE WHEN led_to_success = 1 THEN 1 ELSE 0 END) as successes
                FROM episodes
                WHERE reflection IS NOT NULL AND led_to_success IS NOT NULL
                GROUP BY reflection
                ORDER BY occurrences DESC
                LIMIT 10
                """
            )

            effective_lessons = []
            for row in cursor.fetchall():
                refl_json, fb_type, occurrences, successes = row
                refl = _json_loads(refl_json)
                if refl and refl.get("general_lesson"):
                    success_rate = successes / occurrences if occurrences > 0 else 0
                    effective_lessons.append(
                        {
                            "lesson": refl["general_lesson"],
                            "feedback_type": fb_type,
                            "occurrences": occurrences,
                            "success_rate": round(success_rate, 3),
                        }
                    )

            # Get total count
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            total = cursor.fetchone()[0]

            return {
                "lessons_by_feedback_type": formatted,
                "most_effective_lessons": effective_lessons,
                "total_episodes": total,
                "storage": str(DB_PATH),
                "instruction": "Use these accumulated lessons to avoid repeating past mistakes.",
            }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def clear_episodes(
    older_than_days: int | None = None,
    feedback_type: str | None = None,
) -> dict[str, Any]:
    """
    Clear episodic memory, optionally with filters.

    Args:
        older_than_days: Only clear episodes older than N days
        feedback_type: Only clear episodes of this feedback type

    Returns:
        Number of episodes cleared
    """
    try:
        with _get_db() as conn:
            # Get initial count
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            initial = cursor.fetchone()[0]

            if older_than_days is None and feedback_type is None:
                conn.execute("DELETE FROM episodes")
                conn.execute("DELETE FROM task_attempts")
                conn.commit()
                return {"cleared": initial, "remaining": 0}

            conditions = []
            params: list[Any] = []

            if older_than_days is not None:
                conditions.append("created_at < datetime('now', ? || ' days')")
                params.append(f"-{older_than_days}")

            if feedback_type is not None:
                conditions.append("feedback_type = ?")
                params.append(feedback_type)

            sql = f"DELETE FROM episodes WHERE {' AND '.join(conditions)}"
            conn.execute(sql, params)
            conn.commit()

            # Get remaining count
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            remaining = cursor.fetchone()[0]

            return {"cleared": initial - remaining, "remaining": remaining}

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_episode_stats() -> dict[str, Any]:
    """
    Get statistics about episodic memory.

    Returns:
        Statistics including counts, success rates, common failure types
    """
    try:
        with _get_db() as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            total = cursor.fetchone()[0]

            if total == 0:
                return {
                    "message": "No episodes stored yet.",
                    "total": 0,
                    "storage": str(DB_PATH),
                }

            # By outcome
            cursor = conn.execute(
                "SELECT outcome, COUNT(*) FROM episodes GROUP BY outcome"
            )
            outcomes = dict(cursor.fetchall())

            # By feedback type
            cursor = conn.execute(
                "SELECT feedback_type, COUNT(*) FROM episodes GROUP BY feedback_type"
            )
            feedback_types = dict(cursor.fetchall())

            # Success rate
            success_rate = outcomes.get("success", 0) / total if total > 0 else 0

            # Average attempts
            cursor = conn.execute("SELECT AVG(attempt_number) FROM episodes")
            avg_attempts = cursor.fetchone()[0] or 0

            # Most common failures
            cursor = conn.execute(
                """
                SELECT feedback_type, COUNT(*) as cnt FROM episodes
                WHERE outcome = 'failure'
                GROUP BY feedback_type
                ORDER BY cnt DESC
                LIMIT 5
                """
            )
            failure_types = dict(cursor.fetchall())

            # Lesson effectiveness
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE lesson_applied_from IS NOT NULL) as applied,
                    COUNT(*) FILTER (WHERE led_to_success = 1) as effective
                FROM episodes
                """
            )
            row = cursor.fetchone()
            lessons_applied = row[0] if row else 0
            lessons_effective = row[1] if row else 0
            effectiveness = (
                lessons_effective / lessons_applied if lessons_applied > 0 else 0
            )

            return {
                "total_episodes": total,
                "by_outcome": outcomes,
                "by_feedback_type": feedback_types,
                "success_rate": round(success_rate, 3),
                "average_attempts": round(avg_attempts, 2),
                "most_common_failures": failure_types,
                "lesson_effectiveness": {
                    "lessons_applied": lessons_applied,
                    "led_to_success": lessons_effective,
                    "effectiveness_rate": round(effectiveness, 3),
                },
                "storage": str(DB_PATH),
            }

    except Exception as e:
        return {"error": str(e), "storage": str(DB_PATH)}


@mcp.tool()
def mark_lesson_effective(
    episode_id: str,
    led_to_success: bool,
    effectiveness_score: float = 0.5,
) -> dict[str, Any]:
    """
    Track if applying a lesson from an episode led to success.

    Args:
        episode_id: ID of the episode whose lesson was applied
        led_to_success: Whether applying the lesson led to success
        effectiveness_score: How effective the lesson was (0.0-1.0)

    Returns:
        Confirmation and updated stats
    """
    try:
        with _get_db() as conn:
            # Check episode exists
            cursor = conn.execute(
                "SELECT task FROM episodes WHERE episode_id = ?",
                (episode_id,),
            )
            row = cursor.fetchone()
            if not row:
                return {"error": f"Episode not found: {episode_id}"}

            task = row[0]
            score = max(0.0, min(1.0, effectiveness_score))

            conn.execute(
                """
                UPDATE episodes SET
                    led_to_success = ?,
                    effectiveness_score = ?
                WHERE episode_id = ?
                """,
                (1 if led_to_success else 0, score, episode_id),
            )

            conn.execute(
                """
                INSERT INTO lesson_effectiveness (episode_id, led_to_success, effectiveness_score)
                VALUES (?, ?, ?)
                """,
                (episode_id, 1 if led_to_success else 0, score),
            )
            conn.commit()

            return {
                "episode_id": episode_id,
                "led_to_success": led_to_success,
                "effectiveness_score": score,
                "task": task[:80],
                "updated": True,
            }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def export_lessons(
    feedback_type: str | None = None,
    min_occurrences: int = 1,
    min_success_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Export lessons for learner skill integration.

    Args:
        feedback_type: Filter by feedback type
        min_occurrences: Minimum occurrences to include
        min_success_rate: Minimum success rate to include

    Returns:
        Exportable lessons in learner skill format
    """
    try:
        with _get_db() as conn:
            sql = """
                SELECT reflection, feedback_type, task,
                       COUNT(*) as occurrences,
                       SUM(CASE WHEN led_to_success = 1 THEN 1.0 ELSE 0.0 END) /
                           NULLIF(COUNT(CASE WHEN led_to_success IS NOT NULL THEN 1 END), 0) as success_rate
                FROM episodes
                WHERE reflection IS NOT NULL
            """
            params: list[Any] = []

            if feedback_type:
                sql += " AND feedback_type = ?"
                params.append(feedback_type)

            sql += " GROUP BY reflection HAVING COUNT(*) >= ?"
            params.append(min_occurrences)

            sql += " ORDER BY occurrences DESC LIMIT 50"

            cursor = conn.execute(sql, params)

            lessons = []
            for row in cursor.fetchall():
                refl_json, fb_type, example_task, occurrences, success_rate = row
                refl = _json_loads(refl_json)

                if not refl or not refl.get("general_lesson"):
                    continue

                sr = success_rate or 0
                if sr < min_success_rate:
                    continue

                lessons.append(
                    {
                        "pattern": f"{fb_type}: {refl['general_lesson'][:50]}",
                        "lesson": refl["general_lesson"],
                        "feedback_type": fb_type,
                        "occurrences": occurrences,
                        "success_rate": round(sr, 3),
                        "examples": [example_task[:80]],
                        "confidence": min(1.0, occurrences * sr / 5),
                    }
                )

            return {
                "lessons": lessons[:20],
                "count": len(lessons[:20]),
                "filters": {
                    "feedback_type": feedback_type,
                    "min_occurrences": min_occurrences,
                    "min_success_rate": min_success_rate,
                },
                "instruction": "Use these lessons in the learner skill to add to project memory.",
            }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def link_episode_to_lesson(
    episode_id: str,
    lesson_episode_id: str,
) -> dict[str, Any]:
    """
    Link an episode to the lesson it applied.

    Args:
        episode_id: ID of the new episode
        lesson_episode_id: ID of the episode whose lesson was applied

    Returns:
        Confirmation of linkage
    """
    try:
        with _get_db() as conn:
            # Check both episodes exist
            cursor = conn.execute(
                "SELECT episode_id FROM episodes WHERE episode_id IN (?, ?)",
                (episode_id, lesson_episode_id),
            )
            found = [r[0] for r in cursor.fetchall()]

            if episode_id not in found:
                return {"error": f"Episode not found: {episode_id}"}
            if lesson_episode_id not in found:
                return {"error": f"Lesson episode not found: {lesson_episode_id}"}

            # Update episode
            conn.execute(
                "UPDATE episodes SET lesson_applied_from = ? WHERE episode_id = ?",
                (lesson_episode_id, episode_id),
            )

            # Add link
            conn.execute(
                """
                INSERT INTO episode_links (episode_id, lesson_episode_id)
                VALUES (?, ?)
                """,
                (episode_id, lesson_episode_id),
            )
            conn.commit()

            # Get lesson text
            cursor = conn.execute(
                "SELECT reflection FROM episodes WHERE episode_id = ?",
                (lesson_episode_id,),
            )
            row = cursor.fetchone()
            refl = _json_loads(row[0]) if row else None
            lesson_text = refl.get("general_lesson") if refl else None

            return {
                "episode_id": episode_id,
                "linked_to": lesson_episode_id,
                "lesson_applied": lesson_text,
                "instruction": "After completing this task, call mark_lesson_effective to track if the lesson helped.",
            }

    except Exception as e:
        return {"error": str(e)}


def main():
    """Entry point for the reflection MCP server."""
    logger.info(f"Starting Reflection MCP server, database: {DB_PATH}")
    mcp.run()


if __name__ == "__main__":
    main()
