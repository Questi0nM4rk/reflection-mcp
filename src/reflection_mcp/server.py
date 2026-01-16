"""
Self-Reflection MCP Server

Implements Reflexion pattern with persistent episodic memory.
Based on NeurIPS 2023 research showing 67% → 88% improvement on HumanEval (+21%).

Features:
- Persistent storage (JSON files in ~/.codeagent/data/reflection-episodes/)
- Lesson effectiveness tracking (did applying a lesson lead to success?)
- Cross-session learning aggregation
- Export capability for learner skill integration

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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent storage directory
DATA_DIR = Path(os.environ.get("CODEAGENT_HOME", Path.home() / ".codeagent")) / "data" / "reflection-episodes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastMCP server
mcp = FastMCP(
    "reflection",
    instructions="Self-reflection and episodic memory for learning from failures. "
    "Use this when code fails tests or reviews to improve on subsequent attempts.",
)


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


@dataclass
class Reflection:
    """A reflection on a failure."""

    id: str
    what_went_wrong: str
    root_cause: str
    what_to_try_next: str
    general_lesson: str
    confidence: float  # 0-1 confidence in the reflection
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Episode:
    """An episodic memory of a task attempt."""

    id: str
    task: str
    approach: str
    outcome: OutcomeType
    feedback: str
    feedback_type: FeedbackType
    reflection: Reflection | None
    code_context: str  # Relevant code snippet
    file_path: str | None
    attempt_number: int
    duration_seconds: float | None
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # New fields for effectiveness tracking
    lesson_applied_from: str | None = None  # Episode ID if lesson was applied
    led_to_success: bool | None = None  # Did this episode lead to success?
    effectiveness_score: float = 0.0  # How effective was the lesson (0-1)


@dataclass
class LessonPattern:
    """An aggregated lesson pattern from multiple episodes."""

    id: str
    feedback_type: FeedbackType
    pattern: str  # Common error pattern
    lesson: str  # What to do
    occurrences: int
    success_rate: float  # How often applying this lesson leads to success
    example_tasks: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# In-memory cache (loaded from disk on demand)
_episodes_cache: list[Episode] = []
_task_attempts: dict[str, int] = {}
_lessons_cache: list[LessonPattern] = []
_episodes_loaded: bool = False
_lessons_loaded: bool = False


def _get_episodes_file() -> Path:
    """Get path to episodes JSON file."""
    return DATA_DIR / "episodes.json"


def _get_lessons_file() -> Path:
    """Get path to lessons JSON file."""
    return DATA_DIR / "lessons.json"


def _episode_to_dict(episode: Episode) -> dict:
    """Convert episode to serializable dict."""
    return {
        "id": episode.id,
        "task": episode.task,
        "approach": episode.approach,
        "outcome": episode.outcome.value,
        "feedback": episode.feedback,
        "feedback_type": episode.feedback_type.value,
        "reflection": {
            "id": episode.reflection.id,
            "what_went_wrong": episode.reflection.what_went_wrong,
            "root_cause": episode.reflection.root_cause,
            "what_to_try_next": episode.reflection.what_to_try_next,
            "general_lesson": episode.reflection.general_lesson,
            "confidence": episode.reflection.confidence,
            "created_at": episode.reflection.created_at,
        } if episode.reflection else None,
        "code_context": episode.code_context,
        "file_path": episode.file_path,
        "attempt_number": episode.attempt_number,
        "duration_seconds": episode.duration_seconds,
        "tags": episode.tags,
        "created_at": episode.created_at,
        "lesson_applied_from": episode.lesson_applied_from,
        "led_to_success": episode.led_to_success,
        "effectiveness_score": episode.effectiveness_score,
    }


def _dict_to_episode(data: dict) -> Episode:
    """Convert dict to Episode."""
    reflection = None
    if data.get("reflection"):
        r = data["reflection"]
        reflection = Reflection(
            id=r.get("id", ""),
            what_went_wrong=r.get("what_went_wrong", ""),
            root_cause=r.get("root_cause", ""),
            what_to_try_next=r.get("what_to_try_next", ""),
            general_lesson=r.get("general_lesson", ""),
            confidence=r.get("confidence", 0.5),
            created_at=r.get("created_at", ""),
        )

    return Episode(
        id=data["id"],
        task=data["task"],
        approach=data["approach"],
        outcome=OutcomeType(data.get("outcome", "failure")),
        feedback=data["feedback"],
        feedback_type=FeedbackType(data.get("feedback_type", "test_failure")),
        reflection=reflection,
        code_context=data.get("code_context", ""),
        file_path=data.get("file_path"),
        attempt_number=data.get("attempt_number", 1),
        duration_seconds=data.get("duration_seconds"),
        tags=data.get("tags", []),
        created_at=data.get("created_at", ""),
        lesson_applied_from=data.get("lesson_applied_from"),
        led_to_success=data.get("led_to_success"),
        effectiveness_score=data.get("effectiveness_score", 0.0),
    )


def _load_episodes() -> list[Episode]:
    """Load episodes from disk."""
    global _episodes_cache, _episodes_loaded, _task_attempts

    if _episodes_loaded:
        return _episodes_cache

    episodes_file = _get_episodes_file()
    if episodes_file.exists():
        try:
            with open(episodes_file, "r") as f:
                data = json.load(f)
            _episodes_cache = [_dict_to_episode(e) for e in data.get("episodes", [])]
            _task_attempts = data.get("task_attempts", {})
            logger.info(f"Loaded {len(_episodes_cache)} episodes from disk")
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")
            _episodes_cache = []
    else:
        _episodes_cache = []

    _episodes_loaded = True
    return _episodes_cache


def _save_episodes():
    """Save episodes to disk."""
    episodes_file = _get_episodes_file()
    try:
        data = {
            "episodes": [_episode_to_dict(e) for e in _episodes_cache],
            "task_attempts": _task_attempts,
            "updated_at": datetime.now().isoformat(),
        }
        with open(episodes_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save episodes: {e}")


def _lesson_to_dict(lesson: LessonPattern) -> dict:
    """Convert lesson to serializable dict."""
    return {
        "id": lesson.id,
        "feedback_type": lesson.feedback_type.value,
        "pattern": lesson.pattern,
        "lesson": lesson.lesson,
        "occurrences": lesson.occurrences,
        "success_rate": lesson.success_rate,
        "example_tasks": lesson.example_tasks,
        "created_at": lesson.created_at,
        "updated_at": lesson.updated_at,
    }


def _dict_to_lesson(data: dict) -> LessonPattern:
    """Convert dict to LessonPattern."""
    return LessonPattern(
        id=data["id"],
        feedback_type=FeedbackType(data.get("feedback_type", "test_failure")),
        pattern=data["pattern"],
        lesson=data["lesson"],
        occurrences=data.get("occurrences", 1),
        success_rate=data.get("success_rate", 0.0),
        example_tasks=data.get("example_tasks", []),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def _load_lessons() -> list[LessonPattern]:
    """Load aggregated lessons from disk."""
    global _lessons_cache, _lessons_loaded

    if _lessons_loaded:
        return _lessons_cache

    lessons_file = _get_lessons_file()
    if lessons_file.exists():
        try:
            with open(lessons_file, "r") as f:
                data = json.load(f)
            _lessons_cache = [_dict_to_lesson(l) for l in data.get("lessons", [])]
        except Exception as e:
            logger.error(f"Failed to load lessons: {e}")
            _lessons_cache = []
    else:
        _lessons_cache = []

    _lessons_loaded = True
    return _lessons_cache


def _save_lessons():
    """Save lessons to disk."""
    lessons_file = _get_lessons_file()
    try:
        data = {
            "lessons": [_lesson_to_dict(l) for l in _lessons_cache],
            "updated_at": datetime.now().isoformat(),
        }
        with open(lessons_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save lessons: {e}")


def _get_attempt_number(task: str) -> int:
    """Get and increment attempt number for a task."""
    _load_episodes()  # Ensure loaded
    key = task[:100]  # Normalize task key
    count = _task_attempts.get(key, 0) + 1
    _task_attempts[key] = count
    return count


def _update_lesson_patterns():
    """Aggregate episodes into lesson patterns."""
    episodes = _load_episodes()
    lessons = _load_lessons()

    # Group by feedback type and lesson
    pattern_groups: dict[str, list[Episode]] = {}

    for ep in episodes:
        if ep.reflection and ep.reflection.general_lesson:
            key = f"{ep.feedback_type.value}:{ep.reflection.general_lesson[:50]}"
            if key not in pattern_groups:
                pattern_groups[key] = []
            pattern_groups[key].append(ep)

    # Update or create lesson patterns
    for key, group in pattern_groups.items():
        fb_type, _ = key.split(":", 1)
        lesson_text = group[0].reflection.general_lesson if group[0].reflection else ""

        # Find existing lesson or create new
        existing = next((l for l in lessons if l.pattern == key), None)

        # Calculate success rate
        successes = sum(1 for ep in group if ep.led_to_success)
        total_tracked = sum(1 for ep in group if ep.led_to_success is not None)
        success_rate = successes / total_tracked if total_tracked > 0 else 0.0

        if existing:
            existing.occurrences = len(group)
            existing.success_rate = success_rate
            existing.example_tasks = list(set(ep.task[:80] for ep in group[:5]))
            existing.updated_at = datetime.now().isoformat()
        else:
            new_lesson = LessonPattern(
                id=str(uuid4())[:8],
                feedback_type=FeedbackType(fb_type),
                pattern=key,
                lesson=lesson_text,
                occurrences=len(group),
                success_rate=success_rate,
                example_tasks=list(set(ep.task[:80] for ep in group[:5])),
            )
            lessons.append(new_lesson)

    global _lessons_cache
    _lessons_cache = lessons
    _save_lessons()


def _keyword_similarity(text1: str, text2: str) -> float:
    """Simple keyword-based similarity score."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Remove common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'that',
                 'this', 'these', 'those', 'it', 'its'}

    words1 = words1 - stopwords
    words2 = words2 - stopwords

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


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

    # Parse feedback type
    try:
        fb_type = FeedbackType(feedback_type)
    except ValueError:
        fb_type = FeedbackType.TEST_FAILURE

    # Generate reflection structure
    # In a production system, this would use an LLM to generate better reflections
    # Here we provide structured prompts for the calling agent to fill in

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
            "test_failure": "Focus on what assertion failed and why the code doesn't satisfy the test condition.",
            "lint_error": "Identify the specific rule violation and the correct pattern to use.",
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
    episode_id = str(uuid4())[:8]

    # Parse outcome
    try:
        outcome_type = OutcomeType(outcome)
    except ValueError:
        outcome_type = OutcomeType.FAILURE

    # Parse feedback type
    try:
        fb_type = FeedbackType(feedback_type)
    except ValueError:
        fb_type = FeedbackType.TEST_FAILURE

    # Create reflection object
    refl = Reflection(
        id=str(uuid4())[:8],
        what_went_wrong=reflection.get("what_went_wrong", ""),
        root_cause=reflection.get("root_cause", ""),
        what_to_try_next=reflection.get("what_to_try_next", ""),
        general_lesson=reflection.get("general_lesson", ""),
        confidence=float(reflection.get("confidence", 0.5)),
    )

    # Create episode
    episode = Episode(
        id=episode_id,
        task=task,
        approach=approach,
        outcome=outcome_type,
        feedback=feedback[:2000],  # Truncate long feedback
        feedback_type=fb_type,
        reflection=refl,
        code_context=code_context[:3000],  # Truncate long code
        file_path=file_path,
        attempt_number=_get_attempt_number(task),
        duration_seconds=duration_seconds,
        tags=tags or [],
    )

    episodes = _load_episodes()
    episodes.append(episode)
    _save_episodes()

    # Update lesson patterns periodically
    if len(episodes) % 5 == 0:  # Every 5 episodes
        _update_lesson_patterns()

    return {
        "episode_id": episode_id,
        "task": task[:100],
        "attempt_number": episode.attempt_number,
        "outcome": outcome_type.value,
        "stored": True,
        "total_episodes": len(episodes),
        "lesson": refl.general_lesson,
        "storage": str(_get_episodes_file()),
    }


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
    top_k = min(max(top_k, 1), 20)

    # Filter episodes
    episodes = _load_episodes()
    candidates = episodes.copy()

    if feedback_type:
        try:
            fb_type = FeedbackType(feedback_type)
            candidates = [e for e in candidates if e.feedback_type == fb_type]
        except ValueError:
            pass

    if not include_successes:
        candidates = [e for e in candidates if e.outcome != OutcomeType.SUCCESS]

    # Score by similarity
    scored = []
    for episode in candidates:
        task_sim = _keyword_similarity(task, episode.task)
        error_sim = _keyword_similarity(error_pattern, episode.feedback) if error_pattern else 0

        # Combine scores (task similarity more important)
        score = task_sim * 0.7 + error_sim * 0.3

        # Boost recent episodes slightly
        scored.append((score, episode))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-k
    results = []
    for score, episode in scored[:top_k]:
        results.append({
            "episode_id": episode.id,
            "task": episode.task[:200],
            "approach": episode.approach[:200],
            "outcome": episode.outcome.value,
            "feedback_type": episode.feedback_type.value,
            "attempt_number": episode.attempt_number,
            "similarity_score": round(score, 3),
            "reflection": {
                "what_went_wrong": episode.reflection.what_went_wrong if episode.reflection else "",
                "root_cause": episode.reflection.root_cause if episode.reflection else "",
                "what_to_try_next": episode.reflection.what_to_try_next if episode.reflection else "",
                "general_lesson": episode.reflection.general_lesson if episode.reflection else "",
            } if episode.reflection else None,
            "created_at": episode.created_at,
        })

    return {
        "query_task": task[:100],
        "query_error": error_pattern[:100] if error_pattern else None,
        "episodes": results,
        "count": len(results),
        "instruction": "Apply lessons from these past episodes to your current approach. "
        "Focus on the general_lesson and what_to_try_next fields.",
    }


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
    # Extract lessons from episodes
    past_lessons = []
    if similar_episodes:
        for ep in similar_episodes:
            if ep.get("reflection") and ep["reflection"].get("general_lesson"):
                past_lessons.append({
                    "task": ep.get("task", "")[:80],
                    "lesson": ep["reflection"]["general_lesson"],
                    "what_worked": ep["reflection"].get("what_to_try_next", ""),
                })

    return {
        "original_output_summary": original_output[:300],
        "current_reflection": {
            "root_cause": reflection.get("root_cause", "Unknown"),
            "what_to_try_next": reflection.get("what_to_try_next", ""),
        },
        "past_lessons": past_lessons[:5],  # Top 5 relevant lessons
        "improvement_strategy": {
            "1_address_root_cause": f"Fix: {reflection.get('root_cause', 'the identified issue')}",
            "2_apply_lesson": reflection.get("what_to_try_next", "Apply the learned approach"),
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
    limit = min(max(limit, 1), 50)

    # Filter episodes
    episodes = _load_episodes()
    if task:
        filtered = [e for e in episodes if task.lower() in e.task.lower()]
    else:
        filtered = episodes.copy()

    # Sort by created_at descending (most recent first)
    filtered.sort(key=lambda e: e.created_at, reverse=True)

    # Format results
    history = []
    for episode in filtered[:limit]:
        history.append({
            "episode_id": episode.id,
            "task": episode.task[:100],
            "outcome": episode.outcome.value,
            "feedback_type": episode.feedback_type.value,
            "attempt_number": episode.attempt_number,
            "lesson": episode.reflection.general_lesson if episode.reflection else None,
            "created_at": episode.created_at,
        })

    # Calculate stats
    outcomes = [e.outcome for e in filtered]
    stats = {
        "total": len(filtered),
        "successes": sum(1 for o in outcomes if o == OutcomeType.SUCCESS),
        "partial": sum(1 for o in outcomes if o == OutcomeType.PARTIAL),
        "failures": sum(1 for o in outcomes if o == OutcomeType.FAILURE),
    }

    return {
        "history": history,
        "stats": stats,
        "filter": task,
    }


@mcp.tool()
def get_common_lessons() -> dict[str, Any]:
    """
    Get aggregated lessons from all episodes, grouped by feedback type.

    Returns:
        Common lessons learned, organized by feedback type
    """
    episodes = _load_episodes()
    lessons_by_type: dict[str, list[str]] = {}

    for episode in episodes:
        if episode.reflection and episode.reflection.general_lesson:
            fb_type = episode.feedback_type.value
            if fb_type not in lessons_by_type:
                lessons_by_type[fb_type] = []
            lessons_by_type[fb_type].append(episode.reflection.general_lesson)

    # Deduplicate and format
    formatted = {}
    for fb_type, lessons in lessons_by_type.items():
        # Simple deduplication by keeping unique lessons
        unique = list(set(lessons))
        formatted[fb_type] = unique[:10]  # Top 10 per type

    # Also include aggregated lesson patterns with effectiveness
    _update_lesson_patterns()
    lessons = _load_lessons()
    effective_lessons = [
        {
            "lesson": l.lesson,
            "feedback_type": l.feedback_type.value,
            "occurrences": l.occurrences,
            "success_rate": round(l.success_rate, 3),
        }
        for l in sorted(lessons, key=lambda x: x.success_rate * x.occurrences, reverse=True)[:10]
    ]

    return {
        "lessons_by_feedback_type": formatted,
        "most_effective_lessons": effective_lessons,
        "total_episodes": len(episodes),
        "storage": str(_get_lessons_file()),
        "instruction": "Use these accumulated lessons to avoid repeating past mistakes. "
        "Focus on most_effective_lessons which have high success rates.",
    }


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
    global _episodes_cache, _task_attempts, _episodes_loaded

    episodes = _load_episodes()
    initial_count = len(episodes)

    if older_than_days is None and feedback_type is None:
        # Clear all
        _episodes_cache = []
        _task_attempts.clear()
        _save_episodes()
        return {"cleared": initial_count, "remaining": 0}

    # Filter what to keep
    keep = []
    now = datetime.now()

    for episode in episodes:
        should_remove = True

        if older_than_days is not None:
            episode_time = datetime.fromisoformat(episode.created_at)
            age_days = (now - episode_time).days
            if age_days < older_than_days:
                should_remove = False

        if feedback_type is not None:
            try:
                fb_type = FeedbackType(feedback_type)
                if episode.feedback_type != fb_type:
                    should_remove = False
            except ValueError:
                pass

        if not should_remove:
            keep.append(episode)

    cleared = initial_count - len(keep)
    _episodes_cache = keep
    _save_episodes()

    return {"cleared": cleared, "remaining": len(_episodes_cache)}


@mcp.tool()
def get_episode_stats() -> dict[str, Any]:
    """
    Get statistics about episodic memory.

    Returns:
        Statistics including counts, success rates, common failure types
    """
    episodes = _load_episodes()
    if not episodes:
        return {"message": "No episodes stored yet.", "total": 0, "storage": str(DATA_DIR)}

    # Count by outcome
    outcomes = {}
    for outcome in OutcomeType:
        outcomes[outcome.value] = sum(1 for e in episodes if e.outcome == outcome)

    # Count by feedback type
    feedback_types = {}
    for fb_type in FeedbackType:
        count = sum(1 for e in episodes if e.feedback_type == fb_type)
        if count > 0:
            feedback_types[fb_type.value] = count

    # Calculate success rate
    total = len(episodes)
    success_rate = outcomes.get("success", 0) / total if total > 0 else 0

    # Average attempts per task
    avg_attempts = sum(e.attempt_number for e in episodes) / total if total > 0 else 0

    # Most common failure types (for failures only)
    failure_types = {}
    for e in episodes:
        if e.outcome == OutcomeType.FAILURE:
            fb = e.feedback_type.value
            failure_types[fb] = failure_types.get(fb, 0) + 1

    # Lesson effectiveness stats
    lessons_applied = sum(1 for e in episodes if e.lesson_applied_from)
    lessons_effective = sum(1 for e in episodes if e.led_to_success)
    lesson_effectiveness = lessons_effective / lessons_applied if lessons_applied > 0 else 0

    return {
        "total_episodes": total,
        "by_outcome": outcomes,
        "by_feedback_type": feedback_types,
        "success_rate": round(success_rate, 3),
        "average_attempts": round(avg_attempts, 2),
        "most_common_failures": dict(sorted(failure_types.items(), key=lambda x: x[1], reverse=True)[:5]),
        "lesson_effectiveness": {
            "lessons_applied": lessons_applied,
            "led_to_success": lessons_effective,
            "effectiveness_rate": round(lesson_effectiveness, 3),
        },
        "storage": str(DATA_DIR),
    }


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
    episodes = _load_episodes()

    # Find the episode
    episode = next((e for e in episodes if e.id == episode_id), None)
    if not episode:
        return {"error": f"Episode not found: {episode_id}"}

    episode.led_to_success = led_to_success
    episode.effectiveness_score = max(0.0, min(1.0, effectiveness_score))
    _save_episodes()

    # Update lesson patterns
    _update_lesson_patterns()

    return {
        "episode_id": episode_id,
        "led_to_success": led_to_success,
        "effectiveness_score": episode.effectiveness_score,
        "task": episode.task[:80],
        "updated": True,
    }


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
    _update_lesson_patterns()
    lessons = _load_lessons()

    # Filter
    filtered = lessons
    if feedback_type:
        try:
            fb_type = FeedbackType(feedback_type)
            filtered = [l for l in filtered if l.feedback_type == fb_type]
        except ValueError:
            pass

    filtered = [l for l in filtered if l.occurrences >= min_occurrences]
    filtered = [l for l in filtered if l.success_rate >= min_success_rate]

    # Sort by effectiveness
    filtered.sort(key=lambda l: l.success_rate * l.occurrences, reverse=True)

    # Format for learner skill
    exportable = []
    for lesson in filtered[:20]:  # Top 20
        exportable.append({
            "pattern": f"{lesson.feedback_type.value}: {lesson.lesson}",
            "lesson": lesson.lesson,
            "feedback_type": lesson.feedback_type.value,
            "occurrences": lesson.occurrences,
            "success_rate": round(lesson.success_rate, 3),
            "examples": lesson.example_tasks[:3],
            "confidence": min(1.0, lesson.occurrences * lesson.success_rate / 5),  # Confidence based on data
        })

    return {
        "lessons": exportable,
        "count": len(exportable),
        "filters": {
            "feedback_type": feedback_type,
            "min_occurrences": min_occurrences,
            "min_success_rate": min_success_rate,
        },
        "instruction": "Use these lessons in the learner skill to add to project memory. "
        "High confidence lessons should be prioritized.",
    }


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
    episodes = _load_episodes()

    episode = next((e for e in episodes if e.id == episode_id), None)
    if not episode:
        return {"error": f"Episode not found: {episode_id}"}

    lesson_episode = next((e for e in episodes if e.id == lesson_episode_id), None)
    if not lesson_episode:
        return {"error": f"Lesson episode not found: {lesson_episode_id}"}

    episode.lesson_applied_from = lesson_episode_id
    _save_episodes()

    return {
        "episode_id": episode_id,
        "linked_to": lesson_episode_id,
        "lesson_applied": lesson_episode.reflection.general_lesson if lesson_episode.reflection else None,
        "instruction": "After completing this task, call mark_lesson_effective to track if the lesson helped.",
    }


def main():
    """Entry point for the reflection MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
