"""Basic smoke tests for reflection MCP server."""

from reflection_mcp.server import (
    FeedbackType,
    OutcomeType,
    Reflection,
    Episode,
    LessonPattern,
    _keyword_similarity,
    _episode_to_dict,
    _dict_to_episode,
    _lesson_to_dict,
    _dict_to_lesson,
)


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_type_values(self) -> None:
        """All expected feedback types exist."""
        assert FeedbackType.TEST_FAILURE.value == "test_failure"
        assert FeedbackType.LINT_ERROR.value == "lint_error"
        assert FeedbackType.BUILD_ERROR.value == "build_error"
        assert FeedbackType.REVIEW_COMMENT.value == "review_comment"
        assert FeedbackType.RUNTIME_ERROR.value == "runtime_error"
        assert FeedbackType.SECURITY_ISSUE.value == "security_issue"
        assert FeedbackType.PERFORMANCE_ISSUE.value == "performance_issue"
        assert FeedbackType.TYPE_ERROR.value == "type_error"


class TestOutcomeType:
    """Tests for OutcomeType enum."""

    def test_outcome_type_values(self) -> None:
        """All expected outcome types exist."""
        assert OutcomeType.SUCCESS.value == "success"
        assert OutcomeType.PARTIAL.value == "partial"
        assert OutcomeType.FAILURE.value == "failure"


class TestReflection:
    """Tests for Reflection dataclass."""

    def test_reflection_creation(self) -> None:
        """Reflection can be created with required fields."""
        reflection = Reflection(
            id="test-id",
            what_went_wrong="Test failed",
            root_cause="Missing assertion",
            what_to_try_next="Add the assertion",
            general_lesson="Always verify assertions",
            confidence=0.8,
        )
        assert reflection.id == "test-id"
        assert reflection.what_went_wrong == "Test failed"
        assert reflection.confidence == 0.8


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_creation(self) -> None:
        """Episode can be created with required fields."""
        episode = Episode(
            id="ep-001",
            task="Implement login",
            approach="Used JWT tokens",
            outcome=OutcomeType.FAILURE,
            feedback="Token validation failed",
            feedback_type=FeedbackType.TEST_FAILURE,
            reflection=None,
            code_context="def login(): pass",
            file_path="auth.py",
            attempt_number=1,
            duration_seconds=30.0,
        )
        assert episode.id == "ep-001"
        assert episode.task == "Implement login"
        assert episode.outcome == OutcomeType.FAILURE
        assert episode.attempt_number == 1


class TestLessonPattern:
    """Tests for LessonPattern dataclass."""

    def test_lesson_pattern_creation(self) -> None:
        """LessonPattern can be created with required fields."""
        pattern = LessonPattern(
            id="lp-001",
            feedback_type=FeedbackType.TEST_FAILURE,
            pattern="test_failure:Always check return values",
            lesson="Always check return values",
            occurrences=5,
            success_rate=0.8,
            example_tasks=["task1", "task2"],
        )
        assert pattern.id == "lp-001"
        assert pattern.occurrences == 5
        assert pattern.success_rate == 0.8


class TestKeywordSimilarity:
    """Tests for keyword similarity function."""

    def test_identical_strings(self) -> None:
        """Identical strings have similarity of 1.0."""
        assert _keyword_similarity("hello world", "hello world") == 1.0

    def test_completely_different_strings(self) -> None:
        """Completely different strings have similarity of 0.0."""
        # Using words that won't be filtered as stopwords
        assert _keyword_similarity("apple banana", "cherry dragon") == 0.0

    def test_partial_overlap(self) -> None:
        """Strings with partial overlap have intermediate similarity."""
        sim = _keyword_similarity("python testing code", "python debugging code")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self) -> None:
        """Empty strings return 0.0 similarity."""
        assert _keyword_similarity("", "") == 0.0
        assert _keyword_similarity("hello", "") == 0.0


class TestEpisodeSerialization:
    """Tests for episode serialization."""

    def test_episode_roundtrip(self) -> None:
        """Episode can be serialized and deserialized."""
        reflection = Reflection(
            id="ref-001",
            what_went_wrong="Failed",
            root_cause="Bug",
            what_to_try_next="Fix it",
            general_lesson="Test first",
            confidence=0.9,
        )
        original = Episode(
            id="ep-001",
            task="Test task",
            approach="Test approach",
            outcome=OutcomeType.SUCCESS,
            feedback="Test feedback",
            feedback_type=FeedbackType.TEST_FAILURE,
            reflection=reflection,
            code_context="code",
            file_path="test.py",
            attempt_number=1,
            duration_seconds=10.0,
            tags=["test"],
        )

        serialized = _episode_to_dict(original)
        deserialized = _dict_to_episode(serialized)

        assert deserialized.id == original.id
        assert deserialized.task == original.task
        assert deserialized.outcome == original.outcome
        assert deserialized.reflection is not None
        assert deserialized.reflection.general_lesson == "Test first"


class TestLessonSerialization:
    """Tests for lesson serialization."""

    def test_lesson_roundtrip(self) -> None:
        """LessonPattern can be serialized and deserialized."""
        original = LessonPattern(
            id="lp-001",
            feedback_type=FeedbackType.LINT_ERROR,
            pattern="lint:Fix imports",
            lesson="Fix imports",
            occurrences=3,
            success_rate=0.7,
            example_tasks=["task1"],
        )

        serialized = _lesson_to_dict(original)
        deserialized = _dict_to_lesson(serialized)

        assert deserialized.id == original.id
        assert deserialized.feedback_type == original.feedback_type
        assert deserialized.success_rate == original.success_rate
