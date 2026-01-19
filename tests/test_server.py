"""Basic smoke tests for reflection MCP server."""

from reflection_mcp.server import (
    FeedbackType,
    OutcomeType,
    _keyword_similarity,
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
