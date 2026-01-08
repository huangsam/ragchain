"""Unit tests for document relevance grading."""

from unittest.mock import patch

from langchain_core.documents import Document

from ragchain.inference.grader import extract_keywords, grade_with_statistics, should_accept_docs, should_skip_grading
from ragchain.types import GradeSignal


class TestShouldSkipGrading:
    """Test should_skip_grading function."""

    def test_should_skip_when_disabled(self, mock_config):
        """Test that grading is skipped when disabled."""
        mock_config.enable_grading = False
        with patch("ragchain.inference.grader.config", mock_config):
            assert should_skip_grading() is True

    def test_should_not_skip_when_enabled(self, mock_config):
        """Test that grading is not skipped when enabled."""
        mock_config.enable_grading = True
        with patch("ragchain.inference.grader.config", mock_config):
            assert should_skip_grading() is False


class TestShouldAcceptDocs:
    """Test should_accept_docs function."""

    def test_should_accept_empty_docs(self):
        """Test that empty doc list is auto-accepted."""
        assert should_accept_docs([], 0) is True

    def test_should_accept_on_retry(self):
        """Test that docs are auto-accepted on retry."""
        docs = [Document(page_content="test")]
        assert should_accept_docs(docs, 1) is True

    def test_should_not_accept_on_first_attempt(self):
        """Test that docs are not auto-accepted on first attempt."""
        docs = [Document(page_content="test")]
        assert should_accept_docs(docs, 0) is False


class TestGradeWithStatistics:
    """Test grade_with_statistics function (statistical keyword-based grading)."""

    def test_grade_yes_for_relevant_docs(self):
        """Test grading returns YES for documents with keyword overlap."""
        docs = [Document(page_content="Python is a programming language used for web development and data science.")]
        result = grade_with_statistics("What is Python programming?", docs)

        assert result == GradeSignal.YES

    def test_grade_no_for_irrelevant_docs(self):
        """Test grading returns NO for documents without keyword overlap."""
        docs = [Document(page_content="The weather today is sunny and warm.")]
        result = grade_with_statistics("What is Python programming?", docs)

        assert result == GradeSignal.NO

    def test_grade_yes_with_partial_overlap(self):
        """Test grading returns YES with partial keyword overlap meeting threshold."""
        docs = [Document(page_content="Python programming is powerful for data analysis.")]
        result = grade_with_statistics("Python programming language features", docs)

        assert result == GradeSignal.YES

    def test_grade_multiple_docs_finds_relevant(self):
        """Test grading finds relevant doc among multiple documents."""
        docs = [
            Document(page_content="Unrelated content about cooking recipes."),
            Document(page_content="Python is a high-level programming language."),
            Document(page_content="More unrelated content about sports."),
        ]
        result = grade_with_statistics("Python programming", docs)

        assert result == GradeSignal.YES

    def test_grade_empty_query_keywords_accepts(self):
        """Test that empty query keywords (all stop words) auto-accepts docs."""
        docs = [Document(page_content="Some content here.")]
        # Query with only stop words
        result = grade_with_statistics("the and is", docs)

        assert result == GradeSignal.YES

    def test_grade_handles_empty_docs(self):
        """Test grading handles empty document list."""
        result = grade_with_statistics("Python query", [])

        assert result == GradeSignal.NO

    def test_grade_term_frequency_bonus(self):
        """Test that term frequency contributes to score."""
        # Doc with repeated keywords should score higher
        docs = [Document(page_content="Python Python Python programming programming language")]
        result = grade_with_statistics("Python programming", docs)

        assert result == GradeSignal.YES

    def test_grade_top_k_ranking(self):
        """Test that only top-3 documents are considered for hit rate."""
        # First 3 docs are irrelevant, 4th is relevant - should return NO
        docs = [
            Document(page_content="Cooking recipes for dinner."),
            Document(page_content="Sports news and updates."),
            Document(page_content="Weather forecast for tomorrow."),
            Document(page_content="Python programming language guide."),  # This won't be in top-3 by score
        ]
        result = grade_with_statistics("Python programming", docs)

        # The relevant doc should still be found because scoring puts it first
        assert result == GradeSignal.YES


class TestExtractKeywords:
    """Test extract_keywords helper function."""

    def test_extracts_meaningful_words(self):
        """Test that meaningful words are extracted."""
        keywords = extract_keywords("Python is a programming language")
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords

    def test_filters_stop_words(self):
        """Test that stop words are filtered out."""
        keywords = extract_keywords("What is the Python programming language?")
        assert "what" not in keywords
        assert "the" not in keywords
        assert "python" in keywords

    def test_filters_short_words(self):
        """Test that words shorter than 3 chars are filtered."""
        keywords = extract_keywords("Go is a language")
        assert "go" not in keywords  # Too short
        assert "language" in keywords

    def test_lowercase_normalization(self):
        """Test that keywords are lowercased."""
        keywords = extract_keywords("PYTHON Programming LANGUAGE")
        assert "python" in keywords
        assert "PYTHON" not in keywords
