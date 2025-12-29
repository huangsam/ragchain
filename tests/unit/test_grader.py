"""Unit tests for document relevance grading."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ragchain.core.enums import GradeSignal
from ragchain.core.grader import grade_with_llm, should_accept_docs, should_skip_grading


class TestShouldSkipGrading:
    """Test should_skip_grading function."""

    @patch("ragchain.core.grader.config.enable_grading", False)
    def test_should_skip_when_disabled(self):
        """Test that grading is skipped when disabled."""
        assert should_skip_grading() is True

    @patch("ragchain.core.grader.config.enable_grading", True)
    def test_should_not_skip_when_enabled(self):
        """Test that grading is not skipped when enabled."""
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


class TestGradeWithLLM:
    """Test grade_with_llm function."""

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_yes_response(self, mock_llm_class):
        """Test grading returns YES for relevant response."""
        # Mock LLM to return YES
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "YES"
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Python is a programming language.")]
        result = grade_with_llm("What is Python?", docs)

        assert result == GradeSignal.YES
        mock_llm.invoke.assert_called_once()

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_no_response(self, mock_llm_class):
        """Test grading returns NO for irrelevant response."""
        # Mock LLM to return NO
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "NO"
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Java is a programming language.")]
        result = grade_with_llm("What is Python?", docs)

        assert result == GradeSignal.NO
        mock_llm.invoke.assert_called_once()

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_yes_with_extra_text(self, mock_llm_class):
        """Test grading extracts YES from response with extra text."""
        # Mock LLM to return YES with extra text
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "YES, these documents are relevant."
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Python programming language info.")]
        result = grade_with_llm("Python info", docs)

        assert result == GradeSignal.YES

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_no_with_extra_text(self, mock_llm_class):
        """Test grading extracts NO from response with extra text."""
        # Mock LLM to return NO with extra text
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "NO, completely unrelated."
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Unrelated content.")]
        result = grade_with_llm("Python programming", docs)

        assert result == GradeSignal.NO

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_handles_exception(self, mock_llm_class):
        """Test grading returns NO on LLM exception."""
        # Mock LLM to raise exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Some content.")]
        result = grade_with_llm("Test query", docs)

        assert result == GradeSignal.NO

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_empty_response(self, mock_llm_class):
        """Test grading handles empty LLM response."""
        # Mock LLM to return empty string
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""
        mock_llm_class.return_value = mock_llm

        docs = [Document(page_content="Content.")]
        result = grade_with_llm("Query", docs)

        assert result == GradeSignal.NO

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_multiple_docs(self, mock_llm_class):
        """Test grading with multiple documents."""
        # Mock LLM to return YES
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "YES"
        mock_llm_class.return_value = mock_llm

        docs = [
            Document(page_content="First document about Python."),
            Document(page_content="Second document about programming."),
            Document(page_content="Third document with more info."),
        ]
        result = grade_with_llm("Python programming", docs)

        assert result == GradeSignal.YES
        # Verify the prompt was formatted with all docs
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Doc 0:" in call_args
        assert "Doc 1:" in call_args
        assert "Doc 2:" in call_args

    @patch("ragchain.core.grader.OllamaLLM")
    def test_grade_long_doc_content_truncated(self, mock_llm_class):
        """Test that long document content is truncated in prompt."""
        # Mock LLM to return YES
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "YES"
        mock_llm_class.return_value = mock_llm

        # Create a document with content longer than 200 chars
        long_content = "A" * 300
        docs = [Document(page_content=long_content)]
        result = grade_with_llm("Test query", docs)

        assert result == GradeSignal.YES
        # Verify content was truncated to 200 chars
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Doc 0: " in call_args
