"""Unit tests for router operations."""

from unittest.mock import MagicMock, patch

from ragchain.router import _is_simple_query, intent_router
from ragchain.schema import Intent


def test_is_simple_query():
    """Test simple query detection."""
    assert _is_simple_query("What is Python?") is True
    assert _is_simple_query("Explain recursion") is True
    assert _is_simple_query("Compare Python and Java") is False  # Too long
    assert _is_simple_query("List programming languages") is False  # Not simple pattern


@patch("ragchain.router.config")
def test_intent_router_fast_path(mock_config):
    """Test intent router fast path for simple queries."""
    mock_config.enable_intent_routing = False

    state = {"query": "What is Python?", "intent": Intent.CONCEPT}
    result = intent_router(state)

    assert result["intent"] == Intent.CONCEPT
    assert "original_query" in result


@patch("ragchain.router.config")
@patch("ragchain.router.OllamaLLM")
def test_intent_router_with_llm(mock_llm_class, mock_config):
    """Test intent router with LLM classification."""
    mock_config.enable_intent_routing = True
    mock_config.ollama_model = "test-model"
    mock_config.ollama_base_url = "http://test"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "FACT"
    mock_llm_class.return_value = mock_llm

    state = {"query": "What are the top 10 languages?", "intent": Intent.CONCEPT}
    result = intent_router(state)

    assert result["intent"] == Intent.FACT
    assert result["original_query"] == "What are the top 10 languages?"
    mock_llm.invoke.assert_called_once()
