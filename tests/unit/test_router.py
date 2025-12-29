"""Unit tests for router operations."""

from unittest.mock import MagicMock, patch

from ragchain.core.enums import Intent
from ragchain.core.router import intent_router


@patch("ragchain.core.router.config")
def test_intent_router_fast_path(mock_config):
    """Test intent router fast path for simple queries."""
    mock_config.enable_intent_routing = False

    state = {"query": "What is Python?", "intent": Intent.CONCEPT}
    result = intent_router(state)

    assert result["intent"] == Intent.CONCEPT
    assert "original_query" in result


@patch("ragchain.core.router.config")
@patch("ragchain.core.router.OllamaLLM")
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
