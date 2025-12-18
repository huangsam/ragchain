"""Unit tests for graph operations."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ragchain.graph import _is_simple_query, intent_router


def test_is_simple_query():
    """Test simple query detection."""
    assert _is_simple_query("What is Python?") is True
    assert _is_simple_query("Explain recursion") is True
    assert _is_simple_query("Compare Python and Java") is False  # Too long
    assert _is_simple_query("List programming languages") is False  # Not simple pattern


@patch("ragchain.graph.config")
def test_intent_router_fast_path(mock_config):
    """Test intent router fast path for simple queries."""
    mock_config.enable_intent_routing = False

    state = {"query": "What is Python?", "intent": "CONCEPT"}
    result = intent_router(state)

    assert result["intent"] == "CONCEPT"
    assert "original_query" in result


@patch("ragchain.graph.config")
@patch("ragchain.graph.OllamaLLM")
def test_intent_router_with_llm(mock_llm_class, mock_config):
    """Test intent router with LLM classification."""
    mock_config.enable_intent_routing = True
    mock_config.ollama_model = "test-model"
    mock_config.ollama_base_url = "http://test"

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "FACT"
    mock_llm_class.return_value = mock_llm

    state = {"query": "What are the top 10 languages?", "intent": "CONCEPT"}
    result = intent_router(state)

    assert result["intent"] == "FACT"
    assert result["original_query"] == "What are the top 10 languages?"
    mock_llm.invoke.assert_called_once()


@patch("ragchain.graph.get_ensemble_retriever")
def test_adaptive_retriever(mock_get_retriever):
    """Test adaptive retriever with different intents."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [Document(page_content="Test doc", metadata={})]
    mock_get_retriever.return_value = mock_retriever

    from ragchain.graph import adaptive_retriever

    # Test FACT intent
    state = {"query": "What are top languages?", "intent": "FACT", "retrieved_docs": []}
    result = adaptive_retriever(state)

    assert len(result["retrieved_docs"]) == 1
    mock_get_retriever.assert_called_with(k=8, bm25_weight=0.7, chroma_weight=0.3)

    # Test CONCEPT intent
    state["intent"] = "CONCEPT"
    result = adaptive_retriever(state)

    mock_get_retriever.assert_called_with(k=8, bm25_weight=0.3, chroma_weight=0.7)
