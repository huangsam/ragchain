"""Unit tests for graph operations."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ragchain.inference.router import Intent
from ragchain.types import Node


@patch("ragchain.inference.graph.get_ensemble_retriever")
def test_adaptive_retriever(mock_get_retriever):
    """Test adaptive retriever with different intents."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test doc", metadata={})]
    mock_get_retriever.return_value = mock_retriever

    from ragchain.inference.graph import adaptive_retriever

    # Test FACT intent
    state = {"query": "What are top languages?", "intent": Intent.FACT, "retrieved_docs": []}
    result = adaptive_retriever(state)

    assert len(result["retrieved_docs"]) == 1
    mock_get_retriever.assert_called_with(6, bm25_weight=0.8, chroma_weight=0.2)

    # Test CONCEPT intent
    state["intent"] = Intent.CONCEPT
    result = adaptive_retriever(state)

    mock_get_retriever.assert_called_with(6, bm25_weight=0.4, chroma_weight=0.6)


def test_node_enum():
    """Test Node enum values."""
    assert Node.INTENT_ROUTER == "intent_router"
    assert Node.ADAPTIVE_RETRIEVER == "adaptive_retriever"
    assert Node.RETRIEVAL_GRADER == "retrieval_grader"
    assert Node.QUERY_REWRITER == "query_rewriter"
