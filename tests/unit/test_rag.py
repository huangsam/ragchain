"""Unit tests for RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ragchain.config import config
from ragchain.inference.rag import search
from ragchain.ingestion.storage import ingest_documents


@pytest.mark.asyncio
async def test_ingest_and_search(mock_embedder):
    """Test ingesting and searching documents using mock embeddings."""
    # Use 1024 dimensions to match bge-m3
    with (
        patch("ragchain.ingestion.storage.get_embedder", return_value=mock_embedder),
        patch.object(config, "chroma_server_url", None),
    ):  # Force local Chroma for testing
        # Create sample docs
        docs = [
            Document(
                page_content="Python is a high-level programming language.",
                metadata={"language": "Python"},
            ),
            Document(
                page_content="Java is an object-oriented language.",
                metadata={"language": "Java"},
            ),
        ]

        # Ingest
        result = await ingest_documents(docs)
        assert result["status"] == "ok"
        assert result["count"] >= 2

        # Search
        search_result = await search("Python programming", k=1)
        assert "results" in search_result
        assert len(search_result["results"]) >= 1


@pytest.mark.asyncio
async def test_search_empty_query():
    """Test search with empty query."""
    with patch("ragchain.inference.rag.get_ensemble_retriever") as MockRetriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        MockRetriever.return_value = mock_retriever

        result = await search("", k=5)

        assert result["query"] == ""
        assert result["results"] == []
        MockRetriever.assert_called_once_with(k=5)


@pytest.mark.asyncio
async def test_search_k_zero():
    """Test search with k=0."""
    with patch("ragchain.inference.rag.get_ensemble_retriever") as MockRetriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="Test")]
        MockRetriever.return_value = mock_retriever

        result = await search("test", k=0)

        assert result["query"] == "test"
        assert result["results"] == []  # Should limit to k=0
        MockRetriever.assert_called_once_with(k=0)
