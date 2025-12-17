"""Unit tests for RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ragchain.rag import ingest_documents, search


@pytest.mark.asyncio
async def test_ingest_empty():
    """Test ingesting empty doc list."""
    result = await ingest_documents([])
    assert result["status"] == "ok"
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_ingest_and_search():
    """Test ingesting and searching documents using mock embeddings."""
    # Mock OllamaEmbeddings to avoid requiring Ollama server
    with patch("ragchain.rag.OllamaEmbeddings") as MockEmbeddings:
        mock_embed = MagicMock()
        mock_embed.embed_documents.return_value = [[0.1] * 384 for _ in range(2)]
        mock_embed.embed_query.return_value = [0.1] * 384
        MockEmbeddings.return_value = mock_embed

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
