"""Unit tests for storage utilities."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ragchain.config import config
from ragchain.ingestion.storage import get_embedder, get_vector_store, ingest_documents


def test_get_embedder():
    """Test get_embedder creates OllamaEmbeddings with correct config."""
    with patch("ragchain.ingestion.storage.OllamaEmbeddings") as MockEmbeddings:
        embedder = get_embedder()
        MockEmbeddings.assert_called_once_with(
            model=config.ollama_embed_model,
            base_url=config.ollama_base_url,
            num_ctx=config.ollama_embed_ctx,
        )
        assert embedder == MockEmbeddings.return_value


def test_get_vector_store_local():
    """Test get_vector_store creates local Chroma store."""
    with (
        patch("ragchain.ingestion.storage.OllamaEmbeddings") as MockEmbeddings,
        patch("ragchain.ingestion.storage.Chroma") as MockChroma,
        patch("ragchain.ingestion.storage.Path") as MockPath,
        patch.object(config, "chroma_server_url", None),
    ):
        mock_embed = MagicMock()
        MockEmbeddings.return_value = mock_embed
        mock_path = MagicMock()
        MockPath.return_value = mock_path

        store = get_vector_store()

        mock_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        MockChroma.assert_called_once_with(
            collection_name="ragchain",
            embedding_function=mock_embed,
            persist_directory=config.chroma_persist_directory,
        )
        assert store == MockChroma.return_value


def test_get_vector_store_remote():
    """Test get_vector_store creates remote Chroma store."""
    with (
        patch("ragchain.ingestion.storage.OllamaEmbeddings") as MockEmbeddings,
        patch("ragchain.ingestion.storage.Chroma") as MockChroma,
        patch("chromadb.HttpClient") as MockHttpClient,
        patch.object(config, "chroma_server_url", "http://localhost:8000"),
    ):
        mock_embed = MagicMock()
        MockEmbeddings.return_value = mock_embed
        mock_client = MagicMock()
        MockHttpClient.return_value = mock_client

        store = get_vector_store()

        MockHttpClient.assert_called_once_with(host="localhost", port=8000)
        MockChroma.assert_called_once_with(
            collection_name="ragchain",
            embedding_function=mock_embed,
            client=mock_client,
        )
        assert store == MockChroma.return_value


@pytest.mark.asyncio
async def test_ingest_empty():
    """Test ingesting empty doc list."""
    result = await ingest_documents([])
    assert result["status"] == "ok"
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_ingest_documents_success(sample_documents, mock_embedder):
    """Test ingesting documents successfully."""
    with (
        patch("ragchain.ingestion.storage.get_embedder", return_value=mock_embedder),
        patch("ragchain.ingestion.storage.Chroma") as MockChroma,
        patch("ragchain.retrieval.retrievers.get_ensemble_retriever") as MockFunc,
        patch.object(config, "chroma_server_url", None),
    ):
        mock_store = MagicMock()
        MockChroma.return_value = mock_store

        result = await ingest_documents(sample_documents)

        assert result["status"] == "ok"
        assert result["count"] >= 2  # sample_documents has 2 docs
        mock_store.add_documents.assert_called_once()
        MockFunc.cache_clear.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_documents_chunking(mock_embedder):
    """Test document chunking in ingest_documents."""
    with (
        patch("ragchain.ingestion.storage.get_embedder", return_value=mock_embedder),
        patch("ragchain.ingestion.storage.Chroma") as MockChroma,
        patch("ragchain.retrieval.retrievers.get_ensemble_retriever"),
        patch.object(config, "chroma_server_url", None),
    ):
        mock_store = MagicMock()
        MockChroma.return_value = mock_store

        # Create a long document to test chunking
        long_content = "Test content. " * 1000  # Long content
        docs = [Document(page_content=long_content, metadata={"test": True})]

        result = await ingest_documents(docs)

        assert result["status"] == "ok"
        assert result["count"] > 1  # Should be chunked
        mock_store.add_documents.assert_called_once()
        chunks = mock_store.add_documents.call_args[0][0]
        assert len(chunks) > 1
        # Check that chunks have overlap by checking the length
        assert all(len(chunk.page_content) <= 2500 for chunk in chunks)
