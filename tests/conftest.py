"""Pytest configuration."""

import warnings
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

# Filter out coroutine warnings from mocked CLI tests
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Enable tracemalloc to get the object allocation traceback.*", category=RuntimeWarning)


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provide a temporary Chroma directory for integration tests."""
    import os

    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(tmp_path / "chroma")
    yield str(tmp_path / "chroma")


@pytest.fixture
def mock_embedder():
    """Provide a mocked OllamaEmbeddings instance."""
    embedder = MagicMock()
    embedder.embed_documents.return_value = [[0.1] * 1024 for _ in range(2)]
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder


@pytest.fixture
def mock_chroma_store():
    """Provide a mocked Chroma vector store."""
    store = MagicMock()
    store.add_documents.return_value = None
    store.get.return_value = {"documents": ["Doc1", "Doc2"], "metadatas": [{"key": "value"}, None]}
    store.as_retriever.return_value = MagicMock()
    return store


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"language": "Python"},
        ),
        Document(
            page_content="Java is an object-oriented language.",
            metadata={"language": "Java"},
        ),
    ]


@pytest.fixture
def mock_bm25_retriever():
    """Provide a mocked BM25Retriever."""
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="BM25 result")]
    return retriever


@pytest.fixture
def mock_chroma_retriever():
    """Provide a mocked VectorStoreRetriever."""
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="Chroma result")]
    return retriever


@pytest.fixture
def mock_config():
    """Provide a mocked config object."""
    config = MagicMock()
    config.enable_intent_routing = True
    config.ollama_model = "test-model"
    config.ollama_base_url = "http://test"
    config.enable_grading = True
    return config
