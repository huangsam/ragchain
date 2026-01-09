"""Unit tests for retrieval utilities."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ragchain.inference.retrievers import (
    EnsembleRetriever,
    _create_bm25_retriever,
    _create_chroma_retriever,
    _load_documents_from_chroma,
    get_ensemble_retriever,
)


def test_load_documents_from_chroma(mock_chroma_store):
    """Test loading documents from Chroma store."""
    docs = _load_documents_from_chroma(mock_chroma_store)

    assert len(docs) == 2
    assert docs[0].page_content == "Doc1"
    assert docs[0].metadata == {"key": "value"}
    assert docs[1].page_content == "Doc2"
    assert docs[1].metadata == {}  # Padded with empty dict


def test_create_bm25_retriever(sample_documents):
    """Test creating BM25 retriever."""
    with patch("ragchain.inference.retrievers.BM25Retriever") as MockBM25:
        retriever = _create_bm25_retriever(sample_documents, k=5)

        MockBM25.from_documents.assert_called_once_with(sample_documents, k=5)
        assert retriever == MockBM25.from_documents.return_value


def test_create_chroma_retriever(mock_chroma_store):
    """Test creating Chroma retriever."""
    retriever = _create_chroma_retriever(mock_chroma_store, k=10)

    mock_chroma_store.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
    assert retriever == mock_chroma_store.as_retriever.return_value


def test_ensemble_retriever_parallel_retrieve(mock_bm25_retriever, mock_chroma_retriever):
    """Test parallel retrieval in EnsembleRetriever."""
    # Create a mock retriever instance to test the method
    retriever = MagicMock()
    retriever.bm25_weight = 0.5
    retriever.chroma_weight = 0.5
    retriever.bm25_retriever = mock_bm25_retriever
    retriever.chroma_retriever = mock_chroma_retriever

    # Call the actual method
    from ragchain.inference.retrievers import EnsembleRetriever

    bm25_docs, chroma_docs = EnsembleRetriever._parallel_retrieve(retriever, "test query")

    assert len(bm25_docs) == 1
    assert len(chroma_docs) == 1
    mock_bm25_retriever.invoke.assert_called_once_with("test query")
    mock_chroma_retriever.invoke.assert_called_once_with("test query")


def test_ensemble_retriever_compute_rrf_scores():
    """Test RRF score computation."""
    # Test the method directly on a mock
    retriever = MagicMock()
    retriever.bm25_weight = 0.4
    retriever.chroma_weight = 0.6

    bm25_docs = [Document(page_content="Shared doc"), Document(page_content="BM25 only")]
    chroma_docs = [Document(page_content="Shared doc"), Document(page_content="Chroma only")]

    from ragchain.inference.retrievers import EnsembleRetriever

    sorted_docs = EnsembleRetriever._compute_rrf_scores(retriever, bm25_docs, chroma_docs)

    assert len(sorted_docs) == 3
    # Shared doc should be first due to higher combined score
    assert sorted_docs[0].page_content == "Shared doc"


def test_ensemble_retriever_get_relevant_documents():
    """Test full retrieval pipeline."""
    with (
        patch("ragchain.inference.retrievers.time") as MockTime,
    ):
        MockTime.time.return_value = 0
        # Create a mock retriever with the needed attributes
        retriever = MagicMock()
        retriever.bm25_weight = 0.5
        retriever.chroma_weight = 0.5
        retriever._parallel_retrieve.return_value = ([Document(page_content="Doc1")], [Document(page_content="Doc2")])

        from ragchain.inference.retrievers import EnsembleRetriever

        docs = EnsembleRetriever._get_relevant_documents(retriever, "test query")

        assert len(docs) <= 10  # Limited to top 10


@patch.object(EnsembleRetriever, "__init__", return_value=None)
def test_get_ensemble_retriever_caching(mock_init):
    """Test LRU caching in get_ensemble_retriever."""
    with (
        patch("ragchain.ingestion.storage.get_vector_store") as MockStore,
        patch("ragchain.inference.retrievers._load_documents_from_chroma") as MockLoad,
        patch("ragchain.inference.retrievers._create_bm25_retriever"),
        patch("ragchain.inference.retrievers._create_chroma_retriever"),
        patch("ragchain.inference.retrievers.time") as MockTime,
    ):
        MockTime.time.return_value = 0
        mock_store = MagicMock()
        MockStore.return_value = mock_store
        MockLoad.return_value = [Document(page_content="Test doc")]

        # First call
        retriever1 = get_ensemble_retriever(5, bm25_weight=0.4, chroma_weight=0.6)
        assert MockStore.call_count == 1

        # Second call with same params should use cache
        retriever2 = get_ensemble_retriever(5, bm25_weight=0.4, chroma_weight=0.6)
        assert MockStore.call_count == 1  # Not called again
        assert retriever1 is retriever2

        # Different params should create new
        get_ensemble_retriever(10, bm25_weight=0.4, chroma_weight=0.6)
        assert MockStore.call_count == 2


def test_get_ensemble_retriever_creation():
    """Test retriever creation with mocked dependencies."""
    with (
        patch("ragchain.ingestion.storage.get_vector_store") as MockStore,
        patch("ragchain.inference.retrievers._load_documents_from_chroma") as MockLoad,
        patch("ragchain.inference.retrievers._create_bm25_retriever") as MockBM25,
        patch("ragchain.inference.retrievers._create_chroma_retriever") as MockChroma,
        patch("ragchain.inference.retrievers.time") as MockTime,
        patch.object(EnsembleRetriever, "__init__", return_value=None),
    ):
        MockTime.time.return_value = 0
        mock_store = MagicMock()
        MockStore.return_value = mock_store
        MockLoad.return_value = [Document(page_content="Test doc")]
        mock_bm25 = MagicMock()
        mock_chroma = MagicMock()
        MockBM25.return_value = mock_bm25
        MockChroma.return_value = mock_chroma

        get_ensemble_retriever(5)

        MockLoad.assert_called_once_with(mock_store)
        MockBM25.assert_called_once_with([Document(page_content="Test doc")], 5)
        MockChroma.assert_called_once_with(mock_store, 5)
