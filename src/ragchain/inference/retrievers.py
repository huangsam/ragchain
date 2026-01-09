"""Retrieval utilities for RAG pipeline: ensemble retriever and helpers."""

import logging
import time
from collections import defaultdict
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from ragchain.config import config
from ragchain.utils import timed

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """Custom ensemble retriever combining BM25 and vector search."""

    bm25_retriever: BM25Retriever
    chroma_retriever: VectorStoreRetriever
    bm25_weight: float = 0.4
    chroma_weight: float = 0.6

    def _parallel_retrieve(self, query: str) -> tuple[list[Document], list[Document]]:
        """Retrieve documents from both retrievers in parallel.

        Args:
            query: The search query.

        Returns:
            Tuple of (bm25_docs, chroma_docs).
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_retriever.invoke, query)
            chroma_future = executor.submit(self.chroma_retriever.invoke, query)
            return bm25_future.result(), chroma_future.result()

    def _compute_rrf_scores(self, bm25_docs: list[Document], chroma_docs: list[Document]) -> list[Document]:
        """Compute Reciprocal Rank Fusion scores and return sorted documents.

        Args:
            bm25_docs: Documents from BM25 retrieval.
            chroma_docs: Documents from Chroma retrieval.

        Returns:
            Documents sorted by RRF score.
        """
        # RRF constant k=60 (standard value that prevents rank 1 from dominating)
        rrf_k = 60
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            rrf_score = self.bm25_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        for rank, doc in enumerate(chroma_docs):
            content = doc.page_content
            rrf_score = self.chroma_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        """Retrieve documents using Reciprocal Rank Fusion (RRF) with parallel execution.

        Fetches BM25 and Chroma results in parallel threads, then combines rankings
        using RRF: score = weight / (rank + 60). This allows documents that appear
        in both rankings to outrank those appearing in only one.

        Args:
            query: The search query.

        Returns:
            List of top 10 retrieved documents sorted by RRF score.
        """
        start = time.time()

        bm25_docs, chroma_docs = self._parallel_retrieve(query)

        sorted_docs = self._compute_rrf_scores(bm25_docs, chroma_docs)

        # Limit to configured max results to keep context manageable
        top_docs = sorted_docs[: config.retrieval_max_results]
        elapsed = time.time() - start
        logger.debug(
            f"[EnsembleRetriever] Retrieved {len(bm25_docs)} BM25 + {len(chroma_docs)} semantic, RRF returned {len(top_docs)}/{len(sorted_docs)} in {elapsed:.2f}s"
        )

        return top_docs

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Get relevant documents using parallel retrieval (default behavior)."""
        return self._get_relevant_documents(query)


def _load_documents_from_chroma(store: Chroma) -> list[Document]:
    """Load all documents from Chroma vector store.

    Args:
        store: Chroma vector store instance.

    Returns:
        List of Document objects.
    """
    chroma_data = store.get()
    documents = chroma_data.get("documents", [])
    metadatas = chroma_data.get("metadatas", [])

    # Ensure metadatas has same length as documents, fill with empty dicts if needed
    if len(metadatas) < len(documents):
        metadatas.extend([{} for _ in range(len(documents) - len(metadatas))])

    return [Document(page_content=doc, metadata=meta if meta else {}) for doc, meta in zip(documents, metadatas, strict=True)]


def _create_bm25_retriever(docs: list[Document], k: int) -> BM25Retriever:
    """Create BM25 retriever from documents.

    Args:
        docs: List of documents to index.
        k: Number of results to return.

    Returns:
        Configured BM25Retriever instance.
    """
    return BM25Retriever.from_documents(docs, k=k)


def _create_chroma_retriever(store: Chroma, k: int) -> VectorStoreRetriever:
    """Create Chroma retriever from vector store.

    Args:
        store: Chroma vector store instance.
        k: Number of results to return.

    Returns:
        Configured VectorStoreRetriever instance.
    """
    return store.as_retriever(search_kwargs={"k": k})


@lru_cache(maxsize=32)
@timed(logger, "get_ensemble_retriever")
def get_ensemble_retriever(k: int | None = None, bm25_weight: float = 0.4, chroma_weight: float = 0.6) -> EnsembleRetriever:
    """Create an ensemble retriever combining BM25 and Chroma vector search.

    Uses LRU cache to avoid rebuilding the BM25 index on every request.
    Cache is keyed by (k, bm25_weight, chroma_weight).

    Args:
        k: Number of results per retriever (default: from config.retrieval_k)
        bm25_weight: Weight for BM25 results in RRF (default: 0.4)
        chroma_weight: Weight for Chroma results in RRF (default: 0.6)

    Returns:
        EnsembleRetriever instance (cached if available)
    """
    if k is None:
        k = config.retrieval_k

    from ragchain.ingestion.storage import get_vector_store

    store = get_vector_store()
    docs = _load_documents_from_chroma(store)

    bm25_retriever = _create_bm25_retriever(docs, k)

    chroma_retriever = _create_chroma_retriever(store, k)

    retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=bm25_weight,
        chroma_weight=chroma_weight,
    )

    logger.debug(f"[get_ensemble_retriever] Initialized with {len(docs)} documents")
    return retriever
