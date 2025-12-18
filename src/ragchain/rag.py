"""RAG pipeline orchestration using LangChain."""

import logging
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragchain.config import config

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """Custom ensemble retriever combining BM25 and vector search."""

    bm25_retriever: BM25Retriever
    chroma_retriever: VectorStoreRetriever
    bm25_weight: float = 0.4
    chroma_weight: float = 0.6

    def _parallel_retrieve(self, query: str) -> tuple[List[Document], List[Document]]:
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

    def _compute_rrf_scores(self, bm25_docs: List[Document], chroma_docs: List[Document]) -> List[Document]:
        """Compute Reciprocal Rank Fusion scores and return sorted documents.

        Args:
            bm25_docs: Documents from BM25 retrieval.
            chroma_docs: Documents from Chroma retrieval.

        Returns:
            Documents sorted by RRF score.
        """
        # RRF constant k=60 (standard value that prevents rank 1 from dominating)
        rrf_k = 60
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        # Score BM25 results with weight
        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            rrf_score = self.bm25_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        # Score Chroma results with weight
        for rank, doc in enumerate(chroma_docs):
            content = doc.page_content
            rrf_score = self.chroma_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        # Sort by combined RRF score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]

    def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        """Retrieve documents using Reciprocal Rank Fusion (RRF) with parallel execution.

        Fetches BM25 and Chroma results in parallel threads, then combines rankings
        using RRF: score = weight / (rank + 60). This allows documents that appear
        in both rankings to outrank those appearing in only one.

        Args:
            query: The search query.

        Returns:
            List of retrieved documents sorted by RRF score.
        """
        logger.debug(f"[EnsembleRetriever] Query: {query[:50]}...")
        start = time.time()

        # Fetch from both retrievers in parallel
        bm25_docs, chroma_docs = self._parallel_retrieve(query)
        logger.debug(f"[EnsembleRetriever] Parallel retrieval: BM25={len(bm25_docs)}, Chroma={len(chroma_docs)} in {time.time() - start:.2f}s")

        # Compute RRF scores and sort
        sorted_docs = self._compute_rrf_scores(bm25_docs, chroma_docs)
        logger.debug(f"[EnsembleRetriever] RRF combined {len(sorted_docs)} unique docs in {time.time() - start:.2f}s")

        return sorted_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using parallel retrieval (default behavior)."""
        return self._get_relevant_documents(query)


def get_embedder():
    """Create Ollama embedding function.

    Returns OllamaEmbeddings configured with qwen3-embedding:0.6b model.
    Uses 1024-dimensional vector embeddings with 32k token context window.

    Returns:
        OllamaEmbeddings instance configured with model and base URL from env vars.
    """
    return OllamaEmbeddings(model=config.ollama_embed_model, base_url=config.ollama_base_url, num_ctx=32768)


def get_vector_store():
    """Get or create Chroma vector store for semantic search.

    Returns either remote Chroma (HTTP) or local persistent Chroma depending on
    CHROMA_SERVER_URL environment variable.

    Returns:
        Chroma instance configured with embedder and collection name.
    """
    embedder = get_embedder()

    if config.chroma_server_url:
        # Remote Chroma server - use ChromaClient
        from chromadb import HttpClient

        parsed = urlparse(config.chroma_server_url)
        client = HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            client=client,
        )
    else:
        # Persistent local Chroma
        Path(config.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            persist_directory=config.chroma_persist_directory,
        )


async def ingest_documents(docs: List[Document]) -> dict:
    """Process and store documents in vector store.

    Pipeline: Split docs → Embed chunks → Store in Chroma.

    Args:
        docs: List of LangChain Documents to ingest

    Returns:
        dict with status, count, and message
    """
    if not docs:
        return {"status": "ok", "count": 0, "message": "No documents to ingest"}

    start_time = time.perf_counter()

    # Split documents into chunks (optimized for semantic quality)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Add to vector store (LangChain handles embedding internally)
    store = get_vector_store()
    store.add_documents(chunks)

    # Clear retriever cache to pick up new documents
    get_ensemble_retriever.cache_clear()

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }


def _load_documents_from_chroma(store: Chroma) -> List[Document]:
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

    return [Document(page_content=doc, metadata=meta if meta else {}) for doc, meta in zip(documents, metadatas)]


def _create_bm25_retriever(docs: List[Document], k: int) -> BM25Retriever:
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
def get_ensemble_retriever(k: int = 8, bm25_weight: float = 0.4, chroma_weight: float = 0.6) -> EnsembleRetriever:
    """Create an ensemble retriever combining BM25 and Chroma vector search.

    Uses LRU cache to avoid rebuilding the BM25 index on every request.
    Cache is keyed by (k, bm25_weight, chroma_weight).

    Args:
        k: Number of results per retriever
        bm25_weight: Weight for BM25 results in RRF (default: 0.4)
        chroma_weight: Weight for Chroma results in RRF (default: 0.6)

    Returns:
        EnsembleRetriever instance (cached if available)
    """
    logger.debug(f"[get_ensemble_retriever] Creating new retriever with k={k}, bm25={bm25_weight}, chroma={chroma_weight}")
    start = time.time()

    store = get_vector_store()
    docs = _load_documents_from_chroma(store)

    logger.debug(f"[get_ensemble_retriever] Loaded {len(docs)} documents from Chroma")

    # Create retrievers
    bm25_retriever = _create_bm25_retriever(docs, k)
    logger.debug(f"[get_ensemble_retriever] BM25 initialized with k={k} over {len(docs)} docs")

    chroma_retriever = _create_chroma_retriever(store, k)

    # Create ensemble retriever with specified weights
    retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=bm25_weight,
        chroma_weight=chroma_weight,
    )

    logger.debug(f"[get_ensemble_retriever] Ensemble created in {time.time() - start:.2f}s")
    return retriever


async def search(query: str, k: int = 8) -> dict:
    """Perform ensemble retrieval using BM25 and Chroma vector search.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: 8)

    Returns:
        dict with 'query' and 'results' list of {content, metadata, distance}
    """
    ensemble_retriever = get_ensemble_retriever(k)

    # Retrieve relevant documents
    results = ensemble_retriever.get_relevant_documents(query)

    # Limit to k results
    results = results[:k]

    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }
