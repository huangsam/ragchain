"""RAG pipeline orchestration using LangChain."""

import logging
import time
from collections import defaultdict
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
        import concurrent.futures

        logger.debug(f"[EnsembleRetriever] Query: {query[:50]}...")
        start = time.time()

        # Fetch from both retrievers in parallel threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_retriever.invoke, query)
            chroma_future = executor.submit(self.chroma_retriever.invoke, query)

            bm25_docs = bm25_future.result()
            chroma_docs = chroma_future.result()

        logger.debug(f"[EnsembleRetriever] Parallel retrieval: BM25={len(bm25_docs)}, Chroma={len(chroma_docs)} in {time.time() - start:.2f}s")

        # Compute RRF scores for each document
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

        logger.debug(f"[EnsembleRetriever] RRF combined {len(sorted_docs)} unique docs in {time.time() - start:.2f}s")

        # Return documents in score order
        return [doc_map[content] for content, _ in sorted_docs]

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
    clear_retriever_cache()

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }


# Global retriever cache to avoid rebuilding BM25 index on every request
_retriever_cache: Dict[tuple, EnsembleRetriever] = {}


def clear_retriever_cache():
    """Clear the retriever cache. Call after ingesting new documents."""
    global _retriever_cache
    _retriever_cache.clear()
    logger.info("[get_ensemble_retriever] Cache cleared")


def get_ensemble_retriever(k: int = 8, bm25_weight: float = 0.4, chroma_weight: float = 0.6) -> EnsembleRetriever:
    """Create an ensemble retriever combining BM25 and Chroma vector search.

    Uses a cache to avoid rebuilding the BM25 index on every request.
    Cache is keyed by (k, bm25_weight, chroma_weight).

    Args:
        k: Number of results per retriever
        bm25_weight: Weight for BM25 results in RRF (default: 0.4)
        chroma_weight: Weight for Chroma results in RRF (default: 0.6)

    Returns:
        EnsembleRetriever instance (cached if available)
    """
    cache_key = (k, bm25_weight, chroma_weight)

    if cache_key in _retriever_cache:
        logger.debug(f"[get_ensemble_retriever] Cache hit for k={k}, weights=({bm25_weight}, {chroma_weight})")
        return _retriever_cache[cache_key]

    logger.debug(f"[get_ensemble_retriever] Cache miss, creating new retriever with k={k}, bm25={bm25_weight}, chroma={chroma_weight}")
    start = time.time()

    store = get_vector_store()

    # Get all documents from Chroma for BM25
    chroma_data = store.get()

    # Handle potential None values in metadatas
    documents = chroma_data.get("documents", [])
    metadatas = chroma_data.get("metadatas", [])

    logger.debug(f"[get_ensemble_retriever] Loaded {len(documents)} documents from Chroma")

    # Ensure metadatas has same length as documents, fill with empty dicts if needed
    if len(metadatas) < len(documents):
        metadatas.extend([{} for _ in range(len(documents) - len(metadatas))])

    docs = [Document(page_content=doc, metadata=meta if meta else {}) for doc, meta in zip(documents, metadatas)]

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs, k=k)
    logger.debug(f"[get_ensemble_retriever] BM25 initialized with k={k} over {len(docs)} docs")

    # Create Chroma retriever
    chroma_retriever = store.as_retriever(search_kwargs={"k": k})

    # Create ensemble retriever with specified weights
    retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=bm25_weight,
        chroma_weight=chroma_weight,
    )

    # Cache the retriever
    _retriever_cache[cache_key] = retriever
    logger.debug(f"[get_ensemble_retriever] Ensemble created and cached in {time.time() - start:.2f}s")

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
