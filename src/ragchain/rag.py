"""RAG pipeline orchestration using LangChain."""

import os
import time
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
CHROMA_SERVER_URL = os.environ.get("CHROMA_SERVER_URL", "http://localhost:8000")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding")


class EnsembleRetriever(BaseRetriever):
    """Custom ensemble retriever combining BM25 and vector search."""

    bm25_retriever: BM25Retriever
    chroma_retriever: VectorStoreRetriever
    bm25_weight: float = 0.4
    chroma_weight: float = 0.6

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get results from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        chroma_docs = self.chroma_retriever.invoke(query)

        # Assign scores based on reciprocal rank with weights
        scored_docs = []
        for rank, doc in enumerate(bm25_docs):
            score = self.bm25_weight / (rank + 1)
            scored_docs.append((doc, score))

        for rank, doc in enumerate(chroma_docs):
            score = self.chroma_weight / (rank + 1)
            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates based on content, keep highest score
        seen = set()
        unique_docs = []
        for doc, score in scored_docs:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        return unique_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def get_embedder():
    """Create Ollama embedding function.

    Returns OllamaEmbeddings configured with qwen3-embedding model.
    Uses 4096-dimensional vector embeddings for rich semantic representation.

    Returns:
        OllamaEmbeddings instance configured with model and base URL from env vars.
    """
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL, num_ctx=40960)


def get_vector_store():
    """Get or create Chroma vector store for semantic search.

    Returns either remote Chroma (HTTP) or local persistent Chroma depending on
    CHROMA_SERVER_URL environment variable.

    Returns:
        Chroma instance configured with embedder and collection name.
    """
    embedder = get_embedder()

    if CHROMA_SERVER_URL:
        # Remote Chroma server - use ChromaClient
        from chromadb import HttpClient

        parsed = urlparse(CHROMA_SERVER_URL)
        client = HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            client=client,
        )
    else:
        # Persistent local Chroma
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            persist_directory=CHROMA_PERSIST_DIR,
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

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }


def get_ensemble_retriever(k: int = 8):
    """Create an ensemble retriever combining BM25 and Chroma vector search.

    Args:
        k: Number of results per retriever

    Returns:
        EnsembleRetriever instance
    """
    store = get_vector_store()

    # Get all documents from Chroma for BM25
    chroma_data = store.get()
    docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(chroma_data["documents"], chroma_data["metadatas"])]

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)

    # Create Chroma retriever
    chroma_retriever = store.as_retriever(search_kwargs={"k": k})

    # Create ensemble retriever with weights favoring vector search slightly
    return EnsembleRetriever(bm25_retriever=bm25_retriever, chroma_retriever=chroma_retriever, bm25_weight=0.4, chroma_weight=0.6)


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
