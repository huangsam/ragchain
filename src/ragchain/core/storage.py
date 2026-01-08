"""Storage utilities for RAG pipeline: embeddings, vector store, and document ingestion."""

import logging
import time
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragchain.data.config import config

logger = logging.getLogger(__name__)


class IngestResult(TypedDict):
    """Result of document ingestion operation."""

    status: str
    count: int
    message: str
    elapsed_seconds: float


def get_embedder() -> OllamaEmbeddings:
    """Create Ollama embedding function.

    Returns OllamaEmbeddings configured with bge-m3 model.
    Uses 1024-dimensional vector embeddings with 8k token context window.

    Returns:
        OllamaEmbeddings instance configured with model and base URL from env vars.
    """
    return OllamaEmbeddings(model=config.ollama_embed_model, base_url=config.ollama_base_url, num_ctx=config.ollama_embed_ctx)


def get_vector_store() -> Chroma:
    """Get or create Chroma vector store for semantic search.

    Returns either remote Chroma (HTTP) or local persistent Chroma depending on
    CHROMA_SERVER_URL environment variable.

    Returns:
        Chroma instance configured with embedder and collection name.
    """
    embedder = get_embedder()

    if config.chroma_server_url:
        from chromadb import HttpClient

        parsed = urlparse(config.chroma_server_url)
        client = HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            client=client,
        )
    else:
        Path(config.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            persist_directory=config.chroma_persist_directory,
        )


async def ingest_documents(docs: list[Document]) -> IngestResult:
    """Process and store documents in vector store.

    Pipeline: Split docs → Embed chunks → Store in Chroma.

    Args:
        docs: List of LangChain Documents to ingest

    Returns:
        dict with status, count, and message
    """
    if not docs:
        return {"status": "ok", "count": 0, "message": "No documents to ingest", "elapsed_seconds": 0.0}

    start_time = time.perf_counter()

    # Increased overlap to 500 chars (20%) to fix "Missing Paragraph" issues where key context
    # spans across chunk boundaries (e.g., Comparing interpreted vs compiled languages).
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    chunks = splitter.split_documents(docs)

    store = get_vector_store()

    store.add_documents(chunks)

    # Clear retriever cache to ensure fresh data
    from ragchain.core.retrievers import get_ensemble_retriever

    get_ensemble_retriever.cache_clear()

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }
