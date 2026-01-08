"""Storage utilities for RAG pipeline: embeddings, vector store, and document ingestion."""

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragchain.config import config
from ragchain.types import IngestResult

logger = logging.getLogger(__name__)


def get_embedder() -> OllamaEmbeddings:
    """Create Ollama embedding function with model configuration."""
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

    # Configurable chunking with overlap to preserve context across boundaries
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(docs)

    store = get_vector_store()

    store.add_documents(chunks)

    # Clear retriever cache to ensure fresh data
    from ragchain.inference.retrievers import get_ensemble_retriever

    get_ensemble_retriever.cache_clear()

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }
