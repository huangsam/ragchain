"""RAG pipeline orchestration using LangChain."""

import os
import time
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
CHROMA_SERVER_URL = os.environ.get("CHROMA_SERVER_URL", None)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding")


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

        client = HttpClient(
            host=CHROMA_SERVER_URL.replace("http://", "").split(":")[0], port=int(CHROMA_SERVER_URL.split(":")[-1]) if ":" in CHROMA_SERVER_URL else 8000
        )
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

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
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


async def search(query: str, k: int = 4) -> dict:
    """Perform semantic similarity search on stored documents.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: 4)

    Returns:
        dict with 'query' and 'results' list of {content, metadata, distance}
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k)

    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }
