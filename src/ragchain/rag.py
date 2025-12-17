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
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


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

    Pipeline: Split docs → Batch embed chunks → Store pre-embedded in Chroma.

    Optimizations:
    - Pre-compute all embeddings in one batch call to Ollama (much faster than individual calls)
    - Add pre-embedded documents directly to Chroma (skips embedding overhead)
    - Larger chunks (2500) reduce total chunks needed

    Args:
        docs: List of LangChain Documents to ingest

    Returns:
        dict with keys: status, count, message, elapsed_seconds, chunks_per_sec
    """
    if not docs:
        return {"status": "ok", "count": 0, "message": "No documents to ingest", "elapsed_seconds": 0.0, "chunks_per_sec": 0.0}

    start_time = time.perf_counter()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    split_time = time.perf_counter() - start_time
    print(f"⏱️  Split time: {split_time:.2f}s")

    # Pre-compute all embeddings in batch (much faster than individual calls)
    embed_start = time.perf_counter()
    embedder = get_embedder()
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(chunk_texts)  # Single batch call to Ollama
    embed_time = time.perf_counter() - embed_start
    print(f"⏱️  Embedding time: {embed_time:.2f}s for {len(chunks)} chunks")

    # Add pre-embedded documents directly to Chroma collection (skips redundant embedding)
    store_start = time.perf_counter()
    store = get_vector_store()

    # Generate unique IDs based on content hash + index to ensure no duplicates
    import hashlib
    ids = [f"{hashlib.md5(text.encode()).hexdigest()[:12]}_{i}" for i, text in enumerate(chunk_texts)]

    store._collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    store_time = time.perf_counter() - store_start
    print(f"⏱️  Chroma add time: {store_time:.2f}s")

    elapsed = time.perf_counter() - start_time
    chunks_per_sec = len(chunks) / elapsed if elapsed > 0 else 0
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s ({chunks_per_sec:.1f} chunks/sec)",
        "elapsed_seconds": elapsed,
        "chunks_per_sec": chunks_per_sec,
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
