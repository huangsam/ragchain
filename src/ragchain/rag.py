"""RAG pipeline orchestration using LangChain."""

import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
CHROMA_SERVER_URL = os.environ.get("CHROMA_SERVER_URL", None)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embedder():
    """Return Ollama embedder."""
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_vector_store():
    """Get or create Chroma vector store."""
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
    """Ingest documents: split, embed, and store in Chroma."""
    if not docs:
        return {"status": "ok", "count": 0, "message": "No documents to ingest"}

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Get store and add documents
    store = get_vector_store()
    store.add_documents(chunks)

    return {"status": "ok", "count": len(chunks), "message": f"Ingested {len(chunks)} chunks"}


async def search(query: str, k: int = 4) -> dict:
    """Search the vector store."""
    store = get_vector_store()
    results = store.similarity_search(query, k=k)

    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }
