from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ragchain.rag.embeddings import DummyEmbedding, EmbeddingClient, LocalSentenceTransformer
from ragchain.rag.generation import OllamaGenerator
from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore


def get_embedding_client() -> EmbeddingClient:
    """Return a real embedding model if available, else dummy."""
    if LocalSentenceTransformer is not None:
        return LocalSentenceTransformer()
    return DummyEmbedding()


class IngestRequest(BaseModel):
    titles: List[str]


class SearchRequest(BaseModel):
    query: str
    n_results: int = 4


class AskRequest(BaseModel):
    query: str
    n_results: int = 4
    model: Optional[str] = None


app = FastAPI(title="ragchain API", version="0.1")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest, embedding: Optional[str] = None) -> Dict[str, Any]:
    """Trigger ingest for provided titles. Uses LocalSentenceTransformer by default."""
    emb_client = get_embedding_client()

    # Allow configuration via environment for tests/local runs. This lets
    # callers use a local persistent store (CHROMA_PERSIST_DIRECTORY) or a
    # remote server (CHROMA_SERVER_URL) without changing application code.
    import os

    server_url = os.environ.get("CHROMA_SERVER_URL")
    persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    store = ChromaVectorStore(server_url=server_url, persist_directory=persist_dir)

    # Defer heavy work to existing ingest routine (import locally to avoid cycles)
    from ragchain.rag.ingest import ingest

    report = await ingest(
        titles=req.titles,
        save_dir=None,
        chunk_size=1000,
        overlap=200,
        embedding_client=emb_client,
        vectorstore=store,
    )
    return {"report": report.__dict__}


@app.post("/search")
async def search_endpoint(req: SearchRequest) -> Dict[str, Any]:
    emb = get_embedding_client()
    vec = (await emb.embed_texts([req.query]))[0]
    store = ChromaVectorStore()
    results = await store.search(vec, n_results=req.n_results)
    return {"results": results}


@app.post("/ask")
async def ask_endpoint(req: AskRequest) -> Dict[str, Any]:
    # 1. Retrieval
    emb = get_embedding_client()
    vec = (await emb.embed_texts([req.query]))[0]

    # Configure store (respect env vars for remote/local)
    import os

    server_url = os.environ.get("CHROMA_SERVER_URL")
    persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    store = ChromaVectorStore(server_url=server_url, persist_directory=persist_dir)

    results = await store.search(vec, n_results=req.n_results)

    # 2. Extract context
    # Chroma returns list of lists (batch). We only have one query.
    # We need both documents and metadatas to provide better context.
    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []

    # Combine into a list of dicts for the generator
    context_items = []
    for doc, meta in zip(documents, metadatas):
        context_items.append({"text": doc, "meta": meta})

    # 3. Generation
    generator = OllamaGenerator(model=req.model)
    answer = await generator.generate(req.query, context_items)

    return {
        "answer": answer,
        "context": documents,  # Keep returning raw text list for backward compat/debugging
        "model": generator.model,
    }
