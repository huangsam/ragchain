from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ragchain.rag.embeddings import DummyEmbedding, EmbeddingClient
from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore


class IngestRequest(BaseModel):
    titles: List[str]


class SearchRequest(BaseModel):
    query: str
    n_results: int = 4


app = FastAPI(title="ragchain API", version="0.1")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest, embedding: Optional[str] = None) -> Dict[str, Any]:
    """Trigger ingest for provided titles. Uses DummyEmbedding by default."""
    emb_client: EmbeddingClient = DummyEmbedding()

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
    emb = DummyEmbedding()
    vec = (await emb.embed_texts([req.query]))[0]
    store = ChromaVectorStore()
    results = await store.search(vec, n_results=req.n_results)
    return {"results": results}
