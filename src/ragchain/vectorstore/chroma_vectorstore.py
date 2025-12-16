from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - chromadb is optional for tests
    chromadb = None


class ChromaVectorStore:
    """Simple Chroma adapter with async-friendly methods (runs blocking calls in a threadpool)."""

    def __init__(
        self, client: Optional[Any] = None, collection_name: str = "ragchain", persist_directory: Optional[str] = None
    ):
        if chromadb is None:
            raise ImportError("chromadb is not installed")

        if client is None:
            if persist_directory:
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                client = chromadb.Client()

        self.client = client
        self.collection_name = collection_name
        self._collection = self.client.get_or_create_collection(self.collection_name)
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def upsert_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Upsert documents into Chroma collection.

        docs: list of {id: str, text: str, metadata: dict, embedding: List[float]}
        """
        ids = [d["id"] for d in docs]
        documents = [d.get("text", "") for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = [d["embedding"] for d in docs]

        loop = asyncio.get_running_loop()

        def _upsert():
            # chroma's upsert signature uses keyword args
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        await loop.run_in_executor(self._executor, _upsert)

    async def search(self, embedding: List[float], n_results: int = 4) -> Dict[str, Any]:
        """Search the collection by an embedding vector and return the raw Chroma response."""

        loop = asyncio.get_running_loop()

        def _query():
            # include metadatas and documents to inspect results
            return self._collection.query(
                query_embeddings=[embedding], n_results=n_results, include=["metadatas", "documents", "distances"]
            )

        res = await loop.run_in_executor(self._executor, _query)
        return res
