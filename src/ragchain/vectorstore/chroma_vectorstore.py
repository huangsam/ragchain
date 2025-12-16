from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional

chromadb: ModuleType | None = None
try:
    import chromadb as _chromadb

    chromadb = _chromadb
except Exception:  # pragma: no cover - chromadb is optional for tests
    chromadb = None


class ChromaVectorStore:
    """Flexible Chroma adapter with async-friendly methods (runs blocking calls in a threadpool).

    Behavior:
    - If `server_url` is provided (or CHROMA_SERVER_URL env var is set), uses the remote HTTP client.
    - Else if `persist_directory` is provided, uses a persistent in-process client.
    - Else uses an in-memory ephemeral client.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        collection_name: str = "ragchain",
        persist_directory: Optional[str] = None,
        server_url: Optional[str] = None,
    ):
        if chromadb is None:
            raise ImportError("chromadb is not installed")

        # Server URL takes precedence. Allow environment override too.
        import os

        server_url = server_url or os.environ.get("CHROMA_SERVER_URL")

        if client is None:
            # Choose a client implementation based on configuration.
            # Precedence: explicit server_url (remote HTTP client) > persistent
            # local directory > ephemeral in-memory client. This allows tests and
            # deployments to choose the desired mode via environment or explicit
            # constructor args.
            if server_url:
                # parse server_url like http[s]://host:port
                from urllib.parse import urlparse

                parsed = urlparse(server_url)
                host = parsed.hostname or "localhost"
                port = parsed.port or (443 if parsed.scheme == "https" else 8000)
                ssl = parsed.scheme == "https"
                client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
            elif persist_directory:
                client = chromadb.PersistentClient(path=persist_directory)
            else:
                client = chromadb.Client()

        self.client = client
        self.collection_name = collection_name
        # Create or fetch the named collection - this is a light-weight op for
        # in-process clients and issues a network call for remote clients.
        self._collection = self.client.get_or_create_collection(self.collection_name)
        # Use a threadpool to run blocking Chroma operations without blocking
        # the async event loop; we keep the pool small because operations are
        # typically short-lived.
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
            # Perform the actual upsert synchronously inside threadpool. We
            # build the payload lists above so the synchronous code receives
            # plain Python lists and not async iterables.
            # chroma's upsert signature uses keyword args
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        await loop.run_in_executor(self._executor, _upsert)

    async def search(self, embedding: List[float], n_results: int = 4) -> Dict[str, Any]:
        """Search the collection by an embedding vector and return the raw Chroma response."""

        loop = asyncio.get_running_loop()

        def _query():
            # Query the collection by a single embedding. We pass a list of
            # embeddings (length-1) to match the Chroma API which supports
            # batch queries; this keeps the return structure consistent.
            # Request metadatas and documents so callers can inspect the results
            # and validate metadata-driven filters.
            return self._collection.query(query_embeddings=[embedding], n_results=n_results, include=["metadatas", "documents", "distances"])

        res = await loop.run_in_executor(self._executor, _query)
        return res
