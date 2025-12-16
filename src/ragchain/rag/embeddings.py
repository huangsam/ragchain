from __future__ import annotations

import hashlib
from typing import Iterable, List


class EmbeddingClient:
    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError()


class DummyEmbedding(EmbeddingClient):
    """Deterministic, lightweight embedding used for tests or when no model is present.

    It turns SHA256 digests into small float vectors.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        # Create deterministic embeddings by hashing each input text. This
        # makes tests reproducible and avoids depending on an external model.
        out: List[List[float]] = []
        for t in texts:
            # Use SHA256 digest bytes as a source of pseudo-random but stable
            # values and map the values into a small float range.
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [((b % 127) - 63) / 63.0 for b in h[: self.dim]]
            out.append(vec)
        return out


try:
    from sentence_transformers import SentenceTransformer

    class LocalSentenceTransformer(EmbeddingClient):
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)

        async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
            # sentence-transformers is sync; run in threadpool to avoid blocking the event loop
            import asyncio
            loop = asyncio.get_running_loop()
            # The model.encode method releases the GIL mostly, but running in executor is safer for async
            return await loop.run_in_executor(None, lambda: self.model.encode(list(texts), show_progress_bar=False).tolist())

except Exception:  # pragma: no cover - optional dep
    LocalSentenceTransformer = None  # type: ignore
