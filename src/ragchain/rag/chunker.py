from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text into pieces of `chunk_size` chars with `overlap` chars overlapped.

    Simple character-based sliding window chunker. For token-based chunking,
    integrate tokenizer of the target embedding model later.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    n = len(text)
    if n <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        # next start is end - overlap
        start = end - overlap
    return chunks
