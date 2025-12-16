from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text into pieces of `chunk_size` chars with `overlap` chars overlapped.

    Simple character-based sliding window chunker. For token-based chunking,
    integrate tokenizer of the target embedding model later.
    """
    # Validate caller-supplied parameters early to avoid surprising behavior.
    # These checks are intentionally strict because nonsensical values could
    # cause an infinite loop or produce empty chunks.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    n = len(text)
    # If text is short enough, return it as a single chunk — no windowing needed.
    if n <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    # Slide a fixed-size window across the text with `overlap` characters
    # of overlap between consecutive chunks. We use `while` because the
    # last chunk may be smaller than `chunk_size` and we break explicitly
    # when we reach the end.
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        # Advance the window but preserve `overlap` characters so that
        # context is retained across chunk boundaries.
        start = end - overlap
    return chunks
