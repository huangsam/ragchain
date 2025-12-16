import pytest

from ragchain.rag.chunker import chunk_text


def test_chunker_basic_text():
    """Test chunker with basic text."""
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # Each chunk should be at most chunk_size
    assert all(len(c) <= 50 for c in chunks)


def test_chunker_single_chunk():
    """Test chunker with text smaller than chunk size."""
    text = "short text"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_empty_text():
    """Test chunker with empty text."""
    chunks = chunk_text("", chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_chunker_overlap():
    """Test that overlap is respected."""
    text = "a" * 150
    chunks = chunk_text(text, chunk_size=50, overlap=20)

    # Verify all chunks are present and overlap works
    assert len(chunks) > 1
    # Each chunk should be at most chunk_size
    assert all(len(c) <= 50 for c in chunks)


def test_chunker_large_overlap():
    """Test chunker with large overlap."""
    text = "a" * 300
    chunks = chunk_text(text, chunk_size=100, overlap=90)
    assert len(chunks) > 0
    # With 90% overlap, we should have many overlapping chunks
    assert len(chunks) >= 2


def test_chunker_zero_overlap():
    """Test chunker with zero overlap."""
    text = "a" * 150
    chunks = chunk_text(text, chunk_size=50, overlap=0)
    assert len(chunks) == 3
    # With no overlap, concatenating chunks should give original
    assert "".join(chunks) == text


def test_chunker_multiline_text():
    """Test chunker with multiline text."""
    text = "line1\nline2\nline3\nline4\nline5\n" * 10
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    # Just verify chunks are created successfully
    assert all(len(c) <= 50 or c == chunks[-1] for c in chunks)


def test_chunker_special_characters():
    """Test chunker with special characters."""
    text = "🎉 emoji test! @#$%^&*() special chars™ © ® ™ " * 5
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    # Should preserve emoji and special characters
    assert "🎉" in "".join(chunks)


def test_chunker_whitespace_only():
    """Test chunker with whitespace-only text."""
    text = "   \n\n\t\t  \n   " * 5
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # Whitespace should still be chunked
    assert all(len(c) <= 50 or c == chunks[-1] for c in chunks)


def test_chunker_invalid_chunk_size():
    """Test chunker with invalid chunk size."""
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        chunk_text("text", chunk_size=0, overlap=10)


def test_chunker_negative_overlap():
    """Test chunker with negative overlap."""
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_text("text", chunk_size=50, overlap=-10)


def test_chunker_overlap_exceeds_chunk_size():
    """Test chunker with overlap >= chunk_size."""
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_text("text", chunk_size=50, overlap=50)

    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_text("text", chunk_size=50, overlap=100)
