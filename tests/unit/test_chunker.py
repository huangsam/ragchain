import pytest

from ragchain.rag.chunker import chunk_text


def test_chunk_text_small_no_overlap():
    text = "short text"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert chunks == [text]


def test_chunk_text_large_overlap():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    # expect more than 1 chunk
    assert len(chunks) >= 3
    # check overlap exists
    assert chunks[0].endswith("A")
    assert chunks[1].startswith("A")
    # overlapping region should be present in both
    assert chunks[0][-100:] == chunks[1][:100]


@pytest.mark.parametrize("chunk_size,overlap", [(10, 10), (0, 1), (-1, 1)])
def test_chunk_text_invalid(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=chunk_size, overlap=overlap)
