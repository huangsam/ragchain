import pytest

from ragchain.rag.embeddings import DummyEmbedding, LocalSentenceTransformer


@pytest.mark.asyncio
async def test_dummy_embedding_empty_text():
    """Test DummyEmbedding with empty string."""
    emb = DummyEmbedding()
    result = await emb.embed_texts([""])
    assert len(result) == 1
    assert len(result[0]) == 32  # Default dim


@pytest.mark.asyncio
async def test_dummy_embedding_consistency():
    """Test that DummyEmbedding produces consistent results."""
    emb = DummyEmbedding()
    text = "test text"
    result1 = await emb.embed_texts([text])
    result2 = await emb.embed_texts([text])
    # Embeddings should be identical
    assert result1[0] == result2[0]


@pytest.mark.asyncio
async def test_dummy_embedding_large_text():
    """Test DummyEmbedding with very long text."""
    emb = DummyEmbedding()
    large_text = "a" * 10000
    result = await emb.embed_texts([large_text])
    assert len(result) == 1
    assert len(result[0]) == 32


@pytest.mark.asyncio
async def test_dummy_embedding_batch():
    """Test DummyEmbedding with multiple texts."""
    emb = DummyEmbedding()
    texts = ["text1", "text2", "text3", "very different text"]
    result = await emb.embed_texts(texts)
    assert len(result) == len(texts)
    # All embeddings should be different (with high probability for different inputs)
    assert result[0] != result[1]
    assert result[2] != result[3]


@pytest.mark.asyncio
async def test_dummy_embedding_custom_dim():
    """Test DummyEmbedding with custom dimension (limited by SHA256 32-byte digest)."""
    # SHA256 produces 32 bytes, so max dim is 32
    emb = DummyEmbedding(dim=16)
    result = await emb.embed_texts(["test"])
    assert len(result[0]) == 16


@pytest.mark.skipif(LocalSentenceTransformer is None, reason="sentence-transformers not installed")
@pytest.mark.asyncio
async def test_local_sentence_transformer_consistency():
    """Test that LocalSentenceTransformer produces consistent results."""
    emb = LocalSentenceTransformer()
    text = "machine learning is great"
    result1 = await emb.embed_texts([text])
    result2 = await emb.embed_texts([text])
    # Should be identical (deterministic)
    assert result1[0] == result2[0]


@pytest.mark.skipif(LocalSentenceTransformer is None, reason="sentence-transformers not installed")
@pytest.mark.asyncio
async def test_local_sentence_transformer_batch():
    """Test LocalSentenceTransformer with batch of texts."""
    emb = LocalSentenceTransformer()
    texts = ["hello world", "machine learning", "python programming"]
    result = await emb.embed_texts(texts)
    assert len(result) == len(texts)
    # All should have same dimensionality
    assert all(len(r) == len(result[0]) for r in result)


@pytest.mark.skipif(LocalSentenceTransformer is None, reason="sentence-transformers not installed")
@pytest.mark.asyncio
async def test_local_sentence_transformer_similar_texts():
    """Test that similar texts produce similar embeddings."""
    emb = LocalSentenceTransformer()
    result = await emb.embed_texts(["dog is an animal", "cat is an animal", "programming is hard"])

    # Compute simple cosine similarity
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

    sim_01 = cosine_sim(result[0], result[1])  # dog vs cat (similar)
    sim_02 = cosine_sim(result[0], result[2])  # dog vs programming (different)

    # Similar texts should have higher cosine similarity
    assert sim_01 > sim_02, f"Expected similar texts to be more similar: {sim_01} vs {sim_02}"
