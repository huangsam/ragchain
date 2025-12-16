import pytest

from ragchain.rag.embeddings import LocalSentenceTransformer
from ragchain.rag.ingest import ingest
from ragchain.vectorstore.chroma_vectorstore import ChromaVectorStore

# Skip if sentence-transformers is not installed or if we are in a CI environment without model download capability
try:
    import sentence_transformers  # noqa: F401

    HAS_ST = True
except ImportError:
    HAS_ST = False


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
@pytest.mark.asyncio
async def test_real_embedding_ingest_and_search(tmp_path):
    """
    Verifies that using real embeddings results in semantic search working correctly.
    We ingest two distinct topics and verify that a query matches the correct one.
    """
    # 1. Setup
    # We'll mock the wikipedia fetch to avoid network calls and ensure deterministic content
    from unittest.mock import patch

    # Create a local vector store in a temp dir
    persist_dir = tmp_path / "chroma_db"
    store = ChromaVectorStore(persist_directory=str(persist_dir))

    # Real embedding client
    emb = LocalSentenceTransformer()

    # Mock content
    # Page 1: Python
    python_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability."
    # Page 2: Java
    java_text = (
        "Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible."
    )

    # Mock the wiki_client.fetch_pages to return our fake data
    # We need to mock ragchain.parser.wiki_client.WikiClient.fetch_pages or similar
    # Actually ingest calls `fetch_pages` from `ragchain.parser.wiki_client`.

    # Let's just mock the `ingest` function's internal `fetch_and_parse` or similar?
    # `ingest` calls `fetch_pages` then `parse_page`.
    # Easier approach: Create dummy text files and bypass the fetch part?
    # `ingest` takes `titles`.

    # Let's mock `ragchain.rag.ingest.fetch_wikipedia_pages`
    with patch("ragchain.rag.ingest.fetch_wikipedia_pages") as mock_fetch:
        # Mock return values. fetch_wikipedia_pages returns a list of page dicts
        async def fake_fetch(titles, save_dir=None):
            return [
                {"title": "Python", "summary": {"extract": python_text}, "sections": {}},
                {"title": "Java", "summary": {"extract": java_text}, "sections": {}},
            ]

        mock_fetch.side_effect = fake_fetch

        # 2. Ingest
        await ingest(titles=["Python", "Java"], save_dir=tmp_path / "wikipages", chunk_size=500, overlap=0, embedding_client=emb, vectorstore=store)

    # 3. Search
    # Query for "readability" -> should match Python
    vec_python = (await emb.embed_texts(["code readability philosophy"]))[0]
    results_python = await store.search(vec_python, n_results=1)

    assert len(results_python["documents"][0]) > 0
    top_doc_python = results_python["documents"][0][0]
    top_title_python = results_python["metadatas"][0][0]["title"]

    assert "Python" in top_title_python, f"Expected Python, got {top_title_python}. Doc: {top_doc_python}"
    assert "readability" in top_doc_python

    # Query for "object-oriented" or "dependencies" -> should match Java
    vec_java = (await emb.embed_texts(["implementation dependencies"]))[0]
    results_java = await store.search(vec_java, n_results=1)

    top_title_java = results_java["metadatas"][0][0]["title"]
    assert "Java" in top_title_java, f"Expected Java, got {top_title_java}"
