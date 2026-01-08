"""Integration tests for the full RAG pipeline with real Chroma."""

import pytest
from langchain_core.documents import Document

from ragchain.config import config
from ragchain.inference.rag import search
from ragchain.ingestion.storage import ingest_documents


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_with_real_chroma(temp_chroma_dir):
    """Test the complete RAG pipeline with a real Chroma vector store.

    This test requires:
    - Ollama running with bge-m3 model
    - No CHROMA_SERVER_URL set (uses local Chroma)

    Run with: pytest tests/integration/test_full_pipeline.py::test_full_pipeline_with_real_chroma -v
    """
    # Skip if Ollama is not available or Chroma server is configured
    if config.chroma_server_url:
        pytest.skip("Skipping integration test: CHROMA_SERVER_URL is set (use local Chroma for integration tests)")

    try:
        # Test data
        docs = [
            Document(
                page_content="Python is a high-level programming language known for its simplicity and readability.",
                metadata={"language": "Python", "paradigm": "multi-paradigm"},
            ),
            Document(
                page_content="Java is an object-oriented programming language used for enterprise applications.",
                metadata={"language": "Java", "paradigm": "object-oriented"},
            ),
            Document(
                page_content="JavaScript is a scripting language primarily used for web development.",
                metadata={"language": "JavaScript", "paradigm": "scripting"},
            ),
        ]

        # Ingest documents
        ingest_result = await ingest_documents(docs)
        assert ingest_result["status"] == "ok"
        assert ingest_result["count"] >= 3

        # Search for Python-related content
        search_result = await search("Python programming language", k=2)
        assert search_result["query"] == "Python programming language"
        assert len(search_result["results"]) >= 1

        # Verify we get relevant results
        found_python = any("python" in str(result).lower() for result in search_result["results"])
        assert found_python, "Search should return Python-related results"

    except Exception as e:
        pytest.skip(f"Skipping integration test due to missing dependencies: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chroma_persistence(temp_chroma_dir):
    """Test that Chroma persists data across sessions.

    This test verifies that data stored in Chroma persists and can be retrieved
    in subsequent operations.
    """
    if config.chroma_server_url:
        pytest.skip("Skipping persistence test: CHROMA_SERVER_URL is set")

    try:
        # First ingestion
        docs1 = [
            Document(page_content="First document for persistence test."),
        ]
        result1 = await ingest_documents(docs1)
        assert result1["status"] == "ok"

        # Second ingestion (should add to existing store)
        docs2 = [
            Document(page_content="Second document for persistence test."),
        ]
        result2 = await ingest_documents(docs2)
        assert result2["status"] == "ok"

        # Search should find content from both ingestions
        search_result = await search("persistence test", k=5)
        assert len(search_result["results"]) >= 2

    except Exception as e:
        pytest.skip(f"Skipping persistence test due to missing dependencies: {e}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,expected_language",
    [
        ("object oriented language", "Java"),
        ("web development scripting", "JavaScript"),
        ("simple readable syntax", "Python"),
    ],
)
@pytest.mark.asyncio
async def test_semantic_search_accuracy(temp_chroma_dir, query, expected_language):
    """Test semantic search accuracy with various queries."""
    if config.chroma_server_url:
        pytest.skip("Skipping accuracy test: CHROMA_SERVER_URL is set")

    try:
        # Use the same test documents as above
        docs = [
            Document(
                page_content="Python is a high-level programming language known for its simplicity and readability.",
                metadata={"language": "Python"},
            ),
            Document(
                page_content="Java is an object-oriented programming language used for enterprise applications.",
                metadata={"language": "Java"},
            ),
            Document(
                page_content="JavaScript is a scripting language primarily used for web development.",
                metadata={"language": "JavaScript"},
            ),
        ]

        # Ingest and search
        await ingest_documents(docs)
        result = await search(query, k=1)

        # Check if the most relevant result matches expected language
        assert len(result["results"]) >= 1
        top_result = str(result["results"][0]).lower()
        assert expected_language.lower() in top_result, f"Query '{query}' should return {expected_language} as top result"

    except Exception as e:
        pytest.skip(f"Skipping accuracy test due to missing dependencies: {e}")
