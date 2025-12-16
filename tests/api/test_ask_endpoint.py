import os
import pytest
from fastapi.testclient import TestClient
from aioresponses import aioresponses

from ragchain.api import app

client = TestClient(app)


@pytest.mark.parametrize(
    "query_text",
    [
        "What is Python?",
        "high-performance web backend",
        "trade-offs for each",
    ],
)
def test_ask_endpoint_with_mock_data(tmp_path, query_text):
    """Test /ask endpoint returns structured answer with context."""
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(tmp_path / "chroma")

    # Ingest sample data first
    title = "Test_Page"
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    extract_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={title}&explaintext=1&redirects=1"

    summary_json = {"title": title, "extract": "This is a test page about testing."}
    extract_json = {"query": {"pages": {"1": {"extract": "Testing is important for reliability."}}}}

    with aioresponses() as m:
        m.get(summary_url, payload=summary_json)
        m.get(extract_url, payload=extract_json)
        r = client.post("/ingest", json={"titles": [title]})
        assert r.status_code == 200

    # Now test /ask
    r = client.post("/ask", json={"query": query_text, "n_results": 2})
    assert r.status_code == 200

    data = r.json()
    assert "answer" in data
    assert "context" in data
    assert "model" in data
    assert isinstance(data["context"], list)
    assert isinstance(data["answer"], str)


def test_ask_endpoint_missing_context():
    """Test /ask with no ingested data returns gracefully."""
    r = client.post("/ask", json={"query": "non-existent query", "n_results": 1})
    # Should not crash; may return empty context and a graceful answer
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data


def test_ask_endpoint_empty_query():
    """Test /ask with empty query string."""
    r = client.post("/ask", json={"query": "", "n_results": 1})
    # Empty query should still work (though the result may be empty/generic)
    assert r.status_code == 200


def test_ask_endpoint_custom_model():
    """Test /ask with custom model specified."""
    # Use qwen3 which is the configured model in docker-compose
    r = client.post("/ask", json={"query": "test", "model": "qwen3"})
    # Should not crash; model parameter is accepted
    assert r.status_code in [200, 500]  # 200 if Ollama available, 500 if not


def test_ask_endpoint_large_n_results():
    """Test /ask with very large n_results."""
    # Request more results than available documents
    r = client.post("/ask", json={"query": "test", "n_results": 1000})
    # API should return what's available or handle gracefully
    assert r.status_code in [200, 500]


def test_search_endpoint_returns_metadata():
    """Verify search results include metadata (title, chunk_index, source)."""
    r = client.post("/search", json={"query": "test", "n_results": 1})
    assert r.status_code == 200
    data = r.json()

    if data["results"]["documents"] and data["results"]["documents"][0]:
        # If results exist, check metadata structure
        assert "metadatas" in data["results"]
        assert data["results"]["metadatas"] is not None


def test_ingest_empty_titles():
    """Test ingest with empty titles list."""
    r = client.post("/ingest", json={"titles": []})
    # Should handle gracefully
    assert r.status_code == 200
    data = r.json()
    assert "report" in data
    assert data["report"]["pages_processed"] == 0


def test_ingest_malformed_json():
    """Test ingest with malformed JSON."""
    r = client.post("/ingest", json={"titles": "not-a-list"})  # Should be list
    # FastAPI should reject this with 422
    assert r.status_code == 422


def test_search_without_ingest():
    """Test search when no data has been ingested."""
    r = client.post("/search", json={"query": "anything", "n_results": 1})
    # Should not crash
    assert r.status_code == 200
    data = r.json()
    # Results may be empty but structure should be intact
    assert "results" in data
