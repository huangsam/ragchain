"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from ragchain.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_request_structure(client):
    """Test ingest endpoint accepts correct request format."""
    # This will fail due to no vector store initialized, but we're checking request handling
    response = client.post("/ingest", json={"languages": ["Python"], "n_languages": 1})
    # Should either work or return 500, not 422 (validation error)
    assert response.status_code in [200, 500]


def test_search_request_structure(client):
    """Test search endpoint accepts correct request format."""
    response = client.post("/search", json={"query": "test", "k": 4})
    assert response.status_code in [200, 500]
