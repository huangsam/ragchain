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


def test_search_request_structure(client):
    """Test search endpoint accepts correct request format."""
    response = client.post("/search", json={"query": "test", "k": 4})
    assert response.status_code in [200, 500]
