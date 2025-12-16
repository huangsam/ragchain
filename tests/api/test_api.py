import os

import pytest
from fastapi.testclient import TestClient

from ragchain.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.parametrize("titles", [["Hello_World"], ["Python_(programming_language)"]])
def test_ingest_and_search(tmp_path, titles):
    # Set a temporary persist dir for in-process chroma
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(tmp_path / "chroma")

    # Mock Wikipedia API endpoints used by ingest
    from aioresponses import aioresponses

    for title in titles:
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        sections_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"

        summary_json = {"title": title, "extract": "This is the summary paragraph."}
        sections_json = {
            "sections": [
                {"line": "Section 1", "text": "<p>This is the content of section 1.</p>"},
                {"line": "Section 2", "text": "<p>This is the content of section 2.</p>"},
            ]
        }

        with aioresponses() as m:
            m.get(summary_url, payload=summary_json)
            m.get(sections_url, payload=sections_json)

            r = client.post("/ingest", json={"titles": [title]})
            assert r.status_code == 200
            data = r.json()
            assert "report" in data

    # search should return a result structure (may be empty if vectors not present)
    r = client.post("/search", json={"query": "summary paragraph", "n_results": 1})
    assert r.status_code == 200
    assert "results" in r.json()
