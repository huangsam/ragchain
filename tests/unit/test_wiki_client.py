import pytest
from aioresponses import aioresponses

from ragchain.parser.wiki_client import fetch_wikipedia_pages


@pytest.mark.asyncio
async def test_fetch_wikipedia_pages_writes_tmp(tmp_path):
    title = "Python_(programming_language)"
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    sections_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"

    summary_json = {"title": title, "extract": "Python is..."}
    sections_json = {"sections": [{"line": "Intro", "text": "<p>Python is a language</p>"}]}

    with aioresponses() as m:
        m.get(summary_url, payload=summary_json)
        m.get(sections_url, payload=sections_json)

        results = await fetch_wikipedia_pages([title], concurrency=1, save_dir=tmp_path)
        assert len(results) == 1
        from ragchain.utils import safe_filename

        saved = tmp_path / (safe_filename(results[0]["title"]) + ".json")
        # file should exist
        assert saved.exists()
        data = results[0]
        assert data["summary"]["extract"] == "Python is..."


@pytest.mark.asyncio
async def test_fetch_wikipedia_pages_errors_are_propagated():
    title = "Missingpage"
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    with aioresponses() as m:
        m.get(summary_url, status=404)
        with pytest.raises(Exception):
            await fetch_wikipedia_pages([title], concurrency=1)
