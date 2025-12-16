from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from ragchain.utils import safe_filename

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_EXTRACT = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={title}&explaintext=1&redirects=1"


async def fetch_page(title: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Fetch summary and mobile sections for a Wikipedia page.

    Returns the combined JSON results.
    """
    summary_url = WIKI_SUMMARY.format(title=title)
    # We use the MediaWiki 'extracts' API to get plain-text extracts for a
    # page and then synthesize a small 'sections' shape compatible with the
    # downstream HTML parser. The historical mobile-sections REST endpoint
    # has been removed in some deployments, so we avoid relying on it entirely.

    async with session.get(summary_url) as r:
        r.raise_for_status()
        summary = await r.json()

    extract_url = WIKI_EXTRACT.format(title=title)
    async with session.get(extract_url) as r2:
        r2.raise_for_status()
        data = await r2.json()
        pages = data.get("query", {}).get("pages", {})
        # Take the first page entry (the dict keys are pageids)
        page = next(iter(pages.values())) if pages else {}
        extract_text = page.get("extract", "") or ""
        sections = {"sections": [{"line": "extract", "text": f"<p>{extract_text}</p>"}]}

    return {"title": title, "summary": summary, "sections": sections}


async def fetch_wikipedia_pages(titles: List[str], concurrency: int = 5, save_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch multiple Wikipedia pages concurrently and optionally save raw JSON to disk.

    This uses the REST API for metadata and sections. It will respect a simple
    concurrency limit.
    """
    results: List[Dict[str, Any]] = []
    # Limit concurrent outbound HTTP requests using a semaphore.
    # This keeps memory/connection usage bounded when fetching many pages.
    sem = asyncio.Semaphore(concurrency)

    async def _fetch(title: str):
        # Acquire the semaphore before creating a session and requesting the page.
        # Each task will perform a short-lived ClientSession to avoid sharing state.
        async with sem:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {"User-Agent": "ragchain/0.1 (github.com)"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                return await fetch_page(title, session)

    # Start tasks for all requested titles; tasks execute concurrently but are bound
    # by the semaphore above.
    coros = [asyncio.create_task(_fetch(t)) for t in titles]

    # Process tasks as they finish (as_completed) so slower pages don't block faster ones.
    for c in asyncio.as_completed(coros):
        data = await c
        # Optionally persist raw JSON to disk. This writes to a temporary file
        # and atomically replaces the final file to avoid partial writes.
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(data.get("title", "page")) + ".json"
            tmp_path = save_dir / (filename + ".tmp")
            final_path = save_dir / filename
            tmp_path.write_text(json.dumps(data, ensure_ascii=False))
            tmp_path.replace(final_path)
        # Append the fetched page to results after (optionally) persisting it.
        results.append(data)

    return results
