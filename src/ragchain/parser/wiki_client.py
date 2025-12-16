from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from ragchain.utils import safe_filename


WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_MOBILE_SECTIONS = "https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"


async def fetch_page(title: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Fetch summary and mobile sections for a Wikipedia page.

    Returns the combined JSON results.
    """
    summary_url = WIKI_SUMMARY.format(title=title)
    sections_url = WIKI_MOBILE_SECTIONS.format(title=title)

    async with session.get(summary_url) as r:
        r.raise_for_status()
        summary = await r.json()

    async with session.get(sections_url) as r:
        r.raise_for_status()
        sections = await r.json()

    return {"title": title, "summary": summary, "sections": sections}


async def fetch_wikipedia_pages(
    titles: List[str], concurrency: int = 5, save_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Fetch multiple Wikipedia pages concurrently and optionally save raw JSON to disk.

    This uses the REST API for metadata and sections. It will respect a simple
    concurrency limit.
    """
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(concurrency)

    async def _fetch(title: str):
        async with sem:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {"User-Agent": "ragchain/0.1 (github.com)"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                return await fetch_page(title, session)

    coros = [asyncio.create_task(_fetch(t)) for t in titles]

    for c in asyncio.as_completed(coros):
        data = await c
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(data.get("title", "page")) + ".json"
            tmp_path = save_dir / (filename + ".tmp")
            final_path = save_dir / filename
            tmp_path.write_text(json.dumps(data, ensure_ascii=False))
            tmp_path.replace(final_path)
        results.append(data)

    return results
