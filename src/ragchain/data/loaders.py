"""Custom document loaders for RAG pipeline."""

import asyncio
import logging

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from ragchain.data.utils import log_with_prefix

logger = logging.getLogger(__name__)


async def load_tiobe_languages(n: int = 50) -> list[str]:
    """Fetch top-n programming languages from TIOBE index.

    Args:
        n: Number of languages to fetch (max: 50, default: 50)

    Returns:
        List of programming language names in TIOBE ranking order.
        Returns empty list if fetch fails.
    """
    url = "https://www.tiobe.com/tiobe-index/"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=ClientTimeout(total=15)) as r:
                r.raise_for_status()
                html = await r.text()
    except Exception as e:
        log_with_prefix(logger, logging.WARNING, "load_tiobe_languages", f"Failed to fetch TIOBE index: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    languages = []

    # Extract from top 20 table
    top20_table = soup.find("table", id="top20")
    if top20_table:
        for row in top20_table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) > 4 and (name := cols[4].get_text(strip=True)):
                languages.append(name)

    # Extract from other languages table (21-50)
    other_table = soup.find("table", id="otherPL")
    if other_table:
        for row in other_table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) > 1 and (name := cols[1].get_text(strip=True)):
                languages.append(name)

    return languages[:n]


def _load_single_page(lang: str, retries: int = 2) -> Document | None:
    """Load Wikipedia page for a programming language with retry logic.

    Args:
        lang: Programming language name (e.g., 'Python')
        retries: Number of retry attempts on failure (default: 2)

    Returns:
        Document with page content and language metadata, or None if loading fails.
    """
    import time

    from langchain_community.document_loaders import WikipediaLoader

    for attempt in range(retries + 1):
        try:
            # Set a shorter timeout for Wikipedia loader to avoid hanging
            loader = WikipediaLoader(query=f"{lang} programming language", load_max_docs=1)
            pages = loader.load()
            if pages:
                pages[0].metadata["language"] = lang
                return pages[0]
        except Exception as e:
            if attempt < retries:
                # Wait before retrying (exponential backoff)
                wait_time = 0.5 * (2**attempt)
                time.sleep(wait_time)
            else:
                log_with_prefix(logger, logging.WARNING, "load_wikipedia_page", f"Failed to load Wikipedia page for {lang} after {retries + 1} attempts: {e}")
    return None


async def load_wikipedia_pages(language_names: list[str]) -> list[Document]:
    """Fetch Wikipedia pages for programming languages sequentially.

    Loads Wikipedia articles for given languages one at a time to avoid
    Wikipedia API rate limiting issues.

    Args:
        language_names: List of programming language names to fetch

    Returns:
        List of Documents with Wikipedia content and language metadata.
        Failed languages are silently skipped.
    """
    docs = []
    loop = asyncio.get_event_loop()

    # Load pages sequentially to avoid Wikipedia API rate limiting
    for lang in language_names:
        try:
            result = await loop.run_in_executor(None, _load_single_page, lang)
            if result:
                docs.append(result)
        except Exception as e:
            log_with_prefix(logger, logging.ERROR, "load_wikipedia_pages", f"Error loading page: {e}")

    return docs
