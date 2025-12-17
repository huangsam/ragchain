"""Custom document loaders for RAG pipeline."""

import asyncio
from typing import List

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from langchain_core.documents import Document


async def load_tiobe_languages(n: int = 50) -> List[str]:
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
                html = await r.text()
    except Exception as e:
        print(f"Warning: failed to fetch TIOBE: {e}")
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


def _load_single_page(lang: str) -> Document | None:
    """Load Wikipedia page for a programming language.

    Args:
        lang: Programming language name (e.g., 'Python')

    Returns:
        Document with page content and language metadata, or None if loading fails.
    """
    from langchain_community.document_loaders import WikipediaLoader

    try:
        # Set a shorter timeout for Wikipedia loader to avoid hanging
        loader = WikipediaLoader(query=f"{lang} programming language", load_max_docs=1)
        pages = loader.load()
        if pages:
            pages[0].metadata["language"] = lang
            return pages[0]
    except Exception as e:
        print(f"Warning: failed to load {lang}: {e}")
    return None


async def load_wikipedia_pages(language_names: List[str]) -> List[Document]:
    """Fetch Wikipedia pages for programming languages concurrently.

    Loads Wikipedia articles for given languages using concurrent ThreadPoolExecutor.

    Args:
        language_names: List of programming language names to fetch

    Returns:
        List of Documents with Wikipedia content and language metadata.
        Failed languages are silently skipped.
    """
    docs = []
    loop = asyncio.get_event_loop()

    # Load pages with concurrent futures to avoid blocking
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [loop.run_in_executor(executor, _load_single_page, lang) for lang in language_names]
        results = await asyncio.gather(*futures, return_exceptions=True)

        for result in results:
            if isinstance(result, Document):
                docs.append(result)
            elif isinstance(result, Exception):
                print(f"Error loading page: {result}")

    return docs
