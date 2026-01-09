"""Custom document loaders for RAG pipeline."""

import asyncio
import logging

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from ragchain.utils import log_with_prefix

logger = logging.getLogger(__name__)

# Define the "Bridge Pages" that provide conceptual glue
CONCEPTUAL_TOPICS = [
    "Programming language",
    "Programming language implementation",  # Crucial for Compiled vs Interpreted
    "Programming paradigm",  # Imperative vs Functional etc
    "Type system",  # Static vs Dynamic
    "Memory management",  # Garbage Collection vs Manual
    "History of programming languages",
    "Compiler",
    "Interpreter (computing)",
    "Standard library",
    "Syntax (programming languages)",
]


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


def _load_topic_page(topic: str, retries: int = 2) -> Document | None:
    """Load a specific Wikipedia topic page without query modification.

    Args:
        topic: Exact Wikipedia page title to search for.
        retries: Number of retry attempts on failure.

    Returns:
        Document with content and category metadata.
    """
    import time

    from langchain_community.document_loaders import WikipediaLoader

    for attempt in range(retries + 1):
        try:
            # Use the exact topic as the query, unlike the language loader
            loader = WikipediaLoader(query=topic, load_max_docs=1)
            pages = loader.load()
            if pages:
                # Tag these as 'concept' so you can filter/weight them differently if needed
                pages[0].metadata["category"] = "concept"
                pages[0].metadata["topic"] = topic
                return pages[0]
        except Exception as e:
            if attempt < retries:
                wait_time = 0.5 * (2**attempt)
                time.sleep(wait_time)
            else:
                log_with_prefix(logger, logging.WARNING, "load_topic_page", f"Failed to load topic {topic}: {e}")
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


async def load_conceptual_pages() -> list[Document]:
    """Fetch the pre-defined list of conceptual/theory pages.

    Returns:
        List of Documents containing computer science theory.
    """
    docs = []
    loop = asyncio.get_event_loop()

    logger.info(f"Loading {len(CONCEPTUAL_TOPICS)} conceptual pages...")

    for topic in CONCEPTUAL_TOPICS:
        try:
            # Run blocking Wikipedia calls in executor to keep async loop alive
            result = await loop.run_in_executor(None, _load_topic_page, topic)
            if result:
                docs.append(result)
                logger.info(f"Successfully loaded concept: {topic}")
        except Exception as e:
            log_with_prefix(logger, logging.ERROR, "load_conceptual_pages", f"Error loading {topic}: {e}")

    return docs
