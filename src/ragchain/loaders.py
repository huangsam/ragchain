"""Custom document loaders for RAG pipeline."""
from typing import List

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document


async def load_tiobe_languages(n: int = 50) -> List[str]:
    """Fetch top-n programming languages from TIOBE index."""
    url = "https://www.tiobe.com/tiobe-index/"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as r:
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


async def load_wikipedia_pages(language_names: List[str]) -> List[Document]:
    """Fetch Wikipedia pages for given programming languages using LangChain's WikipediaLoader."""
    from langchain_community.document_loaders import WikipediaLoader

    docs = []
    for lang in language_names:
        try:
            loader = WikipediaLoader(query=f"{lang} programming language", load_max_docs=1)
            pages = loader.load()
            for page in pages:
                page.metadata["language"] = lang
                docs.append(page)
        except Exception as e:
            print(f"Warning: failed to load {lang}: {e}")
            continue

    return docs
