"""RAG pipeline orchestration using LangChain."""

from ragchain.config import config
from ragchain.inference.retrievers import get_ensemble_retriever
from ragchain.types import SearchResult


async def search(query: str, k: int | None = None) -> SearchResult:
    """Perform ensemble retrieval using BM25 and Chroma vector search.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: from config.search_k)

    Returns:
        dict with 'query' and 'results' list of {content, metadata, distance}
    """
    if k is None:
        k = config.search_k
    ensemble_retriever = get_ensemble_retriever(k)

    results = ensemble_retriever.invoke(query)

    results = results[:k]

    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }
