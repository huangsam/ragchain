"""RAG pipeline orchestration using LangChain."""

from ragchain.core.retrievers import get_ensemble_retriever
from ragchain.core.types import SearchResult


async def search(query: str, k: int = 12) -> SearchResult:
    """Perform ensemble retrieval using BM25 and Chroma vector search.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: 12)

    Returns:
        dict with 'query' and 'results' list of {content, metadata, distance}
    """
    ensemble_retriever = get_ensemble_retriever(k)

    results = ensemble_retriever.get_relevant_documents(query)

    results = results[:k]

    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }
