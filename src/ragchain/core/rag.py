"""RAG pipeline orchestration using LangChain."""

import logging
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragchain.data.config import config
from ragchain.data.utils import log_timing, log_with_prefix

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """Custom ensemble retriever combining BM25 and vector search."""

    bm25_retriever: BM25Retriever
    chroma_retriever: VectorStoreRetriever
    bm25_weight: float = 0.4
    chroma_weight: float = 0.6

    def _parallel_retrieve(self, query: str) -> tuple[list[Document], list[Document]]:
        """Retrieve documents from both retrievers in parallel.

        Args:
            query: The search query.

        Returns:
            Tuple of (bm25_docs, chroma_docs).
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_retriever.invoke, query)
            chroma_future = executor.submit(self.chroma_retriever.invoke, query)
            return bm25_future.result(), chroma_future.result()

    def _compute_rrf_scores(self, bm25_docs: list[Document], chroma_docs: list[Document]) -> list[Document]:
        """Compute Reciprocal Rank Fusion scores and return sorted documents.

        Args:
            bm25_docs: Documents from BM25 retrieval.
            chroma_docs: Documents from Chroma retrieval.

        Returns:
            Documents sorted by RRF score.
        """
        # RRF constant k=60 (standard value that prevents rank 1 from dominating)
        rrf_k = 60
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            rrf_score = self.bm25_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        for rank, doc in enumerate(chroma_docs):
            content = doc.page_content
            rrf_score = self.chroma_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score
            doc_map[content] = doc

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        """Retrieve documents using Reciprocal Rank Fusion (RRF) with parallel execution.

        Fetches BM25 and Chroma results in parallel threads, then combines rankings
        using RRF: score = weight / (rank + 60). This allows documents that appear
        in both rankings to outrank those appearing in only one.

        Args:
            query: The search query.

        Returns:
            List of retrieved documents sorted by RRF score.
        """
        log_with_prefix(logger, logging.DEBUG, "EnsembleRetriever", f"Query: {query[:50]}...")
        start = time.time()

        bm25_docs, chroma_docs = self._parallel_retrieve(query)
        log_timing(logger, "EnsembleRetriever", start, f"Parallel retrieval: BM25={len(bm25_docs)}, Chroma={len(chroma_docs)}")

        sorted_docs = self._compute_rrf_scores(bm25_docs, chroma_docs)
        log_timing(logger, "EnsembleRetriever", start, f"RRF combined {len(sorted_docs)} unique docs")

        return sorted_docs

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Get relevant documents using parallel retrieval (default behavior)."""
        return self._get_relevant_documents(query)


def get_embedder():
    """Create Ollama embedding function.

    Returns OllamaEmbeddings configured with bge-m3 model.
    Uses 1024-dimensional vector embeddings with 8k token context window.

    Returns:
        OllamaEmbeddings instance configured with model and base URL from env vars.
    """
    return OllamaEmbeddings(model=config.ollama_embed_model, base_url=config.ollama_base_url, num_ctx=8192)


def get_vector_store():
    """Get or create Chroma vector store for semantic search.

    Returns either remote Chroma (HTTP) or local persistent Chroma depending on
    CHROMA_SERVER_URL environment variable.

    Returns:
        Chroma instance configured with embedder and collection name.
    """
    embedder = get_embedder()

    if config.chroma_server_url:
        from chromadb import HttpClient

        parsed = urlparse(config.chroma_server_url)
        client = HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            client=client,
        )
    else:
        Path(config.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name="ragchain",
            embedding_function=embedder,
            persist_directory=config.chroma_persist_directory,
        )


async def ingest_documents(docs: list[Document]) -> dict:
    """Process and store documents in vector store.

    Pipeline: Split docs → Embed chunks → Store in Chroma.

    Args:
        docs: List of LangChain Documents to ingest

    Returns:
        dict with status, count, and message
    """
    if not docs:
        return {"status": "ok", "count": 0, "message": "No documents to ingest"}

    start_time = time.perf_counter()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    store = get_vector_store()
    store.add_documents(chunks)

    get_ensemble_retriever.cache_clear()

    elapsed = time.perf_counter() - start_time
    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }


def _load_documents_from_chroma(store: Chroma) -> list[Document]:
    """Load all documents from Chroma vector store.

    Args:
        store: Chroma vector store instance.

    Returns:
        List of Document objects.
    """
    chroma_data = store.get()
    documents = chroma_data.get("documents", [])
    metadatas = chroma_data.get("metadatas", [])

    # Ensure metadatas has same length as documents, fill with empty dicts if needed
    if len(metadatas) < len(documents):
        metadatas.extend([{} for _ in range(len(documents) - len(metadatas))])

    return [Document(page_content=doc, metadata=meta if meta else {}) for doc, meta in zip(documents, metadatas, strict=True)]


def _create_bm25_retriever(docs: list[Document], k: int) -> BM25Retriever:
    """Create BM25 retriever from documents.

    Args:
        docs: List of documents to index.
        k: Number of results to return.

    Returns:
        Configured BM25Retriever instance.
    """
    return BM25Retriever.from_documents(docs, k=k)


def _create_chroma_retriever(store: Chroma, k: int) -> VectorStoreRetriever:
    """Create Chroma retriever from vector store.

    Args:
        store: Chroma vector store instance.
        k: Number of results to return.

    Returns:
        Configured VectorStoreRetriever instance.
    """
    return store.as_retriever(search_kwargs={"k": k})


@lru_cache(maxsize=32)
def get_ensemble_retriever(k: int = 8, bm25_weight: float = 0.4, chroma_weight: float = 0.6) -> EnsembleRetriever:
    """Create an ensemble retriever combining BM25 and Chroma vector search.

    Uses LRU cache to avoid rebuilding the BM25 index on every request.
    Cache is keyed by (k, bm25_weight, chroma_weight).

    Args:
        k: Number of results per retriever
        bm25_weight: Weight for BM25 results in RRF (default: 0.4)
        chroma_weight: Weight for Chroma results in RRF (default: 0.6)

    Returns:
        EnsembleRetriever instance (cached if available)
    """
    log_with_prefix(logger, logging.DEBUG, "get_ensemble_retriever", f"Creating new retriever with k={k}, bm25={bm25_weight}, chroma={chroma_weight}")
    start = time.time()

    store = get_vector_store()
    docs = _load_documents_from_chroma(store)

    log_with_prefix(logger, logging.DEBUG, "get_ensemble_retriever", f"Loaded {len(docs)} documents from Chroma")

    bm25_retriever = _create_bm25_retriever(docs, k)
    log_with_prefix(logger, logging.DEBUG, "get_ensemble_retriever", f"BM25 initialized with k={k} over {len(docs)} docs")

    chroma_retriever = _create_chroma_retriever(store, k)

    retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=bm25_weight,
        chroma_weight=chroma_weight,
    )

    log_timing(logger, "get_ensemble_retriever", start, "Ensemble created")
    return retriever


async def search(query: str, k: int = 8) -> dict:
    """Perform ensemble retrieval using BM25 and Chroma vector search.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: 8)

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


async def judge_answer(question: str, context: str, answer: str, model: str = config.ollama_model) -> dict:
    """Evaluate a RAG answer using LLM-as-judge for correctness, relevance, and faithfulness.

    Args:
        question: The original question
        context: The retrieved context documents as concatenated text
        answer: The generated answer to evaluate
        model: Ollama model to use for judging

    Returns:
        dict with evaluation scores and explanations
    """
    import json

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import OllamaLLM

    from ragchain.prompts import JUDGE_PROMPT

    # Truncate context to avoid very long inference times (keep first ~4000 chars)
    max_context_chars = 4000
    if len(context) > max_context_chars:
        truncated_context = context[:max_context_chars] + "\n\n[...context truncated for evaluation...]"
        log_with_prefix(logger, logging.INFO, "judge_answer", f"Truncated context from {len(context)} to {max_context_chars} chars")
    else:
        truncated_context = context

    llm = OllamaLLM(model=model, base_url=config.ollama_base_url, temperature=0.0)

    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    judge_input = prompt.format(question=question, context=truncated_context, answer=answer)

    log_with_prefix(logger, logging.INFO, "judge_answer", f"Judging answer for question: {question[:50]}...")

    start = time.time()
    raw_response = llm.invoke(judge_input)
    log_timing(logger, "judge_answer", start, "LLM judgment completed")

    # Parse JSON response - extract JSON from potential markdown or text wrapper
    def extract_json_object(text: str) -> dict | None:
        """Extract JSON object handling nested braces properly."""
        # Find the first { and match to the corresponding }
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if brace_count != 0:
            return None

        json_str = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    # Try direct parsing first
    try:
        evaluation = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        # Try to extract JSON object with proper brace matching
        evaluation = extract_json_object(raw_response)
        if evaluation and "correctness" in evaluation:
            log_with_prefix(logger, logging.INFO, "judge_answer", "Successfully extracted JSON from wrapped response")
        else:
            # Log the raw response for debugging
            log_with_prefix(logger, logging.ERROR, "judge_answer", f"Failed to parse judge response. Raw response: {raw_response[:500]}")
            return {
                "correctness": {"score": 0, "explanation": "Failed to parse response"},
                "relevance": {"score": 0, "explanation": "Failed to parse response"},
                "faithfulness": {"score": 0, "explanation": "Failed to parse response"},
            }

    # Validate and fix scores - ensure they're in 1-5 range
    for criterion in ["correctness", "relevance", "faithfulness"]:
        if criterion in evaluation and isinstance(evaluation[criterion], dict):
            score = evaluation[criterion].get("score", 0)
            if not isinstance(score, int) or score < 1 or score > 5:
                log_with_prefix(logger, logging.WARNING, "judge_answer", f"Invalid {criterion} score {score}, marking as parse error")
                evaluation[criterion]["score"] = 0
                evaluation[criterion]["explanation"] = f"Invalid score: {score}. " + evaluation[criterion].get("explanation", "")

    return evaluation
