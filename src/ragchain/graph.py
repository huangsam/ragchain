"""LangGraph implementation for intent-based adaptive RAG."""

import logging
import time
from typing import List, Literal

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ragchain.config import config
from ragchain.rag import get_ensemble_retriever
from ragchain.router import (
    INTENT_ROUTER_PROMPT,
    QUERY_REWRITER_PROMPT,
    RETRIEVAL_GRADER_PROMPT,
)
from ragchain.utils import log_timing, log_with_prefix

logger = logging.getLogger(__name__)

__all__ = ["IntentRoutingState", "rag_graph"]


class IntentRoutingState(TypedDict):
    """State for the intent routing RAG graph."""

    query: str
    original_query: str  # Preserve original query for rewriting
    intent: Literal["FACT", "CONCEPT", "COMPARISON"]
    retrieved_docs: list[Document]
    retrieval_grade: Literal["YES", "NO"]
    rewritten_query: str
    retry_count: int


def _is_simple_query(query: str) -> bool:
    """Fast heuristic to detect simple queries that can skip intent routing."""
    query_lower = query.lower()
    simple_patterns = ["what is", "define", "explain", "who is", "when was", "where is", "how does", "why is"]
    return any(pattern in query_lower for pattern in simple_patterns) and len(query.split()) <= 8


def intent_router(state: IntentRoutingState) -> IntentRoutingState:
    """Route query to intent category."""
    start = time.time()
    log_with_prefix(logger, logging.INFO, "intent_router", f"Starting for query: {state['query'][:50]}...")

    # Fast-path: Skip LLM for simple queries if routing is disabled
    if not config.enable_intent_routing or _is_simple_query(state["query"]):
        log_with_prefix(logger, logging.INFO, "intent_router", "Using fast-path, defaulting to CONCEPT")
        return {**state, "intent": "CONCEPT", "original_query": state["query"]}

    llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url, temperature=0)

    prompt = INTENT_ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt).strip().upper()

    # Extract first valid intent
    valid_intents: list[Literal["FACT", "CONCEPT", "COMPARISON"]] = ["FACT", "CONCEPT", "COMPARISON"]
    intent_value: Literal["FACT", "CONCEPT", "COMPARISON"] = next((i for i in valid_intents if i in response), "CONCEPT")

    log_timing(logger, "intent_router", start, f"Classified as {intent_value}")

    return {**state, "intent": intent_value, "original_query": state["query"]}


def adaptive_retriever(state: IntentRoutingState) -> IntentRoutingState:
    """Retrieve with intent-specific weights using parallel execution."""
    start = time.time()
    logger.info(f"[adaptive_retriever] Starting for intent: {state['intent']}")

    query = state.get("rewritten_query") or state["query"]

    weights = {
        "FACT": (0.7, 0.3),  # Keyword-heavy for lists/rankings
        "CONCEPT": (0.3, 0.7),  # Semantic-heavy for natural questions
        "COMPARISON": (0.4, 0.6),  # Semantic-leaning for comparing entities
    }
    bm25_weight, chroma_weight = weights.get(state["intent"], (0.5, 0.5))
    logger.info(f"[adaptive_retriever] Using weights: BM25={bm25_weight}, Chroma={chroma_weight}")

    try:
        retriever = get_ensemble_retriever(k=8, bm25_weight=bm25_weight, chroma_weight=chroma_weight)
        docs = retriever.get_relevant_documents(query)
        logger.info(f"[adaptive_retriever] Retrieved {len(docs)} documents in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"[adaptive_retriever] Error during retrieval: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}


def _should_skip_grading(state: IntentRoutingState) -> bool:
    """Determine if grading should be skipped.

    Args:
        state: Current graph state.

    Returns:
        True if grading should be skipped.
    """
    return not config.enable_grading


def _should_accept_docs(state: IntentRoutingState) -> bool:
    """Determine if documents should be auto-accepted.

    Args:
        state: Current graph state.

    Returns:
        True if docs should be accepted without grading.
    """
    return not state["retrieved_docs"] or state.get("retry_count", 0) > 0


def _grade_with_llm(query: str, docs: List[Document]) -> Literal["YES", "NO"]:
    """Grade document relevance using LLM.

    Args:
        query: The search query.
        docs: Retrieved documents to grade.

    Returns:
        "YES" if relevant, "NO" if not.
    """
    llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url, temperature=0)

    formatted_docs = "\n\n".join([f"Doc {i}: {doc.page_content[:200]}" for i, doc in enumerate(docs)])
    prompt = RETRIEVAL_GRADER_PROMPT.format(query=query, formatted_docs=formatted_docs)
    response = llm.invoke(prompt).strip().upper()

    # More lenient grading: only reject if explicitly negative
    negative_indicators = ["NO", "NOT RELEVANT", "INSUFFICIENT", "IRRELEVANT", "DOES NOT"]
    is_negative = any(indicator in response for indicator in negative_indicators)

    return "NO" if is_negative else "YES"


def retrieval_grader(state: IntentRoutingState) -> IntentRoutingState:
    """Grade if retrieved docs answer the query."""
    start = time.time()
    logger.info(f"[retrieval_grader] Starting with {len(state['retrieved_docs'])} documents")

    # Skip grading if disabled (fast-path)
    if _should_skip_grading(state):
        logger.info("[retrieval_grader] Grading disabled, auto-accepting docs")
        return {**state, "retrieval_grade": "YES"}

    # Auto-accept if no docs or already retried
    if _should_accept_docs(state):
        reason = "No documents to grade" if not state["retrieved_docs"] else "Already retried once"
        logger.info(f"[retrieval_grader] {reason}, accepting docs to avoid infinite loop")
        return {**state, "retrieval_grade": "YES"}

    # Grade with LLM
    grade_value = _grade_with_llm(state["query"], state["retrieved_docs"])
    logger.info(f"[retrieval_grader] Grade: {grade_value} in {time.time() - start:.2f}s")

    return {**state, "retrieval_grade": grade_value}


def query_rewriter(state: IntentRoutingState) -> IntentRoutingState:
    """Rewrite query for better retrieval."""
    start = time.time()
    logger.info(f"[query_rewriter] Rewriting query (attempt {state.get('retry_count', 0) + 1})")

    llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url, temperature=0.5)

    # Always rewrite from the original query
    original = state.get("original_query", state["query"])
    prompt = QUERY_REWRITER_PROMPT.format(query=original)
    rewritten = llm.invoke(prompt).strip()

    logger.info(f"[query_rewriter] Original query: {original}")
    logger.info(f"[query_rewriter] Rewritten query: {rewritten}")
    logger.info(f"[query_rewriter] Completed in {time.time() - start:.2f}s")

    return {**state, "rewritten_query": rewritten, "retry_count": state.get("retry_count", 0) + 1}


def should_retry(state: IntentRoutingState) -> bool:
    """Decide if we should retry retrieval."""
    return state["retrieval_grade"] == "NO" and state.get("retry_count", 0) < 1


# Build the graph
workflow = StateGraph(IntentRoutingState)

# Add nodes
workflow.add_node("intent_router", intent_router)
workflow.add_node("adaptive_retriever", adaptive_retriever)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("query_rewriter", query_rewriter)

# Set entry point
workflow.set_entry_point("intent_router")

# Add edges
workflow.add_edge("intent_router", "adaptive_retriever")
workflow.add_edge("adaptive_retriever", "retrieval_grader")


# Conditional edge: if grade is YES or max retries reached, end; otherwise rewrite and retry
def should_rewrite(state: IntentRoutingState) -> str:
    """Determine if we should continue retrying or end."""
    # If retrieval passed, we're done
    if state["retrieval_grade"] == "YES":
        logger.info("[graph] Retrieval passed, ending")
        return "END"
    # If we've already retried once, accept the current docs and end
    if state.get("retry_count", 0) >= 1:
        logger.info(f"[graph] Max retries reached ({state.get('retry_count', 0)}), ending")
        return "END"
    # Otherwise, try rewriting
    logger.info("[graph] Retrieval failed, will rewrite query")
    return "query_rewriter"


workflow.add_conditional_edges(
    "retrieval_grader",
    should_rewrite,
    {"END": END, "query_rewriter": "query_rewriter"},
)

# After rewrite, retrieve again, then grade again
workflow.add_edge("query_rewriter", "adaptive_retriever")

# Compile the graph
rag_graph = workflow.compile()
