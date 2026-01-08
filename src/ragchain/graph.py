"""LangGraph implementation for intent-based adaptive RAG."""

import logging
import time

from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph

from ragchain.config import config
from ragchain.prompts import QUERY_REWRITER_PROMPT
from ragchain.retrieval.grader import grade_with_statistics, should_accept_docs, should_skip_grading
from ragchain.retrieval.retrievers import get_ensemble_retriever
from ragchain.retrieval.router import intent_router
from ragchain.types import GradeSignal, Intent, IntentRoutingState, Node

logger = logging.getLogger(__name__)

__all__ = ["rag_graph"]


def adaptive_retriever(state: IntentRoutingState) -> IntentRoutingState:
    """Retrieve with intent-specific weights using parallel execution."""
    start = time.time()
    logger.info(f"[adaptive_retriever] Starting for intent: {state['intent']}")

    query = state.get("rewritten_query") or state["query"]

    weights = {
        Intent.FACT: (0.8, 0.2),  # Keyword-heavy for lists/rankings
        Intent.CONCEPT: (0.4, 0.6),  # Semantic-heavy for natural questions
        Intent.COMPARISON: (0.5, 0.5),  # Semantic-leaning for comparing entities
    }
    bm25_weight, chroma_weight = weights.get(state["intent"], (0.5, 0.5))
    logger.info(f"[adaptive_retriever] Using weights: BM25={bm25_weight}, Chroma={chroma_weight}")

    try:
        # Reduce from 12 to 6 to fit context window constraints
        retriever = get_ensemble_retriever(k=6, bm25_weight=bm25_weight, chroma_weight=chroma_weight)
        docs = retriever.get_relevant_documents(query)
        logger.info(f"[adaptive_retriever] Retrieved {len(docs)} documents in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"[adaptive_retriever] Error during retrieval: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}


def retrieval_grader(state: IntentRoutingState) -> IntentRoutingState:
    """Grade if retrieved docs answer the query."""
    start = time.time()
    logger.info(f"[retrieval_grader] Starting with {len(state['retrieved_docs'])} documents")

    # Skip grading if disabled (fast-path)
    if should_skip_grading():
        logger.info("[retrieval_grader] Grading disabled, auto-accepting docs")
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Auto-accept if no docs or already retried
    if should_accept_docs(state["retrieved_docs"], state.get("retry_count", 0)):
        reason = "No documents to grade" if not state["retrieved_docs"] else "Already retried once"
        logger.info(f"[retrieval_grader] {reason}, accepting docs to avoid infinite loop")
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Grade with LLM
    grade_value = grade_with_statistics(state["query"], state["retrieved_docs"])
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
    return state["retrieval_grade"] == GradeSignal.NO and state.get("retry_count", 0) < 1


# Build the graph
workflow = StateGraph(IntentRoutingState)

# Add nodes
workflow.add_node(Node.INTENT_ROUTER, intent_router)
workflow.add_node(Node.ADAPTIVE_RETRIEVER, adaptive_retriever)
workflow.add_node(Node.RETRIEVAL_GRADER, retrieval_grader)
workflow.add_node(Node.QUERY_REWRITER, query_rewriter)

# Set entry point
workflow.set_entry_point(Node.INTENT_ROUTER)

# Add edges
workflow.add_edge(Node.INTENT_ROUTER, Node.ADAPTIVE_RETRIEVER)
workflow.add_edge(Node.ADAPTIVE_RETRIEVER, Node.RETRIEVAL_GRADER)


# Conditional edge: if grade is YES or max retries reached, end; otherwise rewrite and retry
def should_rewrite(state: IntentRoutingState) -> str:
    """Determine if we should continue retrying or end."""
    # If retrieval passed, we're done
    if state["retrieval_grade"] == GradeSignal.YES:
        logger.info("[graph] Retrieval passed, ending")
        return END
    # If we've already retried once, accept the current docs and end
    if state.get("retry_count", 0) >= 1:
        logger.info(f"[graph] Max retries reached ({state.get('retry_count', 0)}), ending")
        return END
    # Otherwise, try rewriting
    logger.info("[graph] Retrieval failed, will rewrite query")
    return Node.QUERY_REWRITER


workflow.add_conditional_edges(
    Node.RETRIEVAL_GRADER,
    should_rewrite,
    {END: END, Node.QUERY_REWRITER: Node.QUERY_REWRITER},
)

# After rewrite, retrieve again, then grade again
workflow.add_edge(Node.QUERY_REWRITER, Node.ADAPTIVE_RETRIEVER)

# Compile the graph
rag_graph = workflow.compile()
