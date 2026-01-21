"""LangGraph implementation for intent-based adaptive RAG."""

import logging

from langgraph.graph import END, StateGraph

from ragchain.config import config
from ragchain.inference.grader import grade_with_statistics, should_accept_docs, should_skip_grading
from ragchain.inference.retrievers import get_ensemble_retriever
from ragchain.inference.router import intent_router
from ragchain.prompts import QUERY_REWRITER_PROMPT
from ragchain.types import GradeSignal, Intent, IntentRoutingState, Node
from ragchain.utils import get_llm, timed

logger = logging.getLogger(__name__)

__all__ = ["rag_graph"]


@timed(logger, "adaptive_retriever")
def adaptive_retriever(state: IntentRoutingState) -> IntentRoutingState:
    """Retrieve with intent-specific weights using parallel execution."""

    query = state.get("rewritten_query") or state["query"]

    weights = {
        Intent.FACT: (0.8, 0.2),  # Keyword-heavy for lists/rankings
        Intent.CONCEPT: (0.4, 0.6),  # Semantic-heavy for natural questions
        Intent.COMPARISON: (0.5, 0.5),  # Semantic-leaning for comparing entities
    }
    bm25_weight, chroma_weight = weights.get(state["intent"], (0.5, 0.5))

    try:
        # Use smaller k for adaptive retrieval to fit context window constraints
        retriever = get_ensemble_retriever(config.graph_k, bm25_weight=bm25_weight, chroma_weight=chroma_weight)
        docs = retriever.invoke(query)
        logger.debug(f"[adaptive_retriever] Retrieved {len(docs)} documents for {state['intent'].value}")
    except Exception as e:
        logger.error(f"[adaptive_retriever] Error during retrieval: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}


@timed(logger, "retrieval_grader")
def retrieval_grader(state: IntentRoutingState) -> IntentRoutingState:
    """Grade if retrieved docs answer the query."""

    # Skip grading if disabled (fast-path)
    if should_skip_grading():
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Auto-accept if no docs or already retried
    if should_accept_docs(state["retrieved_docs"], state.get("retry_count", 0)):
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Grade with LLM
    grade_value = grade_with_statistics(state["query"], state["retrieved_docs"])
    logger.debug(f"[retrieval_grader] Grade: {grade_value} ({len(state['retrieved_docs'])} docs)")

    return {**state, "retrieval_grade": grade_value}


@timed(logger, "query_rewriter")
def query_rewriter(state: IntentRoutingState) -> IntentRoutingState:
    """Rewrite query for better retrieval."""

    llm = get_llm(purpose="rewriting")

    # Always rewrite from the original query
    original = state["original_query"]
    prompt = QUERY_REWRITER_PROMPT.format(query=original)
    rewritten = llm.invoke(prompt).strip()

    logger.debug(f"[query_rewriter] Rewrite attempt {state.get('retry_count', 0) + 1} completed")

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
        return END
    # If we've already retried once, accept the current docs and end
    if state.get("retry_count", 0) >= 1:
        return END
    # Otherwise, try rewriting
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
