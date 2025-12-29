"""Shared schemas and enums for the RAG pipeline."""

from enum import Enum

from langchain_core.documents import Document
from typing_extensions import TypedDict

__all__ = ["Intent", "GradeSignal", "Node", "IntentRoutingState"]


class Intent(str, Enum):
    """Query intent classification."""

    FACT = "FACT"
    CONCEPT = "CONCEPT"
    COMPARISON = "COMPARISON"


class GradeSignal(str, Enum):
    """Relevance grading signal for retrieved documents."""

    YES = "YES"
    NO = "NO"


class Node(str, Enum):
    """Graph node names."""

    INTENT_ROUTER = "intent_router"
    ADAPTIVE_RETRIEVER = "adaptive_retriever"
    RETRIEVAL_GRADER = "retrieval_grader"
    QUERY_REWRITER = "query_rewriter"


class IntentRoutingState(TypedDict):
    """State for the intent routing RAG graph."""

    query: str
    original_query: str  # Preserve original query for rewriting
    intent: Intent
    retrieved_docs: list[Document]
    retrieval_grade: GradeSignal
    rewritten_query: str
    retry_count: int
