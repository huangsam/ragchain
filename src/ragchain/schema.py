"""Shared schemas and enums for the RAG pipeline."""

from enum import Enum

from langchain_core.documents import Document
from typing_extensions import TypedDict

__all__ = ["Intent", "GradeSignal", "Node", "IntentRoutingState"]


class Intent(str, Enum):
    """Query intent classification for adaptive RAG retrieval.

    Determines how to weight BM25 (keyword) vs Chroma (semantic) search:
    - FACT: Keyword-heavy (0.7 BM25 / 0.3 Chroma) for lists/rankings
    - CONCEPT: Balanced (0.3 BM25 / 0.7 Chroma) for explanations
    - COMPARISON: Semantic-heavy (0.4 BM25 / 0.6 Chroma) for comparisons
    """

    FACT = "FACT"  # Queries asking for specific lists, rankings, or enumerated facts
    CONCEPT = "CONCEPT"  # Queries seeking explanations or understanding of concepts
    COMPARISON = "COMPARISON"  # Queries comparing or contrasting multiple items


class GradeSignal(str, Enum):
    """Relevance grading signal for retrieved documents.

    Used in the retrieval grader node to determine if retrieved documents
    sufficiently answer the query. YES allows proceeding, NO triggers query rewriting.
    """

    YES = "YES"  # Documents are relevant and provide useful information
    NO = "NO"  # Documents are not relevant or insufficient for the query


class Node(str, Enum):
    """Graph node names in the LangGraph RAG workflow.

    Each node represents a step in the intent-based adaptive RAG pipeline:
    - INTENT_ROUTER: Classifies query intent (FACT/CONCEPT/COMPARISON)
    - ADAPTIVE_RETRIEVER: Retrieves documents with intent-specific weights
    - RETRIEVAL_GRADER: Grades document relevance, decides retry or end
    - QUERY_REWRITER: Rewrites query for better retrieval on failure
    """

    INTENT_ROUTER = "intent_router"
    ADAPTIVE_RETRIEVER = "adaptive_retriever"
    RETRIEVAL_GRADER = "retrieval_grader"
    QUERY_REWRITER = "query_rewriter"


class IntentRoutingState(TypedDict):
    """State dictionary for the intent-based adaptive RAG LangGraph workflow.

    This TypedDict defines the structure of state passed between nodes in the RAG pipeline.
    It tracks the query lifecycle from intent classification through retrieval, grading, and potential rewriting.
    """

    query: str  # Current query being processed (may be rewritten)
    original_query: str  # Original user query, preserved for rewriting reference
    intent: Intent  # Classified intent (FACT/CONCEPT/COMPARISON) for adaptive retrieval
    retrieved_docs: list[Document]  # Documents retrieved from vector store
    retrieval_grade: GradeSignal  # LLM assessment of document relevance (YES/NO)
    rewritten_query: str  # Rewritten query if grading failed (empty otherwise)
    retry_count: int  # Number of query rewriting attempts (max 1)
