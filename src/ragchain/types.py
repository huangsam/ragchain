"""Shared types and enums for the RAG pipeline."""

from enum import Enum
from typing import Any

from langchain_core.documents import Document
from typing_extensions import TypedDict

__all__ = ["GradeSignal", "IngestResult", "Intent", "IntentRoutingState", "Node", "SearchResult"]


class Intent(str, Enum):
    """Query intent classification for adaptive RAG retrieval.

    Determines how to weight BM25 (keyword) vs Chroma (semantic) search.
    Each intent type adjusts weights accordingly for different use cases.
    """

    # Queries asking for specific lists, rankings, or enumerated facts
    FACT = "FACT"
    # Queries seeking explanations or understanding of concepts
    CONCEPT = "CONCEPT"
    # Queries comparing or contrasting multiple items
    COMPARISON = "COMPARISON"


class GradeSignal(str, Enum):
    """Relevance grading signal for retrieved documents.

    Used in the retrieval grader node to determine if retrieved documents
    sufficiently answer the query. YES allows proceeding, NO triggers query rewriting.
    """

    # Documents are relevant and provide useful information
    YES = "YES"
    # Documents are not relevant or insufficient for the query
    NO = "NO"


class Node(str, Enum):
    """Graph node names in the LangGraph RAG workflow.

    Each node represents a step in the intent-based adaptive RAG pipeline.
    Look at the corresponding functions in graph.py for implementation
    details.
    """

    # Classifies query intent (FACT/CONCEPT/COMPARISON)
    INTENT_ROUTER = "intent_router"
    # Retrieves documents with intent-specific weights
    ADAPTIVE_RETRIEVER = "adaptive_retriever"
    # Grades document relevance, decides retry or end
    RETRIEVAL_GRADER = "retrieval_grader"
    # Rewrites query for better retrieval on failure
    QUERY_REWRITER = "query_rewriter"


class IntentRoutingState(TypedDict):
    """State dictionary for the intent-based adaptive RAG LangGraph workflow.

    This TypedDict defines the structure of state passed between nodes in the RAG pipeline.
    It tracks the query lifecycle from intent classification through retrieval, grading, and potential rewriting.
    """

    # Current query being processed (may be rewritten)
    query: str
    # Original user query, preserved for rewriting reference
    original_query: str
    # Classified intent (FACT/CONCEPT/COMPARISON) for adaptive retrieval
    intent: Intent
    # Documents retrieved from vector store
    retrieved_docs: list[Document]
    # LLM assessment of document relevance (YES/NO)
    retrieval_grade: GradeSignal
    # Rewritten query if grading failed (empty otherwise)
    rewritten_query: str
    # Number of query rewriting attempts (max 1)
    retry_count: int


class IngestResult(TypedDict):
    """Result of document ingestion operation.

    This TypedDict captures the outcome of an ingestion process, including status,
    number of documents ingested, a message, and elapsed time.
    """

    # "SUCCESS" or "FAILURE"
    status: str
    # Number of documents ingested
    count: int
    # Additional information about the ingestion
    message: str
    # Time taken for the ingestion process
    elapsed_seconds: float


class SearchResult(TypedDict):
    """Result of a RAG search operation.

    This TypedDict captures the outcome of a search process, including the query,
    and a list of result dictionaries containing relevant information.
    """

    # The search query
    query: str
    # List of result dictionaries with relevant information
    results: list[dict[str, Any]]
