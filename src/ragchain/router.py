"""Router and grader prompts for intent-based adaptive RAG."""

import logging
import time
from enum import Enum

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from typing_extensions import TypedDict

from ragchain.config import config
from ragchain.utils import log_timing, log_with_prefix

logger = logging.getLogger(__name__)

__all__ = [
    "RAG_ANSWER_TEMPLATE",
    "INTENT_ROUTER_PROMPT",
    "RETRIEVAL_GRADER_PROMPT",
    "QUERY_REWRITER_PROMPT",
    "Intent",
    "IntentRoutingState",
    "GradeSignal",
    "intent_router",
    "_is_simple_query",
]


class Intent(str, Enum):
    """Query intent classification."""

    FACT = "FACT"
    CONCEPT = "CONCEPT"
    COMPARISON = "COMPARISON"


class GradeSignal(str, Enum):
    """Relevance grading signal for retrieved documents."""

    YES = "YES"
    NO = "NO"


class IntentRoutingState(TypedDict):
    """State for the intent routing RAG graph."""

    query: str
    original_query: str  # Preserve original query for rewriting
    intent: Intent
    retrieved_docs: list[Document]  # Wait, Document is not imported here
    retrieval_grade: GradeSignal  # Not imported
    rewritten_query: str
    retry_count: int


# Helps with answering open-ended prompts
RAG_ANSWER_TEMPLATE = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""

# Helps with classification for one category
INTENT_ROUTER_PROMPT = """Classify this query into ONE category:

FACT: Asks for a specific list, ranking, or enumerated facts
  Examples: "What are the top 10 languages?", "List languages with static typing"

CONCEPT: Asks for explanation or understanding of a concept
  Examples: "What is functional programming?", "Explain garbage collection"

COMPARISON: Asks to compare or contrast multiple items
  Examples: "Compare Go and Rust", "What are differences between Python and Java?"

Query: {query}

Answer with only the category name (FACT, CONCEPT, or COMPARISON):"""

# Helps with grading the relevance of retrieved documents
RETRIEVAL_GRADER_PROMPT = """You are a grader for retrieval quality. Judge if these documents are relevant to the query.

Query: {query}

Retrieved Documents:
{formatted_docs}

GRADING RULES:
1. If ANY document mentions the query topic → ANSWER: YES
2. If ANY document contains information related to the query → ANSWER: YES
3. Only answer NO if ALL documents are completely unrelated to the query topic

INSTRUCTION: This is a lenient grading. Most queries should receive YES unless the documents are obviously wrong.

Answer with ONLY the word YES or NO, nothing else:"""

# Helps with rewriting queries to be more explicit
QUERY_REWRITER_PROMPT = """Your previous retrieval for this query didn't return relevant documents:
Original Query: {query}

Rewrite this query to be more explicit, adding keywords that might be in a list or ranking.

Examples:
- "What are the top 10 languages?" → "TIOBE index top 10 most popular programming languages ranking list"
- "Compare Go and Rust" → "Go versus Rust comparison features differences systems programming"

Rewritten Query:"""


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
        return {**state, "intent": Intent.CONCEPT, "original_query": state["query"]}

    llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url, temperature=0)

    prompt = INTENT_ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt).strip().upper()

    # Extract first valid intent
    valid_intents: list[Intent] = [Intent.FACT, Intent.CONCEPT, Intent.COMPARISON]
    intent_value: Intent = next((i for i in valid_intents if i.value in response), Intent.CONCEPT)

    log_timing(logger, "intent_router", start, f"Classified as {intent_value}")

    return {**state, "intent": intent_value, "original_query": state["query"]}
