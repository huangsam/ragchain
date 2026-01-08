"""Router and grader prompts for intent-based adaptive RAG."""

import logging
import time

from langchain_ollama import OllamaLLM

from ragchain.config import config
from ragchain.prompts import INTENT_ROUTER_PROMPT
from ragchain.types import Intent, IntentRoutingState
from ragchain.utils import log_timing, log_with_prefix

logger = logging.getLogger(__name__)

__all__ = ["intent_router"]


def intent_router(state: IntentRoutingState) -> IntentRoutingState:
    """Route query to intent category."""
    start = time.time()
    log_with_prefix(logger, logging.INFO, "intent_router", f"Starting for query: {state['query'][:50]}...")

    # Fast-path: Skip LLM for simple queries if routing is disabled
    query_lower = state["query"].lower()
    simple_patterns = ["what is", "define", "explain", "who is", "when was", "where is", "how does", "why is"]
    is_simple = any(pattern in query_lower for pattern in simple_patterns) and len(state["query"].split()) <= 8

    if not config.enable_intent_routing or is_simple:
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
