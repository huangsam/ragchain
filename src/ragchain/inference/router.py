"""Router and grader prompts for intent-based adaptive RAG."""

import logging

from ragchain.config import config
from ragchain.prompts import INTENT_ROUTER_PROMPT
from ragchain.types import Intent, IntentRoutingState
from ragchain.utils import get_llm, timed

logger = logging.getLogger(__name__)

__all__ = ["intent_router"]


@timed(logger, "intent_router")
def intent_router(state: IntentRoutingState) -> IntentRoutingState:
    """Route query to intent category."""

    # Fast-path: Skip LLM for simple queries if routing is disabled
    query_lower = state["query"].lower()
    simple_patterns = ["what is", "define", "explain", "who is", "when was", "where is", "how does", "why is"]
    is_simple = any(pattern in query_lower for pattern in simple_patterns) and len(state["query"].split()) <= 8

    if not config.enable_intent_routing or is_simple:
        logger.debug("[intent_router] Using fast-path, defaulting to CONCEPT")
        return {**state, "intent": Intent.CONCEPT, "original_query": state["query"]}

    llm = get_llm(purpose="routing")

    prompt = INTENT_ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt).strip().upper()

    # Extract first valid intent
    valid_intents: list[Intent] = [Intent.FACT, Intent.CONCEPT, Intent.COMPARISON]
    intent_value: Intent = next((i for i in valid_intents if i.value in response), Intent.CONCEPT)

    logger.debug(f"[intent_router] Classified as {intent_value}")

    return {**state, "intent": intent_value, "original_query": state["query"]}
