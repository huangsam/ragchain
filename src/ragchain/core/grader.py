"""Document relevance grading for RAG pipeline."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from ragchain.core.enums import GradeSignal
from ragchain.data.config import config
from ragchain.prompts import RETRIEVAL_GRADER_PROMPT

logger = logging.getLogger(__name__)


def should_skip_grading() -> bool:
    """Determine if grading should be skipped.

    Returns:
        True if grading should be skipped.
    """
    return not config.enable_grading


def should_accept_docs(retrieved_docs: list[Document], retry_count: int) -> bool:
    """Determine if documents should be auto-accepted.

    Args:
        retrieved_docs: List of retrieved documents.
        retry_count: Current retry count.

    Returns:
        True if docs should be accepted without grading.
    """
    return not retrieved_docs or retry_count > 0


def grade_with_llm(query: str, docs: List[Document]) -> GradeSignal:
    """Grade document relevance using LLM.

    Args:
        query: The search query.
        docs: Retrieved documents to grade.

    Returns:
        GradeSignal.YES if relevant, GradeSignal.NO if not.
    """
    try:
        llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url, temperature=0)

        formatted_docs = "\n\n".join([f"Doc {i}: {doc.page_content[:200]}" for i, doc in enumerate(docs)])
        logger.info(f"[grade_with_llm] Grading {len(docs)} docs for query: {query}")

        prompt = RETRIEVAL_GRADER_PROMPT.format(query=query, formatted_docs=formatted_docs)
        response = llm.invoke(prompt).strip().upper()

        # Extract first word to be robust to extra text
        first_word = response.split()[0] if response else ""
        # Clean punctuation from the first word
        first_word = "".join(c for c in first_word if c.isalnum())

        logger.debug(f"[grade_with_llm] Raw response: {response!r}")
        result = GradeSignal.YES if first_word == GradeSignal.YES.value else GradeSignal.NO
        logger.info(f"[grade_with_llm] Grade result: {result.value}")
        return result
    except Exception as e:
        logger.error(f"[grade_with_llm] Exception: {e}", exc_info=True)
        return GradeSignal.NO
