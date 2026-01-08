"""Document relevance grading for RAG pipeline."""

import logging
import re

from langchain_core.documents import Document

from ragchain.config import config
from ragchain.types import GradeSignal

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


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Args:
        text: Text to extract keywords from.

    Returns:
        Set of lowercase keywords (excluding common stop words).
    """
    # Simple stop words list
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "this",
        "these",
        "those",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
        "have",
        "had",
        "been",
        "being",
    }

    # Extract words, lowercase, filter stop words and short words
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in stop_words}


def grade_with_statistics(query: str, docs: list[Document]) -> GradeSignal:
    """Grade document relevance using ranking metrics (MRR-inspired scoring).

    Scores each document based on keyword overlap, then uses hit rate
    to determine if top-ranked documents are relevant.

    Args:
        query: The search query.
        docs: Retrieved documents to grade.

    Returns:
        GradeSignal.YES if relevant documents found in top positions, GradeSignal.NO otherwise.
    """
    try:
        logger.info(f"[grade_with_statistics] Grading {len(docs)} docs for query: {query}")

        # Extract keywords from query
        query_keywords = extract_keywords(query)

        if not query_keywords:
            logger.warning("[grade_with_statistics] No keywords extracted from query, accepting docs")
            return GradeSignal.YES

        # Score each document based on keyword overlap and TF-like scoring
        doc_scores = []
        for i, doc in enumerate(docs):
            doc_text = doc.page_content.lower()
            doc_keywords = extract_keywords(doc.page_content)

            # Overlap ratio (Jaccard-like)
            overlap = query_keywords & doc_keywords
            overlap_ratio = len(overlap) / len(query_keywords) if query_keywords else 0

            # Term frequency bonus (how many times query keywords appear)
            tf_score = sum(doc_text.count(keyword) for keyword in query_keywords) / len(query_keywords)

            # Combined score (weighted: 70% overlap, 30% term frequency)
            score = 0.7 * overlap_ratio + 0.3 * min(tf_score, 1.0)

            doc_scores.append((i, score, overlap_ratio))
            logger.debug(f"[grade_with_statistics] Doc {i}: score={score:.3f} (overlap={overlap_ratio:.2%}, tf={tf_score:.2f})")

        # Sort by score descending (best documents first)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Hit rate approach: Check if top-ranked documents meet threshold
        # Use MRR-inspired logic: Higher weight to top-ranked docs
        relevance_threshold = 0.25  # Documents with score >= 0.25 are relevant

        # Check if we have any relevant documents in top-3 positions (hit@3)
        top_k = min(3, len(doc_scores))
        for rank, (doc_idx, score, _) in enumerate(doc_scores[:top_k], 1):
            if score >= relevance_threshold:
                reciprocal_rank = 1.0 / rank
                logger.info(f"[grade_with_statistics] Grade result: YES (doc {doc_idx} at rank {rank}, score={score:.3f}, MRR={reciprocal_rank:.3f})")
                return GradeSignal.YES

        # Log the best score for debugging
        if doc_scores:
            best_doc, best_score, best_overlap = doc_scores[0]
            logger.info(f"[grade_with_statistics] Grade result: NO (best doc {best_doc} score={best_score:.3f} < threshold {relevance_threshold})")
        else:
            logger.info("[grade_with_statistics] Grade result: NO (no documents to score)")

        return GradeSignal.NO

    except Exception as e:
        logger.error(f"[grade_with_statistics] Exception: {e}", exc_info=True)
        # On error, accept docs to avoid blocking the pipeline
        return GradeSignal.YES
