"""LLM-as-judge evaluation for RAG answers."""

import json
import logging
from typing import cast

from langchain_core.prompts import ChatPromptTemplate

from ragchain.config import config
from ragchain.prompts import JUDGE_PROMPT, RAG_ANSWER_TEMPLATE
from ragchain.types import IntentRoutingState
from ragchain.utils import get_llm, log_with_prefix, timed

logger = logging.getLogger(__name__)


@timed(logger, "judge_answer")
async def judge_answer(question: str, context: str, answer: str, model: str = config.ollama_model) -> dict:
    """Evaluate a RAG answer using LLM-as-judge for correctness, relevance, and faithfulness.

    Args:
        question: The original question
        context: The retrieved context documents as concatenated text
        answer: The generated answer to evaluate
        model: Ollama model to use for judging

    Returns:
        dict with evaluation scores and explanations
    """
    # Truncate context to balance speed and quality (keep first ~2500 chars)
    max_context_chars = 2500
    if len(context) > max_context_chars:
        truncated_context = context[:max_context_chars] + "\n\n[...truncated...]"
    else:
        truncated_context = context

    llm = get_llm(model=model, purpose="judging")

    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    judge_input = prompt.format(question=question, context=truncated_context, answer=answer)

    raw_response = llm.invoke(judge_input)

    # Parse JSON response - extract JSON from potential markdown or text wrapper
    def extract_json_object(text: str) -> dict | None:
        """Extract JSON object handling nested braces properly."""
        # Find the first { and match to the corresponding }
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if brace_count != 0:
            return None

        json_str = text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    # Try direct parsing first
    try:
        evaluation = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        # Try to extract JSON object with proper brace matching
        evaluation = extract_json_object(raw_response)
        if not (evaluation and "correctness" in evaluation):
            # Log the raw response for debugging
            log_with_prefix(logger, logging.ERROR, "judge_answer", f"Failed to parse judge response. Raw response: {raw_response[:500]}")
            return {
                "correctness": {"score": 0, "explanation": "Failed to parse response"},
                "relevance": {"score": 0, "explanation": "Failed to parse response"},
                "faithfulness": {"score": 0, "explanation": "Failed to parse response"},
            }

    # Validate and fix scores - ensure they're in 1-5 range
    for criterion in ["correctness", "relevance", "faithfulness"]:
        if criterion in evaluation and isinstance(evaluation[criterion], dict):
            score = evaluation[criterion].get("score", 0)
            if not isinstance(score, int) or score < 1 or score > 5:
                logger.warning(f"[judge_answer] Invalid {criterion} score {score}, marking as parse error")
                evaluation[criterion]["score"] = 0
                evaluation[criterion]["explanation"] = f"Invalid score: {score}. " + evaluation[criterion].get("explanation", "")

    return evaluation


async def evaluate_questions(questions: list[str], model: str = config.ollama_model) -> list[dict]:
    """Evaluate RAG system on a list of questions.

    Args:
        questions: List of questions to evaluate
        model: LLM model to use for generation and judging

    Returns:
        List of evaluation results with question, answer, and scores
    """
    from ragchain.inference.graph import rag_graph

    llm = get_llm(model=model, purpose="generation")

    evaluations = []

    for question in questions:
        # Run RAG pipeline
        initial_state = cast(
            IntentRoutingState,
            {
                "query": question,
                "original_query": question,
                "intent": "CONCEPT",
                "retrieved_docs": [],
                "retrieval_grade": "NO",
                "rewritten_query": "",
                "retry_count": 0,
            },
        )

        final_state = rag_graph.invoke(initial_state)  # type: ignore[arg-type]
        retrieved_docs = final_state["retrieved_docs"]

        logger.info(f"[evaluate_questions] Retrieved {len(retrieved_docs)} docs for question: {question[:50]}...")
        if not retrieved_docs:
            continue

        # Generate answer
        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=question))

        # Judge the answer
        evaluation = await judge_answer(question, context, answer, model)

        evaluations.append({"question": question, "answer": answer, "evaluation": evaluation})

    return evaluations
