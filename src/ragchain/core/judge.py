"""LLM-as-judge evaluation for RAG answers."""

import json
import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from ragchain.data.config import config
from ragchain.data.utils import log_timing, log_with_prefix
from ragchain.prompts import JUDGE_PROMPT, RAG_ANSWER_TEMPLATE

logger = logging.getLogger(__name__)


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
    # Truncate context to avoid very long inference times (keep first ~4000 chars)
    max_context_chars = 4000
    if len(context) > max_context_chars:
        truncated_context = context[:max_context_chars] + "\n\n[...context truncated for evaluation...]"
        log_with_prefix(logger, logging.INFO, "judge_answer", f"Truncated context from {len(context)} to {max_context_chars} chars")
    else:
        truncated_context = context

    llm = OllamaLLM(model=model, base_url=config.ollama_base_url, temperature=0.0)

    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    judge_input = prompt.format(question=question, context=truncated_context, answer=answer)

    log_with_prefix(logger, logging.INFO, "judge_answer", f"Judging answer for question: {question[:50]}...")

    start = time.time()
    raw_response = llm.invoke(judge_input)
    log_timing(logger, "judge_answer", start, "LLM judgment completed")

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
        if evaluation and "correctness" in evaluation:
            log_with_prefix(logger, logging.INFO, "judge_answer", "Successfully extracted JSON from wrapped response")
        else:
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
                log_with_prefix(logger, logging.WARNING, "judge_answer", f"Invalid {criterion} score {score}, marking as parse error")
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
    from ragchain.core.graph import rag_graph

    llm = OllamaLLM(model=model, base_url=config.ollama_base_url, temperature=0.1)

    evaluations = []

    for question in questions:
        # Run RAG pipeline
        initial_state = {
            "query": question,
            "intent": "CONCEPT",
            "retrieved_docs": [],
            "retrieval_grade": "NO",
            "rewritten_query": "",
            "retry_count": 0,
        }

        final_state = rag_graph.invoke(initial_state)  # type: ignore[arg-type]
        retrieved_docs = final_state["retrieved_docs"]

        if not retrieved_docs:
            log_with_prefix(logger, logging.WARNING, "evaluate_questions", f"No documents retrieved for: {question[:50]}...")
            continue

        # Generate answer
        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=question))

        # Judge the answer
        evaluation = await judge_answer(question, context, answer, model)

        evaluations.append({"question": question, "answer": answer, "evaluation": evaluation})

    return evaluations
