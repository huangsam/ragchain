"""FastAPI application for RAG endpoints."""

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from ragchain.core.rag import search
from ragchain.data.config import config
from ragchain.data.utils import log_timing, log_with_prefix
from ragchain.prompts import RAG_ANSWER_TEMPLATE

logger = logging.getLogger(__name__)
app = FastAPI()


def _handle_endpoint_error(e: Exception, endpoint: str) -> None:
    """Log and raise HTTP exception for endpoint errors."""
    log_with_prefix(logger, logging.ERROR, endpoint, f"Error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    """Request schema for semantic search endpoint."""

    query: str
    k: int = 8

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        """Validate k is within acceptable range."""
        if v <= 0 or v > 50:
            raise ValueError("k must be between 1 and 50")
        return v


class AskRequest(BaseModel):
    """Request schema for RAG-based question answering endpoint."""

    query: str
    model: str = config.ollama_model


@app.get("/health")
async def health():
    """Health check endpoint. Returns API status."""
    return {"status": "ok"}


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    """Perform semantic search on ingested documents.

    Returns top-k most similar documents based on vector similarity.
    """
    try:
        result = await search(req.query, k=req.k)
        return result
    except Exception as e:
        _handle_endpoint_error(e, "/search")


@app.post("/ask")
async def ask(req: AskRequest):
    """Answer questions using intent-based adaptive RAG.

    Uses LangGraph to route queries by intent, adapting retrieval strategy
    and grading results for quality. Retries with rewritten queries if needed.
    """
    log_with_prefix(logger, logging.INFO, "/ask", f"Received query: {req.query[:50]}...")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import OllamaLLM

        from ragchain.core.graph import rag_graph

        initial_state = {
            "query": req.query,
            "intent": "CONCEPT",
            "retrieved_docs": [],
            "retrieval_grade": "NO",
            "rewritten_query": "",
            "retry_count": 0,
        }

        log_with_prefix(logger, logging.INFO, "/ask", "Starting LangGraph pipeline")
        graph_start = time.time()
        final_state = rag_graph.invoke(initial_state)  # type: ignore[arg-type]
        log_timing(logger, "/ask", graph_start, "LangGraph completed")

        retrieved_docs = final_state["retrieved_docs"]
        log_with_prefix(logger, logging.INFO, "/ask", f"Retrieved {len(retrieved_docs)} documents")

        log_with_prefix(logger, logging.INFO, "/ask", "Generating answer")
        gen_start = time.time()
        llm = OllamaLLM(model=req.model, base_url=config.ollama_base_url, temperature=0.1)

        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=req.query))
        log_timing(logger, "/ask", gen_start, "Answer generated")

        return {"query": req.query, "answer": answer}
    except Exception as e:
        _handle_endpoint_error(e, "/ask")
