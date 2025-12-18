"""FastAPI application for RAG endpoints."""

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from ragchain.config import config
from ragchain.loaders import load_tiobe_languages, load_wikipedia_pages
from ragchain.rag import ingest_documents, search

logger = logging.getLogger(__name__)
app = FastAPI()

# RAG answer generation prompt template
RAG_ANSWER_TEMPLATE = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""


class IngestRequest(BaseModel):
    """Request schema for document ingestion endpoint.

    Either specify languages list or n_languages to fetch from TIOBE.
    """

    languages: list[str] | None = None
    n_languages: int = 10

    @field_validator("n_languages")
    @classmethod
    def validate_n_languages(cls, v: int) -> int:
        """Validate n_languages is within acceptable range."""
        if v <= 0 or v > 100:
            raise ValueError("n_languages must be between 1 and 100")
        return v


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


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Ingest programming languages into vector store.

    Fetches Wikipedia articles and stores them in Chroma for semantic search.
    Returns ingestion result with chunk count.
    """
    try:
        if req.languages:
            langs = req.languages
        else:
            langs = await load_tiobe_languages(req.n_languages)

        if not langs:
            return {"status": "error", "message": "No languages fetched"}

        docs = await load_wikipedia_pages(langs)

        if not docs:
            return {"status": "error", "message": "No documents loaded"}

        result = await ingest_documents(docs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    """Perform semantic search on ingested documents.

    Returns top-k most similar documents based on vector similarity.
    """
    try:
        result = await search(req.query, k=req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """Answer questions using intent-based adaptive RAG.

    Uses LangGraph to route queries by intent, adapting retrieval strategy
    and grading results for quality. Retries with rewritten queries if needed.
    """
    start = time.time()
    logger.info(f"[/ask] Received query: {req.query[:50]}...")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import OllamaLLM

        from ragchain.graph import rag_graph

        # Initialize state dict
        initial_state = {
            "query": req.query,
            "intent": "CONCEPT",
            "retrieved_docs": [],
            "retrieval_grade": "NO",
            "rewritten_query": "",
            "retry_count": 0,
        }

        # Run the agentic RAG graph directly (sync)
        logger.info("[/ask] Starting LangGraph pipeline")
        graph_start = time.time()
        final_state = rag_graph.invoke(initial_state)  # type: ignore[arg-type]
        logger.info(f"[/ask] LangGraph completed in {time.time() - graph_start:.2f}s")

        retrieved_docs = final_state["retrieved_docs"]
        logger.info(f"[/ask] Retrieved {len(retrieved_docs)} documents")

        # Generate answer from retrieved docs

        logger.info("[/ask] Generating answer")
        gen_start = time.time()
        llm = OllamaLLM(model=req.model, base_url=config.ollama_base_url, temperature=0.7)

        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=req.query))
        logger.info(f"[/ask] Answer generated in {time.time() - gen_start:.2f}s")

        total_elapsed = time.time() - start
        logger.info(f"[/ask] Completed in {total_elapsed:.2f}s")

        return {"query": req.query, "answer": answer}
    except Exception as e:
        logger.error(f"[/ask] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
