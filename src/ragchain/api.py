"""FastAPI application for RAG endpoints."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragchain.loaders import load_tiobe_languages, load_wikipedia_pages
from ragchain.rag import ingest_documents, search

app = FastAPI()


class IngestRequest(BaseModel):
    languages: list[str] | None = None
    n_languages: int = 10


class SearchRequest(BaseModel):
    query: str
    k: int = 4


class AskRequest(BaseModel):
    query: str
    model: str = "qwen3"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Ingest programming languages from TIOBE or provided list."""
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
    """Search the vector store."""
    try:
        result = await search(req.query, k=req.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """Search + generate answer using Ollama LLM."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_community.llms.ollama import Ollama

        from ragchain.rag import OLLAMA_BASE_URL, get_vector_store

        store = get_vector_store()
        retriever = store.as_retriever(search_kwargs={"k": 4})
        llm = Ollama(model=req.model, base_url=OLLAMA_BASE_URL, temperature=0.7)

        # Build a simple RAG chain using LCEL
        template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        answer = chain.invoke(req.query)

        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
