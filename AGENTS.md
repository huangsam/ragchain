# AGENTS — Project Overview

This document summarizes the code layout under `src/ragchain`, key configuration in `pyproject.toml`, and the testing/CLI/tooling conventions used in this repository. It is intended for contributors and CI to quickly understand where things live and what to configure.

---

## 📁 Repository layout (src/ragchain)

A compact tree view of the repository layout:

```
src/ragchain/
├── api.py                # FastAPI app (/health, /ingest, /search, /ask)
├── cli.py                # Click-based CLI (serve, ingest, search, ask)
├── loaders.py            # Document loaders for Wikipedia and other sources
├── rag.py                # LangChain RAG pipeline (embedding, chunking, retrieval, generation)
├── graph.py              # LangGraph intent-based adaptive RAG orchestration
├── router.py             # LLM prompts for intent routing and retrieval grading
└── __init__.py           # Package initialization
```

**Key architectural notes:**

- **`rag.py`** is the core retrieval layer:
  - `get_embedder()` — Creates OllamaEmbeddings with `qwen3-embedding` model for 4096-dimensional vectors
  - `get_vector_store()` — Returns Chroma (local persistent or remote HTTP) with LangChain integration
  - `ingest_documents()` — Fetches documents → parses → chunks recursively → embeds → upserts to vector store
  - `search()` — Legacy ensemble retrieval (BM25 + Chroma with RRF)
  - `EnsembleRetriever` — Custom retriever implementing Reciprocal Rank Fusion (RRF) with configurable weights
  - `get_ensemble_retriever()` — Factory with intent-specific weight support

- **`graph.py`** is the agentic orchestrator using LangGraph:
  - `IntentRoutingState` — Typed state management for the RAG graph
  - `intent_router()` — LLM-based query classification (FACT/CONCEPT/COMPARISON)
  - `adaptive_retriever()` — Retrieves with intent-specific BM25/Chroma weights
  - `retrieval_grader()` — LLM-based validation of document relevance
  - `query_rewriter()` — Enhances queries on retrieval failure for automatic retry
  - `rag_graph` — Compiled LangGraph with conditional retry logic

- **`router.py`** contains LLM prompts:
  - `INTENT_ROUTER_PROMPT` — Query classification
  - `RETRIEVAL_GRADER_PROMPT` — Document relevance validation
  - `QUERY_REWRITER_PROMPT` — Query enhancement

- **`loaders.py`** provides document loading utilities:
  - Wikipedia article fetching (via built-in Wikipedia API or custom parsers)
  - Extensible for other sources (local files, APIs, etc.)

- **`api.py`** exposes FastAPI endpoints:
  - `/health` — Health check
  - `/ingest` — Ingest documents
  - `/search` — Legacy ensemble search
  - `/ask` — Intent-based adaptive RAG (uses `rag_graph`)

- **`cli.py`** provides Click-based commands for ingest, search, query, and stack management

- Supports both **local persistent Chroma** (`CHROMA_PERSIST_DIRECTORY`) and **remote HTTP Chroma** (`CHROMA_SERVER_URL`)
- Uses **ensemble retrieval** with Reciprocal Rank Fusion (RRF) combining BM25 keyword search and semantic vector search
- Implements **intent-based adaptive RAG** via LangGraph:
  - FACT queries: 0.8 BM25 / 0.2 Chroma (keyword-heavy for enumerations)
  - CONCEPT queries: 0.4 BM25 / 0.6 Chroma (balanced)
  - COMPARISON queries: 0.3 BM25 / 0.7 Chroma (semantic-heavy)
- **Self-correcting**: Automatically rewrites and re-retrieves if grading fails (max 1 retry)
- Tests use deterministic embeddings and mock external HTTP where possible (using `aioresponses`)

---

## 🧰 Tooling and configuration (`pyproject.toml` highlights)

**Runtime dependencies:**

- **LangChain ecosystem** — LangChain, LangChain-Community, LangChain-Ollama, LangChain-Chroma for unified RAG orchestration
- **LangGraph** — `langgraph` for agentic RAG orchestration with state management and conditional routing
- **Ollama integration** — `langchain-ollama` for embedding (`qwen3-embedding`) and LLM generation
- **Vector store** — `chromadb` for semantic search (supports local persistent and remote HTTP)
- **BM25** — `rank-bm25` for keyword-based retrieval and ensemble ranking
- **FastAPI & Uvicorn** — REST API server
- **Click** — CLI framework for stack management and data operations
- **Pydantic Settings** — Environment configuration management
- **Data fetching** — `aiohttp` for async HTTP, `beautifulsoup4` + `wikipedia` for document loading

**Developer tooling** (installed via `uv sync`):

- **Ruff** (linter & formatter) — `line-length = 160`
- **isort** — `profile = "black"`, `line_length = 160`
- **mypy** — static type checking (configured to ignore missing imports)
- **pytest** + **pytest-asyncio** — testing framework
- **aioresponses** — mock async HTTP requests in tests

**Project entry points:**

- `ragchain` console script → `ragchain.cli:cli` (enables `ragchain serve`, `ragchain up`, `ragchain down`, etc.)

**Recommended Python version:** **3.12** (LangChain ecosystem has optimized wheels)

---

## 🧪 Running tests and remote Chroma

**Unit tests** (using mocked dependencies):

```bash
uv run --with-editable . pytest -q
```

**Integration tests** against a running local Chroma service:

```bash
# Start a Chroma test stack
ragchain up --profile test

# Run full pipeline integration tests
CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py

# Tear down the test stack
ragchain down --profile test
```

**Local development and demo:**

- `ragchain up` — Starts the demo stack (`docker compose up -d --profile demo`): Chroma + ragchain API + demo-runner
- `ragchain up --profile test` — Starts minimal test stack (Chroma only) for CI-like testing
- `ragchain down` — Stops the current docker compose stack
- `docker compose up --build` — Manually start the full demo (builds all services)

**Stack components:**

- **Chroma** (vector database) — `http://localhost:8000` (configured for both test and demo profiles)
- **ragchain API** — `http://localhost:8003` (demo profile only)
- **demo-runner** — Automatically runs example ingest + search workflows on startup (demo profile only)

---

## 🔧 Notes & Rationale

- **LangGraph agentic RAG** — Intent-aware routing enables self-correcting retrieval that adapts to query type
- **Reciprocal Rank Fusion** — Principled fusion of BM25 and semantic rankings (score = 1/(rank+60)) prevents rank 1 from dominating
- **Intent classification** — Distinguishes FACT (exact lists), CONCEPT (explanation), and COMPARISON (contrast) queries for optimal retrieval
- **Self-correcting** — Retrieval grader validates document relevance; rewrites queries on failure for automatic recovery
- **qwen3-embedding model** — 4096-dimensional dense embeddings for superior semantic search (via Ollama)
- **Flexible storage** — Supports both local persistent Chroma (`CHROMA_PERSIST_DIRECTORY`) and remote HTTP (`CHROMA_SERVER_URL`)
- **Composable pipeline** — Easy to swap components (embedders, vector stores, LLM models, intent weights) via environment configuration
- **Deterministic testing** — Tests use mock HTTP (via `aioresponses`) and can run without Ollama/Chroma servers
- **Docker Compose profiles** — `test` profile for CI, `demo` profile for full feature showcase
