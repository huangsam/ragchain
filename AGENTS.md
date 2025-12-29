# AGENTS — Project Overview

This document summarizes the code layout under `src/ragchain`, key configuration in `pyproject.toml`, and the testing/CLI/tooling conventions used in this repository. It is intended for contributors and CI to quickly understand where things live and what to configure.

---

## 📁 Repository layout (src/ragchain)

A compact tree view of the repository layout:

```
src/ragchain/
├── api.py                # FastAPI app (/health, /ingest, /search, /ask)
├── cli.py                # Click-based CLI (serve, ingest, search, ask)
├── config.py             # Centralized configuration management (singleton)
├── loaders.py            # Document loaders for Wikipedia and other sources
├── rag.py                # LangChain RAG pipeline (embedding, chunking, retrieval, generation)
├── graph.py              # LangGraph intent-based adaptive RAG orchestration
├── router.py             # Intent routing logic
├── prompts.py            # LLM prompt templates
├── utils.py              # Utility functions for logging, timing, and other helpers
└── __init__.py           # Package initialization
```

**Key architectural notes:**

- **`config.py`** provides centralized configuration management:
  - Singleton `Config` class for environment variable handling
  - Typed attributes for Ollama models, Chroma settings, and feature flags
  - Used throughout the codebase for consistent configuration access

- **`rag.py`** is the core retrieval layer:
  - `get_embedder()` — Creates OllamaEmbeddings with `bge-m3` model for 1024-dimensional vectors with 8k context
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

- **`prompts.py`** contains prompt templates:
  - `RAG_ANSWER_TEMPLATE` — Answer generation from context
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

- **`utils.py`** provides logger helpers to simplify the monitoring experience, including:
  - `log_with_prefix()` — Logs messages with a consistent prefix for easier filtering
  - `log_timing()` — Measures and logs the duration of operations

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
- **Ollama integration** — `langchain-ollama` for embedding (`bge-m3`) and LLM generation
- **Vector store** — `chromadb` for semantic search (supports local persistent and remote HTTP)
- **BM25** — `rank-bm25` for keyword-based retrieval and ensemble ranking
- **FastAPI & Uvicorn** — REST API server
- **Click** — CLI framework for stack management and data operations
- **Pydantic Settings** — Environment configuration management
- **Data fetching** — `aiohttp` for async HTTP, `beautifulsoup4` + `wikipedia` for document loading

**Developer tooling** (installed via `uv sync`):

- **Ruff** (linter & formatter) — `line-length = 160`
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

- **LangGraph agentic RAG** — Intent-aware routing adapts retrieval weights for FACT/CONCEPT/COMPARISON queries
- **Reciprocal Rank Fusion** — Principled ensemble ranking (score = 1/(rank+60)) combining BM25 keyword and semantic search
- **Self-correcting** — Automatic query rewriting on retrieval failure (max 1 retry) with LLM-based relevance grading
- **Performance optimized** — Parallel retrieval (ThreadPoolExecutor), retriever caching, optional grading, and fast-path routing
- **bge-m3 model** — 1024-dimensional embeddings with 8k context window for superior semantic search (via Ollama)
- **Flexible & composable** — Supports local/remote Chroma storage; easily swappable embedders, vector stores, and LLM models via config
- **Deterministic testing** — Mock HTTP (aioresponses) enables testing without Ollama/Chroma servers; Docker profiles for CI/demo
