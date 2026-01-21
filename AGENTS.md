# AGENTS — Project Overview

This document summarizes the code layout under `src/ragchain`, key configuration in `pyproject.toml`, and the testing/CLI/tooling conventions used in this repository. It is intended for contributors and CI to quickly understand where things live and what to configure.

---

## 📁 Repository layout (src/ragchain)

A compact tree view of the repository layout:

```
src/ragchain/
├── cli.py                # Click-based CLI (ingest, search, ask, evaluate)
├── prompts.py            # LLM prompt templates
├── config.py             # Configuration management (singleton)
├── types.py              # Shared enums and TypedDicts
├── utils.py              # Utility functions for logging, timing, and other helpers
├── evaluation/           # Answer generation and evaluation
│   ├── __init__.py
│   └── judge.py          # LLM-as-judge evaluation for RAG answers
├── ingestion/            # Document loading and storage
│   ├── __init__.py
│   ├── loaders.py        # Document loaders for Wikipedia and other sources
│   └── storage.py        # Storage utilities: embeddings, vector store, document ingestion
├── inference/            # Retrieval, routing, and orchestration
│   ├── __init__.py
│   ├── graph.py          # LangGraph intent-based adaptive RAG orchestration
│   ├── rag.py            # RAG search orchestration
│   ├── retrievers.py     # Retrieval utilities: ensemble retriever and helpers
│   ├── router.py         # Intent routing logic
│   └── grader.py         # Document relevance grading
└── __init__.py           # Package initialization
```

**Key architectural notes:**

- **`data/config.py`** provides centralized configuration management:
  - Singleton `Config` class for environment variable handling
  - Typed attributes for Ollama models, Chroma settings, and feature flags
  - Used throughout the codebase for consistent configuration access

- **`inference/rag.py`** is the RAG search orchestration:
  - `search()` — Ensemble retrieval using BM25 and Chroma vector search

- **`ingestion/storage.py`** handles storage and ingestion:
  - `get_embedder()` — Creates OllamaEmbeddings with `bge-m3` model for 1024-dimensional vectors with 8k context
  - `get_vector_store()` — Returns Chroma (local persistent or remote HTTP) with LangChain integration
  - `ingest_documents()` — Fetches documents → parses → chunks recursively → embeds → upserts to vector store

- **`inference/retrievers.py`** provides retrieval logic:
  - `EnsembleRetriever` — Custom retriever implementing Reciprocal Rank Fusion (RRF) with configurable weights
  - `get_ensemble_retriever()` — Factory with intent-specific weight support

- **`inference/graph.py`** is the agentic orchestrator using LangGraph:
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

- **`ingestion/loaders.py`** provides document loading utilities:
  - Wikipedia article fetching (via built-in Wikipedia API or custom parsers)
  - Extensible for other sources (local files, APIs, etc.)

- **`cli.py`** provides Click-based commands:
  - `ingest` — Load documents into vector store
  - `search` — Semantic search over ingested documents
  - `ask` — Intent-based adaptive RAG with LLM generation
  - `evaluate` — LLM-as-judge evaluation framework

- **`utils.py`** provides logger helpers to simplify the monitoring experience, including:
  - `log_with_prefix()` — Logs messages with a consistent prefix for easier filtering

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
- **Click** — CLI framework for data operations and queries
- **Pydantic Settings** — Environment configuration management
- **Data fetching** — `aiohttp` for async HTTP, `beautifulsoup4` + `wikipedia` for document loading

**Developer tooling** (installed via `uv sync`):

- **Ruff** (linter & formatter) — `line-length = 160`
- **mypy** — static type checking (configured to ignore missing imports)
- **pytest** + **pytest-asyncio** — testing framework with integration markers
- **aioresponses** — mock async HTTP requests in tests

**Project entry points:**

- `ragchain` console script → `ragchain.cli:cli` (enables `ragchain ingest`, `ragchain search`, `ragchain ask`, etc.)

**Recommended Python version:** **3.12** (LangChain ecosystem has optimized wheels)

---

## ⚙️ Environment Variables

The following environment variables can be used to configure the RAGChain system:

**Vector Store Configuration:**

- `CHROMA_PERSIST_DIRECTORY` — Directory for local Chroma persistence (default: `./chroma_data`)
- `CHROMA_SERVER_URL` — URL for remote Chroma server (default: `http://localhost:8000`)

**Ollama Configuration:**

- `OLLAMA_BASE_URL` — Base URL for Ollama API (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` — Model name for embeddings (default: `qwen3-embedding:4b`)
- `OLLAMA_MODEL` — Model name for text generation (default: `qwen3:8b`)
- `OLLAMA_EMBED_CTX` — Context window size for embedding model (default: `4096`)
- `OLLAMA_GEN_CTX` — Context window size for generation model (default: `8192`)

**Document Processing:**

- `CHUNK_SIZE` — Size of document chunks in characters (default: `2500`)
- `CHUNK_OVERLAP` — Overlap between chunks in characters (default: `500`)

**Retrieval Configuration:**

- `SEARCH_K` — Number of documents to retrieve per retriever for search API (default: `10`)
- `GRAPH_K` — Number of documents for graph-based RAG pipeline (default: `6`)
- `RRF_MAX_RESULTS` — Maximum results to return after RRF fusion (default: `10`)

**Feature Flags:**

- `ENABLE_GRADING` — Enable/disable document relevance grading (default: `true`)
- `ENABLE_INTENT_ROUTING` — Enable/disable intent-based routing (default: `true`)

---

## 🧪 Running tests and remote Chroma

**Unit tests** (using mocked dependencies):

```bash
uv run --with-editable . pytest -q
```

**Integration tests** against a running local Chroma service:

```bash
# Start Ollama (if not already running)
ollama serve

# Run full pipeline integration tests (uses local Chroma persistence)
CHROMA_SERVER_URL= uv run --with-editable . pytest -m integration
```

**Local development:**

- `docker compose up -d` — Starts Chroma vector database
- `ragchain ingest` — Ingest all 50 programming languages + 10 conceptual bridge pages
- `ragchain search "Python programming"` — Search ingested documents
- `ragchain ask "What is Python?"` — Ask questions with RAG + LLM
- `ragchain evaluate` — Run LLM-as-judge evaluation

**Stack components:**

- **Chroma** (vector database) — `http://localhost:8000`

---

## 🔧 Notes & Rationale

- **LangGraph agentic RAG** — Intent-aware routing adapts retrieval weights for FACT/CONCEPT/COMPARISON queries
- **Reciprocal Rank Fusion** — Principled ensemble ranking (score = 1/(rank+60)) combining BM25 keyword and semantic search
- **Self-correcting** — Automatic query rewriting on retrieval failure (max 1 retry) with LLM-based relevance grading
- **Performance optimized** — Parallel retrieval (ThreadPoolExecutor), retriever caching, optional grading, and fast-path routing
- **bge-m3 model** — 1024-dimensional embeddings with 8k context window for superior semantic search (via Ollama)
- **Flexible & composable** — Supports local/remote Chroma storage; easily swappable embedders, vector stores, and LLM models via config
- **Deterministic testing** — Mock HTTP (aioresponses) enables testing without Ollama/Chroma servers; Docker profiles for CI/demo
