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
└── __init__.py           # Package initialization
```

**Key architectural notes:**

- **`rag.py`** is the core orchestrator using LangChain:
  - `get_embedder()` — Creates OllamaEmbeddings with `qwen3-embedding` model for 4096-dimensional vectors
  - `get_vector_store()` — Returns Chroma (local persistent or remote HTTP) with LangChain integration
  - `ingest()` — Fetches documents → parses → chunks recursively → embeds → upserts to vector store
  - `search()` — Ensemble retrieval using BM25 and Chroma vector search for improved relevance
  - `generate()` — LLM-powered answer generation using Ollama (configurable model)

- **`loaders.py`** provides document loading utilities:
  - Wikipedia article fetching (via built-in Wikipedia API or custom parsers)
  - Extensible for other sources (local files, APIs, etc.)

- **`api.py`** exposes FastAPI endpoints for the full RAG pipeline
- **`cli.py`** provides Click-based commands for ingest, search, query, and stack management

- Supports both **local persistent Chroma** (`CHROMA_PERSIST_DIRECTORY`) and **remote HTTP Chroma** (`CHROMA_SERVER_URL`)
- Uses **ensemble retrieval** combining BM25 keyword search and semantic vector search for better results
- Tests use deterministic embeddings and mock external HTTP where possible (using `aioresponses`)

---

## 🧰 Tooling and configuration (`pyproject.toml` highlights)

**Runtime dependencies:**

- **LangChain ecosystem** — LangChain, LangChain-Community, LangChain-Ollama, LangChain-Chroma for unified RAG orchestration
- **Ollama integration** — `langchain-ollama` for embedding (`qwen3-embedding`) and LLM generation
- **Vector store** — `chromadb` for semantic search (supports local persistent and remote HTTP)
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

- **LangChain integration** — Unified orchestration of embedding, chunking, retrieval, and generation stages
- **qwen3-embedding model** — 4096-dimensional dense embeddings for superior semantic search (via Ollama)
- **Flexible storage** — Supports both local persistent Chroma (`CHROMA_PERSIST_DIRECTORY`) and remote HTTP (`CHROMA_SERVER_URL`)
- **Composable pipeline** — Easy to swap components (embedders, vector stores, LLM models) via environment configuration
- **Deterministic testing** — Tests use mock HTTP (via `aioresponses`) and can run without Ollama/Chroma servers
- **Docker Compose profiles** — `test` profile for CI, `demo` profile for full feature showcase
