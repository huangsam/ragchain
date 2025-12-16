# AGENTS — Project Overview

This document summarizes the code layout under `src/`, key configuration in `pyproject.toml`, and the testing/CLI/tooling conventions used in this repository. It is intended for contributors and CI to quickly understand where things live and what to configure.

---

## 📁 Repository layout (src/ragchain)

A compact tree view of the repository layout:

```
ragchain/
├── api.py                # FastAPI app (/health, /ingest, /search)
├── cli.py                # Click-based CLI (`ragchain serve`)
├── parser/
│   ├── wiki_client.py    # Concurrent fetches of Wikipedia pages; atomic writes
│   └── html_parser.py    # Extracts text from mobile-sections HTML
├── rag/
│   ├── chunker.py        # char-based sliding-window chunker
│   ├── embeddings.py     # DummyEmbedding + optional sentence-transformers adapter
│   └── ingest.py         # Orchestrates fetch -> parse -> chunk -> embed -> upsert
├── vectorstore/
│   └── chroma_vectorstore.py  # Chroma adapter: remote / persistent / ephemeral modes
├── utils.py              # Utility helpers (e.g., `safe_filename`)
└── tests/
    ├── unit/
    ├── integration/
    └── conftest.py      # `chroma_store` fixture for inprocess & remote tests
```

- `chroma_vectorstore.py` supports remote HTTP (`CHROMA_SERVER_URL`), persistent on-disk (`CHROMA_PERSIST_DIRECTORY`), or ephemeral in-memory modes.
- Tests favor deterministic behavior (e.g., `DummyEmbedding`) and mock external HTTP where possible (using `aioresponses`).

---

## 🧰 Tooling and configuration (`pyproject.toml` highlights)

- Runtime deps of note:
  - `aiohttp` — HTTP client for fetches
  - `chromadb` — optional; used by `ChromaVectorStore`
  - `fastapi`, `uvicorn` — API server
  - `click` — CLI
  - `sentence-transformers` — optional for non-dummy embeddings
  - `pydantic-settings` — compatibility with Pydantic v2 in some environments

- Developer tooling (installed via `uv sync` / `uv add`):
  - Ruff (linter & formatter) — configured with `line-length = 160`
  - isort — `profile = "black"`, `line_length = 160`
  - mypy — static typing checks (configured to ignore missing imports and run in silent follow-import mode)

- Project entry points:
  - `ragchain` console script -> `ragchain.cli:main` (install in editable mode to use `ragchain serve`)

- Recommended Python version: **3.12** (some deps such as `chromadb` and sentence-transformers have prebuilt wheels for this version).

---

## 🧪 Running tests and remote Chroma

- Run unit tests:

```bash
uv run --with-editable . pytest -q
```

- Run remote integration tests against a local Chroma service:

```bash
docker-compose up -d
CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py
```

- The test fixture `chroma_store` will skip remote tests cleanly if no server is reachable and guide you to run `docker-compose up -d`.

---

## 🔧 Notes & Rationale

- Tests prefer deterministic behavior (e.g., `DummyEmbedding`) to avoid network/third-party flakiness.
- The Chroma adapter is split to make it easy to run in-process persistent stores for local dev and remote HTTP servers in CI.
- The codebase uses small, well-scoped threadpools to bridge blocking SDKs into async code.
