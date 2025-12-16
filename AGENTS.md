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
│   └── html_parser.py    # Extracts text from sections-like JSON (e.g., MediaWiki extracts)
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
  - `ragchain` console script -> `ragchain.cli:cli` (install in editable mode to use `ragchain serve`, `ragchain up`, and `ragchain down`)

- Recommended Python version: **3.12** (some deps such as `chromadb` and sentence-transformers have prebuilt wheels for this version).

---

## 🧪 Running tests and remote Chroma

- Run unit tests:

```bash
uv run --with-editable . pytest -q
```

- Run remote integration tests against a local Chroma service:

```bash
# Start a Chroma test stack for CI-like tests
docker compose up -d --profile test
# Run the remote integration test that targets the running server
CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py
# Tear down the test stack
docker compose --profile test down
```

- Local/demo conveniences:

  - `ragchain up` will run `docker compose up -d --profile demo` to start the demo stack (Chroma + ragchain + demo-runner). For CI / integration tests use `docker compose up -d --profile test` instead.
  - `ragchain down` will stop the demo compose stack.
  - A `demo-compose.yml` is included that starts Chroma, the ragchain API, and a small demo runner that performs an example ingest + search; run it with `docker-compose -f demo-compose.yml up --build`.

- The test fixture `chroma_store` will skip remote tests cleanly if no server is reachable and guide you to run `docker compose -f demo-compose.yml --profile test up -d --build`.

---

## 🔧 Notes & Rationale

- Tests prefer deterministic behavior (e.g., `DummyEmbedding`) to avoid network/third-party flakiness.
- The Chroma adapter is split to make it easy to run in-process persistent stores for local dev and remote HTTP servers in CI.
- The codebase uses small, well-scoped threadpools to bridge blocking SDKs into async code.
