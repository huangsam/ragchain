# ragchain

Lightweight RAG ingestion scaffold:

- Async Wikipedia scraper (REST API first, then mobile-sections HTML parsed via BeautifulSoup)
- Local, non-paid embedding support (dummy embedding or local sentence-transformers)
- Chunking utilities and ingest orchestration
- Unit tests via pytest + pytest-asyncio
- Docker Compose to run ChromaDB locally (replaces Mongo)

Quick start

1. Create a virtualenv and install deps from `pyproject.toml`.
2. Run `python main.py --titles "Python_(programming_language)" --save-dir wikipages` to fetch a page.
3. Run `docker compose up -d` to start Mongo (for later integration tests / vectorstore).

See `src/ragchain/` for implementation details.

Chroma notes

- We provide a `ChromaVectorStore` adapter at `src/ragchain/vectorstore/chroma_vectorstore.py` for local development.
- To enable Chroma functionality, install the dependency: `uv add chromadb`.
- Some environments may need `pydantic-settings` installed due to Pydantic v2 changes: `uv add pydantic-settings`.
- If your environment still raises errors when importing `chromadb`, the tests that exercise Chroma will be skipped; consult the error message and consider pinning `pydantic` or installing `pydantic-settings`.

Running remote integration tests

If you want to run the integration tests against a running Chroma server locally, start the service with Docker Compose and point `CHROMA_SERVER_URL` at it. The test suite will use `http://localhost:8000` by default when present.

1. Start Chroma locally:

```bash
docker-compose up -d
```

2. Run the remote integration tests:

```bash
CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py
```

If you prefer not to run the server locally, set `CHROMA_SERVER_URL` to a reachable Chroma instance and the tests will use that instead.
