# ragchain

Lightweight RAG ingestion scaffold:

- Async Wikipedia scraper (REST API first, then mobile-sections HTML parsed via BeautifulSoup)
- Local, non-paid embedding support (dummy embedding or local sentence-transformers)
- Chunking utilities and ingest orchestration
- Unit tests via pytest + pytest-asyncio
- Docker Compose to run MongoDB locally

Quick start

1. Create a virtualenv and install deps from `pyproject.toml`.
2. Run `python main.py --titles "Python_(programming_language)" --save-dir wikipages` to fetch a page.
3. Run `docker compose up -d` to start Mongo (for later integration tests / vectorstore).

See `src/ragchain/` for implementation details.

Chroma notes

- We provide a `ChromaVectorStore` adapter at `src/ragchain/vectorstore/chroma_vectorstore.py` for local development.
- To enable Chroma functionality, install the dependency: `uv add chromadb`.
- Some environments may need `pydantic-settings` installed due to pydantic v2 changes: `uv add pydantic-settings`.
- If your environment still raises errors when importing `chromadb`, the tests that exercise Chroma will be skipped; consult the error message and consider pinning `pydantic` or installing `pydantic-settings`.
