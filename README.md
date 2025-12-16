# ragchain

Lightweight RAG ingestion scaffold — scraper, chunker, embeddings, and a Chroma adapter.

## Quick start

1. Install deps and dev tools: `uv sync` (or use your virtualenv + pip).
2. Fetch pages: `python main.py --titles "Python_(programming_language)" --save-dir wikipages`.
3. Run tests: `uv run --with-editable . pytest -q`.

### Run the API

```bash
# development server
python -m uvicorn ragchain.api:app --reload --port 8000
# or via CLI after editable install
ragchain serve --port 8000
```

### Chroma & remote tests

- Local Chroma (Docker): `docker-compose up -d` (server defaults to http://localhost:8000).
- Run remote tests: `CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py`

## Notes

- Use `CHROMA_PERSIST_DIRECTORY` for an on-disk in-process Chroma during local runs.
- If `chromadb` isn't available, Chroma-related tests are skipped.
- Recommended Python: **3.12**.

See `AGENTS.md` for a short repo overview and developer notes.
