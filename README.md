# ragchain

Lightweight RAG ingestion scaffold — scraper, chunker, embeddings, and a Chroma adapter.

---

## 🚀 Quick Start (Users)

Start the demo stack and interact with the API:

```bash
# Start the demo stack (Chroma + ragchain API + demo-runner)
ragchain up

# Health check
curl http://127.0.0.1:8003/health

# Ingest pages
curl -X POST http://127.0.0.1:8003/ingest \
  -H 'Content-Type: application/json' \
  -d '{"titles":["Python_(programming_language)"]}'

# Search
curl -X POST http://127.0.0.1:8003/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"python language","n_results":1}'

# Stop the stack
ragchain down
```

### What's running

- **Chroma** (vector store): http://localhost:8000
- **ragchain API**: http://localhost:8003
- **demo-runner**: automatically runs sample ingest + search on startup

---

## 👨‍💻 Development Setup

```bash
# Install deps and dev tools
uv sync

# Run unit tests
uv run --with-editable . pytest -q

# Run a sample ingest (saves to wikipages/)
python main.py --titles "Python_(programming_language)" --save-dir wikipages

# Start Chroma stack (one terminal)
ragchain up

# In another terminal, start the local API server
ragchain serve --port 8001
# or
uv run --with-editable . python -m uvicorn ragchain.api:app --reload --port 8001

# Test the API
export CHROMA_SERVER_URL=http://localhost:8000
curl -X POST http://127.0.0.1:8001/ingest \
  -H 'Content-Type: application/json' \
  -d '{"titles":["Python_(programming_language)"]}'

# Run integration tests against running Chroma
docker compose -f demo-compose.yml --profile test up -d --build
CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py
```

---

## 📝 Notes

- Python: **3.12** recommended (some deps have optimized wheels).
- **CHROMA_PERSIST_DIRECTORY**: use for on-disk in-process Chroma during local runs.
- **CHROMA_SERVER_URL**: point ragchain at a running Chroma instance (e.g., from `ragchain up`).
- If `chromadb` isn't installed, Chroma-related tests are skipped.

See [AGENTS.md](AGENTS.md) for project layout, tooling, and architecture notes.
