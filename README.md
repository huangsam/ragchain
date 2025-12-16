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

- Local Chroma (Docker): `docker-compose -f test-compose.yml up -d --build` (server defaults to http://localhost:8000).

- To ingest pages AND persist them to the running Chroma instance (recommended newcomer flow):

  1. Start Chroma: `docker-compose -f test-compose.yml up -d --build` (or `ragchain up`)
  2. Export the server URL: `export CHROMA_SERVER_URL=http://localhost:8000`
  3. Run the ingest CLI (example): `python main.py --titles "Python_(programming_language)" --save-dir wikipages`

- Convenience: use the CLI to manage local Chroma

  - Start services: `ragchain up` (shorthand for `docker-compose up -d`)
  - Stop services: `ragchain down` (shorthand for `docker-compose down`)
  - Remove volumes: `ragchain down --remove-volumes`

- Demo compose file

  - A `demo-compose.yml` is provided to bring up a demo stack (Chroma, the ragchain API, and a demo client that runs ingest & search). Start it with:

    ```bash
    docker compose -f demo-compose.yml up --build
    # or
    docker-compose -f demo-compose.yml up --build
    ```

  - The demo-runner will wait for the API and perform a sample ingest + search, printing results to the demo container logs.

- Run remote tests: `CHROMA_SERVER_URL=http://localhost:8000 uv run --with-editable . pytest tests/integration/test_full_pipeline.py`

## Notes

- Use `CHROMA_PERSIST_DIRECTORY` for an on-disk in-process Chroma during local runs.
- If `chromadb` isn't available, Chroma-related tests are skipped.
- Recommended Python: **3.12**.

See `AGENTS.md` for a short repo overview and developer notes.
