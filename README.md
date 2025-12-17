# RAGchain

Your local RAG stack — no APIs, no cloud, full control.

**Key Features:**
- Analyze programming languages with TIOBE-ranked Wikipedia articles
- Semantic search with qwen3-embedding (4096-dimensional vectors)
- LLM-powered answer generation via local Ollama
- Docker Compose demo stack
- Full CLI for ingest/search/query workflows

**Motivation:**
- A minimal, self-contained RAG pipeline that runs entirely locally
- Perfect for prototyping and teaching where reproducibility matters

## Quick start

Start the demo stack and interact with the RAG pipeline to analyze programming languages:

```bash
# Start the demo stack (Chroma + ragchain API + demo-runner)
docker compose --profile demo up --build -d

# Search ingested programming language data
ragchain search "functional programming paradigm" --k 4
ragchain search "memory management" --k 5

# Ask questions using RAG (requires local Ollama)
# Ensure you have run `ollama pull qwen3` locally first
ragchain ask "What is Python used for?"
ragchain ask "Compare Go and Rust for systems programming"

# Or manually ingest a different set of languages
ragchain ingest --n 10  # Fetches top 10 from TIOBE

# Stop the stack
docker compose --profile demo down -v
```

### What's running

- **Chroma** (vector store): http://localhost:8000 — persists programming language embeddings
- **ragchain API**: http://localhost:8003 — REST endpoints for ingest, search, and ask
- **demo-runner**: automatically fetches top 20 TIOBE languages and ingests Wikipedia articles on startup

**Demo Focus:** The current demo is tailored for **programming language analysis**. It:
1. Fetches top languages from the TIOBE index
2. Loads Wikipedia articles for each language
3. Chunks and embeds them with qwen3-embedding (4096-dimensional vectors)
4. Enables semantic search over language documentation
5. Supports RAG queries via local Ollama

## Notes

- Python: **3.12** recommended (LangChain ecosystem has optimized wheels).

**Vector Store Configuration:**
- **CHROMA_PERSIST_DIRECTORY**: use for on-disk in-process Chroma during local runs.
- **CHROMA_SERVER_URL**: point ragchain at a running Chroma instance (e.g., from `ragchain up`).

**LLM Configuration:**
- **OLLAMA_BASE_URL**: configure where Ollama is running (default: `http://localhost:11434`).
- **OLLAMA_EMBED_MODEL**: embedding model (default: `qwen3-embedding`).
- **OLLAMA_MODEL**: LLM model for generation (ensure pulled: `ollama pull qwen3`).

See [AGENTS.md](AGENTS.md) for project layout, tooling, and architecture notes.
