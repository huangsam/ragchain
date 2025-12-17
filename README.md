# RAGchain

Your local RAG stack — no APIs, no cloud, full control.

**Key Features:** Analyze programming languages with TIOBE-ranked Wikipedia articles, semantic search with qwen3-embedding (4096-dimensional vectors), LLM-powered answer generation via local Ollama, Docker Compose demo stack, and a full CLI for ingest/search/query workflows.

**Motivation:** A minimal, self-contained RAG pipeline that runs entirely locally—no external APIs, no cloud dependencies—perfect for prototyping, teaching, and production use cases where data privacy and reproducibility matter.

---

## 🚀 Quick Start (Users)

Start the demo stack and interact with the RAG pipeline to analyze programming languages:

```bash
# Start the demo stack (Chroma + ragchain API + demo-runner)
ragchain up

# The demo-runner automatically ingests the top 20 TIOBE-ranked languages from Wikipedia

# Search ingested programming language data
ragchain search "functional programming paradigm" --k 4
ragchain search "memory management" --k 5

# Ask questions using RAG (requires local Ollama)
# Ensure you have run `ollama pull qwen3` locally first
ragchain query "What is Python used for?"
ragchain query "Compare Go and Rust for systems programming"

# Or manually ingest a different set of languages
ragchain ingest --n 10  # Fetches top 10 from TIOBE

# Stop the stack
ragchain down
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

Note: Use `docker compose up -d --profile demo` or `ragchain up --profile test` (minimal stack for CI). See [AGENTS.md](AGENTS.md) for architecture details.

---

## 📝 Notes

- Python: **3.12** recommended (LangChain ecosystem has optimized wheels).
- **CHROMA_PERSIST_DIRECTORY**: use for on-disk in-process Chroma during local runs.
- **CHROMA_SERVER_URL**: point ragchain at a running Chroma instance (e.g., from `ragchain up`).
- **OLLAMA_BASE_URL**: configure where Ollama is running (default: `http://localhost:11434`).
- **OLLAMA_EMBED_MODEL**: embedding model (default: `qwen3-embedding` for 4096-dimensional vectors).
- **OLLAMA_MODEL**: LLM model for generation (ensure pulled: `ollama pull qwen3`).

See [AGENTS.md](AGENTS.md) for project layout, tooling, and architecture notes.
