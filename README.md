# RAGchain

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/huangsam/ragchain/ci.yml)](https://github.com/huangsam/ragchain/actions)
[![License](https://img.shields.io/github/license/huangsam/ragchain)](https://github.com/huangsam/ragchain/blob/main/LICENSE)

Your local RAG stack — no APIs, no cloud, full control.

**Key Features:**
- Intent-based adaptive RAG with self-correcting retrieval (auto-retry on validation failure)
- Ensemble search via Reciprocal Rank Fusion combining BM25 + semantic vectors
- Local-only: Ollama for embeddings/LLM, Chroma for vector store, no external APIs
- Analyze programming languages via Docker Compose demo stack

## Quick start

```bash
# Start the demo stack
docker compose --profile demo up --build -d

# Search ingested programming language data
ragchain search "functional programming paradigm" --k 4
ragchain search "memory management" --k 5

# Try some queries (requires `deepseek-r1` and `bge-m3` locally)
ragchain ask "What is Python used for?"
ragchain ask "Compare Go and Rust for systems programming"
ragchain ask "What are the key features of functional programming in Haskell?"
ragchain ask "How has Java evolved since its release?"
ragchain ask "What are the main differences between interpreted and compiled languages?"
ragchain ask "Which languages are commonly used for machine learning?"
ragchain ask "What are the top 10 most popular languages?"

# Ingest a different set of languages
ragchain ingest --n 10

# Stop the stack
docker compose --profile demo down -v
```

**What's running:**
- **Chroma** (vector store) at http://localhost:8000
- **ragchain API** at http://localhost:8003
- **demo-runner** ingests top 50 TIOBE languages on startup

## Intent-Based Retrieval

The `/ask` endpoint adapts to query type:

| Type | Example | Strategy |
|---|---|---|
| FACT | "Top 10 languages?" | Keyword-heavy for lists |
| CONCEPT | "What is functional programming?" | Balanced search |
| COMPARISON | "Compare Go and Rust" | Semantic-focused |

See [AGENTS.md](AGENTS.md) for architecture.
