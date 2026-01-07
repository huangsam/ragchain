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
# 1. Start Chroma vector database
docker compose up -d

# 2. Ingest programming language documents
ragchain ingest --n 50

# 3. Search ingested documents
ragchain search "functional programming paradigm" --k 4
ragchain search "memory management" --k 5

# 4. Ask questions with RAG + LLM
ragchain ask "What is Python used for?"
ragchain ask "Compare Go and Rust for systems programming"
ragchain ask "What are the top 10 most popular languages?"

# 5. Evaluate RAG quality with LLM-as-judge
ragchain evaluate

# Clean up
docker compose down -v
```

**Requirements:**
- Docker (for Chroma)
- Ollama with `bge-m3` (embeddings) and `deepseek-r1` (generation)
- Python 3.12+

## Intent-Based Retrieval

The `ragchain ask` command adapts to query type:

| Type | Example | Strategy |
|---|---|---|
| FACT | "Top 10 languages?" | Keyword-heavy for lists |
| CONCEPT | "What is functional programming?" | Balanced search |
| COMPARISON | "Compare Go and Rust" | Semantic-focused |

See [AGENTS.md](AGENTS.md) for architecture.
