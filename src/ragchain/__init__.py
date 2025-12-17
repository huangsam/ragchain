"""RAGChain: Retrieval-Augmented Generation pipeline using LangChain and Ollama.

A lightweight RAG implementation combining:
- Document loading from TIOBE index and Wikipedia
- Semantic search with Chroma vector store
- LLM-based answer generation using Ollama

Modules:
    loaders: Document loading utilities
    rag: Core RAG orchestration
    api: FastAPI REST endpoints
    cli: Command-line interface
"""
