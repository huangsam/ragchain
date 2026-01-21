"""Centralized configuration management for RAGChain."""

import os
from typing import Optional


class Config:
    """Singleton configuration class for all environment variables."""

    _instance: Optional["Config"] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Vector store configuration
        self.chroma_persist_directory: str = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        self.chroma_server_url: str = os.environ.get("CHROMA_SERVER_URL", "http://localhost:8000")

        # Ollama configuration
        # We use qwen3-embedding:4b for embeddings and qwen3:8b for generation
        # Embedding context: chunks are ~700 tokens, 4096 provides headroom
        # Generation context: needs to fit 12 docs × ~800 tokens = ~10k tokens
        self.ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_embed_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
        self.ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
        self.ollama_embed_ctx: int = int(os.environ.get("OLLAMA_EMBED_CTX", "4096"))
        self.ollama_gen_ctx: int = int(os.environ.get("OLLAMA_GEN_CTX", "8192"))
        # Smaller context windows for lightweight LLM tasks (routing, judging, rewriting)
        self.ollama_routing_ctx: int = int(os.environ.get("OLLAMA_ROUTING_CTX", "2048"))
        self.ollama_judging_ctx: int = int(os.environ.get("OLLAMA_JUDGING_CTX", "4096"))
        self.ollama_rewriting_ctx: int = int(os.environ.get("OLLAMA_REWRITING_CTX", "2048"))

        # Document chunking configuration
        # Chunk size is 2500 characters (625 tokens) with a 20% overlap by default
        self.chunk_size: int = int(os.environ.get("CHUNK_SIZE", "2500"))
        self.chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "500"))

        # Retrieval configuration
        # We define values search_k, graph_k, rrf_max_results which control the documents retrieved
        self.search_k: int = int(os.environ.get("SEARCH_K", "10"))  # For direct search API
        self.graph_k: int = int(os.environ.get("GRAPH_K", "6"))  # For graph-based RAG pipeline
        self.rrf_max_results: int = int(os.environ.get("RRF_MAX_RESULTS", "10"))  # Max results after RRF

        # Performance optimization flags
        # Enable or disable parts of the RAG graph for performance reasons
        self.enable_grading: bool = os.environ.get("ENABLE_GRADING", "true").lower() == "true"
        self.enable_intent_routing: bool = os.environ.get("ENABLE_INTENT_ROUTING", "true").lower() == "true"


# Global singleton instance
config = Config()
