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
        self.ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_embed_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
        self.ollama_model: str = os.environ.get("OLLAMA_MODEL", "deepseek-r1")
        # Embedding context: chunks are ~700 tokens, 4096 provides headroom
        self.ollama_embed_ctx: int = int(os.environ.get("OLLAMA_EMBED_CTX", "4096"))
        # Generation context: needs to fit 12 docs × ~800 tokens = ~10k tokens
        self.ollama_gen_ctx: int = int(os.environ.get("OLLAMA_GEN_CTX", "8192"))

        # Document chunking configuration
        self.chunk_size: int = int(os.environ.get("CHUNK_SIZE", "2500"))
        self.chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "500"))

        # Retrieval configuration
        self.retrieval_k: int = int(os.environ.get("RETRIEVAL_K", "10"))  # Number of docs per retriever
        self.retrieval_max_results: int = int(os.environ.get("RETRIEVAL_MAX_RESULTS", "10"))  # Max results after RRF
        self.retrieval_k_adaptive: int = int(os.environ.get("RETRIEVAL_K_ADAPTIVE", "6"))  # For graph adaptive retrieval

        # Performance optimization flags
        self.enable_grading: bool = os.environ.get("ENABLE_GRADING", "true").lower() == "true"
        self.enable_intent_routing: bool = os.environ.get("ENABLE_INTENT_ROUTING", "true").lower() == "true"


# Global singleton instance
config = Config()
