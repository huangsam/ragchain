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

        # Performance optimization flags
        self.enable_grading: bool = os.environ.get("ENABLE_GRADING", "false").lower() == "true"
        self.enable_intent_routing: bool = os.environ.get("ENABLE_INTENT_ROUTING", "true").lower() == "true"

        # CLI configuration
        self.ragchain_api_url: str = os.environ.get("RAGCHAIN_API_URL", "http://localhost:8003")


# Global singleton instance
config = Config()
