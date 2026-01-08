"""Unit tests for configuration."""

import os
from unittest.mock import patch

from ragchain.config import Config, config


def test_config_singleton():
    """Test that config is a singleton."""
    config2 = Config()
    assert config is config2


def test_config_defaults():
    """Test default configuration values."""
    # Test that config has expected attributes
    assert hasattr(config, "ollama_model")
    assert hasattr(config, "ollama_embed_model")
    assert hasattr(config, "ollama_base_url")
    assert hasattr(config, "chroma_server_url")
    assert hasattr(config, "chroma_persist_directory")
    assert hasattr(config, "enable_intent_routing")


@patch.dict(
    os.environ,
    {
        "OLLAMA_MODEL": "test-model",
        "OLLAMA_EMBED_MODEL": "test-embed",
        "OLLAMA_BASE_URL": "http://test:8080",
        "CHROMA_SERVER_URL": "http://test:8000",
        "CHROMA_PERSIST_DIRECTORY": "/tmp/test",
        "ENABLE_INTENT_ROUTING": "false",
    },
)
def test_config_env_vars():
    """Test configuration reads environment variables."""
    # Reset singleton to pick up new env vars
    Config._instance = None
    test_config = Config()

    assert test_config.ollama_model == "test-model"
    assert test_config.ollama_embed_model == "test-embed"
    assert test_config.ollama_base_url == "http://test:8080"
    assert test_config.chroma_server_url == "http://test:8000"
    assert test_config.chroma_persist_directory == "/tmp/test"
    assert test_config.enable_intent_routing is False
