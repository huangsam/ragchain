"""Pytest configuration."""
import pytest


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provide a temporary Chroma directory."""
    import os

    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(tmp_path / "chroma")
    yield str(tmp_path / "chroma")
