"""Pytest configuration."""

import warnings

import pytest

# Filter out coroutine warnings from mocked CLI tests
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Enable tracemalloc to get the object allocation traceback.*", category=RuntimeWarning)


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provide a temporary Chroma directory."""
    import os

    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(tmp_path / "chroma")
    yield str(tmp_path / "chroma")
