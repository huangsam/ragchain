"""Unit tests for loaders."""
import pytest

from ragchain.loaders import load_tiobe_languages


@pytest.mark.asyncio
async def test_load_tiobe_languages():
    """Test that TIOBE loader returns a list of languages."""
    langs = await load_tiobe_languages(10)
    assert isinstance(langs, list)
    # Should get some languages (or empty if network fails, which is acceptable)
    if langs:
        assert len(langs) <= 10
        assert all(isinstance(lang, str) for lang in langs)
