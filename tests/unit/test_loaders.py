"""Unit tests for loaders."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragchain.ingestion.loaders import load_tiobe_languages


@pytest.mark.asyncio
@patch("ragchain.ingestion.loaders.aiohttp.ClientSession")
async def test_load_tiobe_languages(mock_session_class):
    """Test that TIOBE loader returns a list of languages."""
    # Mock the HTTP response
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.text = AsyncMock(
        return_value="""
    <html>
        <body>
            <table id="top20">
                <tr><th>Header</th></tr>
                <tr><td></td><td></td><td></td><td></td><td>Python</td></tr>
                <tr><td></td><td></td><td></td><td></td><td>Java</td></tr>
                <tr><td></td><td></td><td></td><td></td><td>JavaScript</td></tr>
            </table>
        </body>
    </html>
    """
    )
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session_class.return_value.__aenter__.return_value = mock_session

    langs = await load_tiobe_languages(10)
    assert isinstance(langs, list)
    assert len(langs) == 3
    assert langs == ["Python", "Java", "JavaScript"]


@pytest.mark.asyncio
@patch("ragchain.ingestion.loaders.aiohttp.ClientSession")
async def test_load_tiobe_languages_network_error(mock_session_class):
    """Test TIOBE loader handles network errors gracefully."""
    # Mock a network error
    mock_session_class.return_value.__aenter__.side_effect = Exception("Network error")

    langs = await load_tiobe_languages(10)
    assert isinstance(langs, list)
    assert len(langs) == 0  # Should return empty list on error
