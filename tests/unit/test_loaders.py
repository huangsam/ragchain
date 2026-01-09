"""Unit tests for loaders."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from ragchain.ingestion.loaders import load_conceptual_pages, load_tiobe_languages


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


@patch("langchain_community.document_loaders.WikipediaLoader")
def test_load_topic_page_success(mock_loader_class):
    """Test loading a topic page successfully."""
    from ragchain.ingestion.loaders import _load_topic_page

    # Mock Wikipedia loader
    mock_loader = MagicMock()
    mock_doc = Document(page_content="Compiler content", metadata={})
    mock_loader.load.return_value = [mock_doc]
    mock_loader_class.return_value = mock_loader

    result = _load_topic_page("Compiler")

    assert result is not None
    assert result.page_content == "Compiler content"
    assert result.metadata["category"] == "concept"
    assert result.metadata["topic"] == "Compiler"
    mock_loader.load.assert_called_once()


@patch("langchain_community.document_loaders.WikipediaLoader")
def test_load_topic_page_not_found(mock_loader_class):
    """Test loading a topic page when it doesn't exist."""
    from ragchain.ingestion.loaders import _load_topic_page

    # Mock Wikipedia loader returning empty list
    mock_loader = MagicMock()
    mock_loader.load.return_value = []
    mock_loader_class.return_value = mock_loader

    result = _load_topic_page("NonexistentTopic")

    assert result is None


@patch("time.sleep")
@patch("langchain_community.document_loaders.WikipediaLoader")
def test_load_topic_page_retry_on_error(mock_loader_class, mock_sleep):
    """Test that topic loader retries on failure."""
    from ragchain.ingestion.loaders import _load_topic_page

    # Mock Wikipedia loader to fail twice, then succeed
    mock_loader = MagicMock()
    mock_doc = Document(page_content="Interpreter content", metadata={})
    mock_loader.load.side_effect = [Exception("Network error"), Exception("Timeout"), [mock_doc]]
    mock_loader_class.return_value = mock_loader

    result = _load_topic_page("Interpreter (computing)", retries=2)

    assert result is not None
    assert result.metadata["category"] == "concept"
    assert mock_loader.load.call_count == 3  # 1 initial + 2 retries
    assert mock_sleep.call_count == 2  # Verify sleep was called for retries


@pytest.mark.asyncio
@patch("ragchain.ingestion.loaders._load_topic_page")
async def test_load_conceptual_pages_success(mock_load_topic):
    """Test loading all conceptual pages."""
    # Mock successful loading
    mock_doc1 = Document(page_content="Compiler", metadata={"category": "concept", "topic": "Compiler"})
    mock_doc2 = Document(page_content="Interpreter", metadata={"category": "concept", "topic": "Interpreter (computing)"})
    mock_load_topic.side_effect = [mock_doc1, mock_doc2, None, None, None, None, None, None, None, None]

    result = await load_conceptual_pages()

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].metadata["topic"] == "Compiler"
    assert result[1].metadata["topic"] == "Interpreter (computing)"


@pytest.mark.asyncio
@patch("ragchain.ingestion.loaders._load_topic_page")
async def test_load_conceptual_pages_partial_failure(mock_load_topic):
    """Test loading conceptual pages with some failures."""
    # Mock some successful, some failed loads
    mock_doc = Document(page_content="Type system", metadata={"category": "concept", "topic": "Type system"})
    mock_load_topic.side_effect = [
        Exception("Error"),  # Fail
        mock_doc,  # Success
        None,  # Skip
        Exception("Error"),  # Fail
        None,  # Skip
        None,
        None,
        None,
        None,
        None,
    ]

    result = await load_conceptual_pages()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].metadata["topic"] == "Type system"
