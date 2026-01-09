"""Unit tests for utilities."""

import logging

from ragchain.utils import get_logger


def test_get_logger():
    """Test get_logger returns a logger instance."""
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test"
