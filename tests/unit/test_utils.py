"""Unit tests for utilities."""

import logging

from ragchain.utils import get_logger, log_timing


def test_get_logger():
    """Test get_logger returns a logger instance."""
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test"


def test_log_timing(caplog):
    """Test log_timing logs elapsed time."""
    import time

    logger = get_logger("test")
    start_time = time.time() - 0.1  # 0.1 seconds ago

    with caplog.at_level(logging.DEBUG):
        log_timing(logger, "TEST", start_time, "Operation completed")

    assert "[TEST] Operation completed in" in caplog.text
    assert "s" in caplog.text
