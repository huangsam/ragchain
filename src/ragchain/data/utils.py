"""Utility functions for the ragchain package."""

import logging
import time
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)


def log_with_prefix(logger: logging.Logger, level: int, prefix: str, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a message with a prefix.

    Args:
        logger: The logger to use.
        level: The logging level (e.g., logging.DEBUG).
        prefix: The prefix to add to the message.
        message: The log message.
        *args: Additional arguments for the logger.
        **kwargs: Additional keyword arguments for the logger.
    """
    logger.log(level, f"[{prefix}] {message}", *args, **kwargs)


def log_timing(logger: logging.Logger, prefix: str, start_time: float, message: str, *args: Any, **kwargs: Any) -> None:
    """Log a timing message with elapsed time.

    Args:
        logger: The logger to use.
        prefix: The prefix for the log message.
        start_time: The start time (from time.time() or time.perf_counter()).
        message: The base message to log.
        *args: Additional arguments for the logger.
        **kwargs: Additional keyword arguments for the logger.
    """
    elapsed = time.time() - start_time
    log_with_prefix(logger, logging.DEBUG, prefix, f"{message} in {elapsed:.2f}s", *args, **kwargs)
