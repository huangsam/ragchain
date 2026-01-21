"""Shared utility functions for the ragchain package."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from langchain_ollama import OllamaLLM

from ragchain.config import config


def get_llm(
    *,
    model: str | None = None,
    temperature: float = 0.1,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    purpose: Literal["generation", "routing", "judging", "rewriting"] = "generation",
) -> OllamaLLM:
    """Factory function to create a configured OllamaLLM instance.

    Centralizes LLM configuration with sensible defaults.
    Use the `purpose` parameter to get recommended settings for different use cases.

    Note: For total request time limits, wrap calls with asyncio.wait_for() since
    httpx timeouts don't work well with streaming responses.

    Args:
        model: Ollama model name (default: config.ollama_model)
        temperature: Sampling temperature (default: 0.1 for consistency)
        num_ctx: Context window size (default: varies by purpose)
        num_predict: Max output tokens (default: varies by purpose, prevents runaway generation)
        purpose: Use case hint for default settings:
            - "generation": Standard answer generation (temp=0.1, 1024 tokens max)
            - "routing": Intent classification (temp=0, 32 tokens max)
            - "judging": LLM-as-judge answer evaluation (temp=0, 512 tokens max)
            - "rewriting": Query rewriting (temp=0.5, 128 tokens max)

    Returns:
        Configured OllamaLLM instance
    """
    # Apply purpose-specific defaults (including num_predict to prevent runaway generation)
    purpose_defaults: dict[str, dict[str, Any]] = {
        "generation": {"temperature": 0.1, "num_ctx": config.ollama_gen_ctx, "num_predict": 1024, "reasoning": True},
        "routing": {"temperature": 0.0, "num_ctx": config.ollama_routing_ctx, "num_predict": 32, "reasoning": False},
        "judging": {"temperature": 0.0, "num_ctx": config.ollama_judging_ctx, "num_predict": 512, "reasoning": False},
        "rewriting": {"temperature": 0.5, "num_ctx": config.ollama_rewriting_ctx, "num_predict": 128, "reasoning": False},
    }

    defaults = purpose_defaults.get(purpose, purpose_defaults["generation"])

    return OllamaLLM(
        model=model or config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=temperature if temperature != 0.1 else defaults["temperature"],
        num_ctx=num_ctx or defaults["num_ctx"],
        num_predict=num_predict or defaults["num_predict"],
        reasoning=defaults["reasoning"],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
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


def timed(logger: logging.Logger, prefix: str, level: int = logging.DEBUG) -> Callable:
    """Decorator to log execution time of a function.

    Args:
        logger: The logger to use.
        prefix: The prefix for the log message.
        level: The logging level (default: DEBUG).

    Returns:
        Decorated function that logs elapsed time.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.log(level, f"[{prefix}] {func.__name__} completed in {elapsed:.2f}s")
            return result

        return wrapper

    return decorator
