"""Shared types for the ingestion pipeline."""

from typing_extensions import TypedDict

__all__ = ["IngestResult"]


class IngestResult(TypedDict):
    """Result of document ingestion operation.

    This TypedDict captures the outcome of an ingestion process, including status,
    number of documents ingested, a message, and elapsed time.
    """

    # "SUCCESS" or "FAILURE"
    status: str
    # Number of documents ingested
    count: int
    # Additional information about the ingestion
    message: str
    # Time taken for the ingestion process
    elapsed_seconds: float
