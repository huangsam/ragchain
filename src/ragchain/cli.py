from __future__ import annotations

import logging

import click
import uvicorn

logger = logging.getLogger("ragchain.cli")


@click.group()
def cli() -> None:  # pragma: no cover - CLI glue
    """ragchain CLI"""


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on", type=int)
@click.option("--reload/--no-reload", default=True, help="Enable/disable auto-reload")
def serve(host: str, port: int, reload: bool) -> None:  # pragma: no cover - runtime
    """Run the API server (uvicorn)"""
    logger.info("Starting ragchain API on %s:%d (reload=%s)", host, port, reload)
    uvicorn.run("ragchain.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover - manual run
    cli()
