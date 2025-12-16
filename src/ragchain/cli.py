from __future__ import annotations

import asyncio
import json
import logging
import threading
import itertools
import sys

import click
import shutil
import subprocess
import uvicorn
import httpx

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


def _compose_cmd() -> list[str]:
    """Return a docker-compose command list, preferring the classic
    `docker-compose` binary and falling back to `docker compose` if available.
    """
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    if shutil.which("docker"):
        # Use the docker CLI with the compose subcommand
        return ["docker", "compose"]
    raise click.ClickException("docker-compose or docker CLI not found on PATH")


@cli.command()
@click.option("--detached/--no-detached", default=True, help="Run compose up detached")
@click.option("--build/--no-build", default=True, help="Pass --build to `up`")
@click.option("--profile", default="demo", help="Compose profile to start (demo|test)")
def up(detached: bool, build: bool, profile: str) -> None:  # pragma: no cover - manual
    """Start the local Docker Compose services (Chroma + ragchain + demo-runner).

    This uses `demo-compose.yml` by default with the `demo` profile. Use
    `--profile test` for a minimal stack suitable for integration tests.
    """
    cmd = _compose_cmd() + ["-f", "docker-compose.yml", "--profile", profile, "up"]
    if detached:
        cmd.append("-d")
    if build:
        cmd.append("--build")

    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=".")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
        raise click.ClickException(f"docker-compose up failed: {exc}")

    click.echo(f"✓ Demo stack started (profile={profile}).")
    click.echo("  Tip: export CHROMA_SERVER_URL=http://localhost:8000 for local ragchain server")


@cli.command()
@click.option("--volumes/--no-volumes", default=False, help="Remove named volumes (wipes data)")
@click.option("--remove-orphans/--no-remove-orphans", default=False, help="Pass --remove-orphans to down")
@click.option("--profile", default="demo", help="Compose profile to stop (demo|test)")
def down(volumes: bool, remove_orphans: bool, profile: str) -> None:  # pragma: no cover - manual
    """Stop and remove Docker Compose services started by `ragchain up` (demo-compose.yml)."""
    cmd = _compose_cmd() + ["-f", "docker-compose.yml", "--profile", profile, "down"]
    if volumes:
        cmd.append("-v")
    if remove_orphans:
        cmd.append("--remove-orphans")

    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=".")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
        raise click.ClickException(f"docker-compose down failed: {exc}")

    click.echo("✓ Demo stack stopped and removed.")


@cli.command()
@click.argument("query_type", default="demo", required=False)
@click.option("--api-url", default="http://127.0.0.1:8003", help="API URL")
@click.option("--n-results", default=6, help="Number of context chunks to retrieve")
def query(query_type: str, api_url: str, n_results: int) -> None:  # pragma: no cover - manual
    """Query the ragchain RAG system.

    Use 'demo' for the pre-set insightful query, or provide your own query text.
    """
    # Map 'demo' to the preset query
    if query_type.lower() == "demo":
        query_text = "Based on the ingested Wikipedia pages, which three programming languages would you recommend for building a high-performance web backend today, and what are the trade-offs for each?"
    else:
        query_text = query_type

    class Spinner:
        def __init__(self):
            self.spinner = itertools.cycle(["|", "/", "-", "\\"])
            self.running = False
            self.thread = None

        def start(self):
            self.running = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()

        def _spin(self):
            while self.running:
                sys.stdout.write(f"\r⏳ Querying ragchain... {next(self.spinner)}")
                sys.stdout.flush()
                threading.Event().wait(0.1)

        def stop(self):
            self.running = False
            if self.thread:
                self.thread.join()
            sys.stdout.write("\r" + " " * 40 + "\r")  # Clear the spinner line
            sys.stdout.flush()

    async def _run_query():
        spinner = Spinner()
        spinner.start()
        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{api_url}/ask",
                        json={"query": query_text, "n_results": n_results},
                        timeout=60.0
                    )
                    response.raise_for_status()
                    result = response.json()

                    spinner.stop()

                    # Pretty-print the result
                    click.echo("\n" + "=" * 80)
                    click.echo(f"Query: {query_text}\n")
                    click.echo("Answer:")
                    click.echo("-" * 80)
                    click.echo(result["answer"])
                    click.echo("\n" + "-" * 80)
                    click.echo(f"Model: {result['model']}")
                    click.echo(f"Context chunks retrieved: {len(result['context'])}")
                    click.echo("=" * 80 + "\n")

                except httpx.ConnectError:
                    spinner.stop()
                    raise click.ClickException(
                        f"Could not connect to {api_url}. Is the ragchain API running? "
                        "Try: ragchain up"
                    )
                except httpx.HTTPStatusError as exc:
                    spinner.stop()
                    raise click.ClickException(f"API error: {exc.status_code} {exc.response.text}")
                except Exception as exc:
                    spinner.stop()
                    raise click.ClickException(f"Query failed: {exc}")
        finally:
            spinner.stop()

    asyncio.run(_run_query())


if __name__ == "__main__":  # pragma: no cover - manual run
    cli()
