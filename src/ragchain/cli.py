from __future__ import annotations

import asyncio
import itertools
import logging
import shutil
import subprocess
import sys
import threading

import click
import httpx
import uvicorn

logger = logging.getLogger("ragchain.cli")


class _QuerySpinner:
    """Simple CLI spinner for long-running operations."""

    def __init__(self):
        self.spinner = itertools.cycle(["|", "/", "-", "\\"])
        self.running = False
        self.thread = None

    def start(self):
        """Start the spinner animation."""
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        """Animate the spinner."""
        while self.running:
            sys.stdout.write(f"\r⏳ Querying ragchain... {next(self.spinner)}")
            sys.stdout.flush()
            threading.Event().wait(0.1)

    def stop(self):
        """Stop the spinner and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * 40 + "\r")  # Clear the spinner line
        sys.stdout.flush()


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
        query_text = (
            "Based on the ingested Wikipedia pages, which three programming "
            "languages would you recommend for building a high-performance web backend today, "
            "and what are the trade-offs for each?"
        )
    else:
        query_text = query_type

    async def _run_query():
        spinner = _QuerySpinner()
        spinner.start()
        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{api_url}/ask",
                        json={"query": query_text, "n_results": n_results},
                        timeout=60.0,
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
                    raise click.ClickException(f"Could not connect to {api_url}. Is the ragchain API running? Try: ragchain up")
                except httpx.HTTPStatusError as exc:
                    spinner.stop()
                    raise click.ClickException(f"API error: {exc.status_code} {exc.response.text}")
                except Exception as exc:
                    spinner.stop()
                    raise click.ClickException(f"Query failed: {exc}")
        finally:
            spinner.stop()

    asyncio.run(_run_query())


@cli.command()
@click.option("--api-url", default="http://127.0.0.1:8003", help="API URL")
def status(api_url: str) -> None:  # pragma: no cover - manual
    """Check the `/health` endpoint of the ragchain API and print a short status.

    Example: `ragchain status --api-url http://localhost:8003`
    """
    try:
        resp = httpx.get(f"{api_url}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json() if resp.text else {}
        # Pretty message
        click.echo(f"OK: {data}")
    except httpx.ConnectError:
        raise click.ClickException(f"Could not connect to {api_url}. Is the ragchain API running? Try: ragchain up")
    except httpx.HTTPStatusError as exc:
        raise click.ClickException(f"API error: {exc.response.status_code} {exc.response.text}")
    except Exception as exc:
        raise click.ClickException(f"Health check failed: {exc}")


@cli.command()
@click.argument("titles", nargs=-1, required=True)
@click.option("--api-url", default="http://127.0.0.1:8003", help="API URL")
def ingest(titles: tuple[str, ...], api_url: str) -> None:  # pragma: no cover - manual
    """Ingest Wikipedia pages by title(s) into the vector store.

    Example: `ragchain ingest "Python_(programming_language)" "Java_(programming_language)"`
    """
    async def _run_ingest():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_url}/ingest",
                    json={"titles": list(titles)},
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()

                click.echo("\n" + "=" * 80)
                click.echo("Ingest Report:")
                click.echo("-" * 80)
                for key, val in result.get("report", {}).items():
                    click.echo(f"  {key}: {val}")
                click.echo("=" * 80 + "\n")

        except httpx.ConnectError:
            raise click.ClickException(f"Could not connect to {api_url}. Is the ragchain API running? Try: ragchain up")
        except httpx.HTTPStatusError as exc:
            raise click.ClickException(f"API error: {exc.status_code} {exc.response.text}")
        except Exception as exc:
            raise click.ClickException(f"Ingest failed: {exc}")

    asyncio.run(_run_ingest())


@cli.command()
@click.argument("query_text")
@click.option("--api-url", default="http://127.0.0.1:8003", help="API URL")
@click.option("--n-results", default=3, help="Number of results to return")
def search(query_text: str, api_url: str, n_results: int) -> None:  # pragma: no cover - manual
    """Search the vector store for semantically similar documents.

    Example: `ragchain search "python language" --n-results 5`
    """
    async def _run_search():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_url}/search",
                    json={"query": query_text, "n_results": n_results},
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                click.echo("\n" + "=" * 80)
                click.echo(f"Search Results for: {query_text}")
                click.echo("-" * 80)

                results = result.get("results", {})
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                    click.echo(f"\n[{i}] {meta.get('title', 'Unknown')}")
                    click.echo(f"    {doc[:200]}...")

                click.echo("\n" + "=" * 80 + "\n")

        except httpx.ConnectError:
            raise click.ClickException(f"Could not connect to {api_url}. Is the ragchain API running? Try: ragchain up")
        except httpx.HTTPStatusError as exc:
            raise click.ClickException(f"API error: {exc.status_code} {exc.response.text}")
        except Exception as exc:
            raise click.ClickException(f"Search failed: {exc}")

    asyncio.run(_run_search())


if __name__ == "__main__":  # pragma: no cover - manual run
    cli()
