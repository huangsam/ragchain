from __future__ import annotations

import logging

import click
import shutil
import subprocess
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
@click.option("--remove-volumes/--no-remove-volumes", default=False, help="Remove volumes with -v")
@click.option("--remove-orphans/--no-remove-orphans", default=False, help="Pass --remove-orphans to down")
@click.option("--profile", default="demo", help="Compose profile to stop (demo|test)")
def down(remove_volumes: bool, remove_orphans: bool, profile: str) -> None:  # pragma: no cover - manual
    """Stop and remove Docker Compose services started by `ragchain up` (demo-compose.yml)."""
    cmd = _compose_cmd() + ["-f", "docker-compose.yml", "--profile", profile, "down"]
    if remove_volumes:
        cmd.append("-v")
    if remove_orphans:
        cmd.append("--remove-orphans")

    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=".")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
        raise click.ClickException(f"docker-compose down failed: {exc}")

    click.echo("✓ Demo stack stopped and removed.")

if __name__ == "__main__":  # pragma: no cover - manual run
    cli()
