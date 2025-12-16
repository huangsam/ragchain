import pytest
from click.testing import CliRunner

from ragchain.cli import cli


def test_cli_help():
    """Test cli help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_query_help():
    """Test query command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_up_help():
    """Test up command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["up", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_down_help():
    """Test down command help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["down", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_invalid_command():
    """Test CLI with invalid command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["nonexistent"])
    assert result.exit_code != 0
    assert "Error" in result.output or "No such command" in result.output


def test_cli_version():
    """Test if version command exists (if implemented)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    # Should either work or give a helpful error
    assert result.exit_code in [0, 2]  # 0=success, 2=unrecognized option


def test_cli_query_demo_exists():
    """Test that query command exists and shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    # Should show help without error
    assert result.exit_code == 0
    # Should mention something about querying
    assert "query" in result.output.lower()


def test_cli_status_success(monkeypatch):
    """Test `ragchain status` prints OK on healthy API."""
    runner = CliRunner()

    class DummyResp:
        status_code = 200
        text = "{\"status\": \"ok\"}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok"}

    def fake_get(url, timeout=...):
        assert url.endswith("/health")
        return DummyResp()

    monkeypatch.setattr("httpx.get", fake_get)

    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "OK:" in result.output


def test_cli_status_connect_error(monkeypatch):
    """Test `ragchain status` handles connection error gracefully."""
    runner = CliRunner()

    def fake_get(url, timeout=...):
        raise Exception("connection refused")

    monkeypatch.setattr("httpx.get", fake_get)
    result = runner.invoke(cli, ["status"])
    assert result.exit_code != 0
    assert "Could not connect" in result.output or "Health check failed" in result.output
