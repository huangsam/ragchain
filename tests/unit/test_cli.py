"""Unit tests for CLI."""

from unittest.mock import patch

from click.testing import CliRunner

from ragchain.cli import cli


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@patch("uvicorn.run")
def test_serve_command(mock_uvicorn_run):
    """Test serve command starts uvicorn."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve"])
    assert result.exit_code == 0
    mock_uvicorn_run.assert_called_once()
