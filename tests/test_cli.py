"""Smoke tests for the CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from claudegnostic import __version__
from claudegnostic.cli import app


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "claudegnostic" in result.stdout.lower()


def test_hello_default(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout
    assert "world" in result.stdout


def test_hello_named(runner: CliRunner) -> None:
    result = runner.invoke(app, ["hello", "--name", "claude"])
    assert result.exit_code == 0
    assert "claude" in result.stdout
