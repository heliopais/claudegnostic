"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from claudegnostic import __version__

app = typer.Typer(
    name="claudegnostic",
    help="Diagnostics and statistics for Claude Code sessions.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"claudegnostic {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Diagnostics and statistics for Claude Code sessions."""


@app.command()
def hello(name: str = "world") -> None:
    """Placeholder command — confirms the CLI is wired up."""
    console.print(f"Hello, [bold cyan]{name}[/bold cyan]!")
