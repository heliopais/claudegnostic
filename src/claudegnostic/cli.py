"""Typer CLI entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from claudegnostic import __version__
from claudegnostic.ingest import default_sessions_root, ingest_root
from claudegnostic.storage import connect, default_db_path

if TYPE_CHECKING:
    import duckdb

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
def ingest(
    root: Path = typer.Option(
        None,
        "--root",
        help="Sessions root to scan (default: ~/.claude/projects).",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    db: Path = typer.Option(
        None,
        "--db",
        help="DuckDB path (default: XDG_DATA_HOME/claudegnostic/stats.duckdb).",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Discover Claude Code session JSONL files and ingest them into DuckDB."""
    actual_root = root if root is not None else default_sessions_root()
    actual_db = db if db is not None else default_db_path()

    started = time.perf_counter()
    with connect(actual_db) as conn:
        report = ingest_root(conn, actual_root)
    elapsed = time.perf_counter() - started

    table = Table(title="Ingest summary", show_header=False, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Root", str(actual_root))
    table.add_row("Database", str(actual_db))
    table.add_row("Files scanned", str(report.files_scanned))
    table.add_row("Files failed", str(len(report.files_failed)))
    table.add_row("Turns added", str(report.turns_added))
    table.add_row("Turns updated", str(report.turns_updated))
    table.add_row("Sessions touched", str(report.sessions_touched))
    table.add_row("Elapsed", f"{elapsed:.2f}s")
    console.print(table)

    if report.files_failed:
        console.print(
            f"[yellow]Warning:[/yellow] {len(report.files_failed)} file(s) failed to parse.",
        )


def _render_top_sessions(conn: duckdb.DuckDBPyConnection, top: int) -> Table:
    rows = conn.execute(
        """
        SELECT session_id, cwd, git_branch, turn_count,
               total_output_tokens, total_input_tokens
        FROM sessions
        ORDER BY total_output_tokens DESC NULLS LAST
        LIMIT ?
        """,
        [top],
    ).fetchall()

    table = Table(title=f"Top {top} sessions by output tokens")
    table.add_column("Session", style="cyan", no_wrap=True)
    table.add_column("cwd", overflow="fold")
    table.add_column("Branch")
    table.add_column("Turns", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Input", justify="right")
    for sid, cwd, branch, turns, out, inp in rows:
        table.add_row(
            (sid or "")[:8],
            cwd or "",
            branch or "",
            str(turns or 0),
            f"{out or 0:,}",
            f"{inp or 0:,}",
        )
    return table


def _render_cache_ratio(conn: duckdb.DuckDBPyConnection) -> Table:
    row = conn.execute(
        """
        SELECT
            count(*) FILTER (WHERE cache_hit_ratio IS NOT NULL) AS n,
            quantile_cont(cache_hit_ratio, 0.10) AS p10,
            quantile_cont(cache_hit_ratio, 0.50) AS p50,
            quantile_cont(cache_hit_ratio, 0.90) AS p90,
            avg(cache_hit_ratio) AS mean
        FROM sessions
        """
    ).fetchone()

    table = Table(title="Cache hit ratio")
    table.add_column("Stat", style="cyan")
    table.add_column("Value", justify="right")
    if row is None or not row[0]:
        table.add_row("Sessions", "0")
        return table
    n, p10, p50, p90, mean = row
    table.add_row("Sessions", str(n))
    table.add_row("p10", f"{p10:.3f}" if p10 is not None else "—")
    table.add_row("p50", f"{p50:.3f}" if p50 is not None else "—")
    table.add_row("p90", f"{p90:.3f}" if p90 is not None else "—")
    table.add_row("mean", f"{mean:.3f}" if mean is not None else "—")
    return table


def _render_tool_leaderboard(conn: duckdb.DuckDBPyConnection, top: int) -> Table:
    rows = conn.execute(
        """
        WITH exploded AS (
            SELECT unnest(tool_names) AS name, tool_result_bytes
            FROM turns
            WHERE tool_names IS NOT NULL AND len(tool_names) > 0
        )
        SELECT name,
               count(*) AS call_count,
               sum(coalesce(tool_result_bytes, 0)) AS total_bytes
        FROM exploded
        GROUP BY name
        ORDER BY call_count DESC
        LIMIT ?
        """,
        [top],
    ).fetchall()

    table = Table(title=f"Top {top} tools by call count")
    table.add_column("Tool", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Result bytes", justify="right")
    for name, calls, total_bytes in rows:
        table.add_row(name or "—", f"{calls:,}", f"{total_bytes or 0:,}")
    return table


def _render_compaction_summary(conn: duckdb.DuckDBPyConnection) -> Table:
    row = conn.execute(
        """
        SELECT
            count(*) AS sessions,
            sum(turn_count) AS turns,
            sum(sidechain_turn_count) AS sidechain_turns,
            sum(compaction_count) AS compactions,
            count(*) FILTER (WHERE compaction_count > 0) AS sessions_with_compaction,
            count(*) FILTER (WHERE sidechain_turn_count > 0) AS sessions_with_sidechain
        FROM sessions
        """
    ).fetchone()

    table = Table(title="Compaction summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    if row is None:
        return table
    sessions, turns, sidechain, compactions, s_with_c, s_with_sc = row
    table.add_row("Sessions", str(sessions or 0))
    table.add_row("Turns", str(turns or 0))
    table.add_row("Sidechain turns", str(sidechain or 0))
    table.add_row("Compactions", str(compactions or 0))
    table.add_row("Sessions with compaction", str(s_with_c or 0))
    table.add_row("Sessions with sidechain", str(s_with_sc or 0))
    return table


@app.command()
def stats(
    db: Path = typer.Option(
        None,
        "--db",
        help="DuckDB path (default: XDG_DATA_HOME/claudegnostic/stats.duckdb).",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    top: int = typer.Option(10, "--top", "-n", help="Number of rows in 'top-N' tables."),
) -> None:
    """Print summary statistics from the ingested DuckDB."""
    actual_db = db if db is not None else default_db_path()

    with connect(actual_db) as conn:
        console.print(_render_top_sessions(conn, top))
        console.print(_render_cache_ratio(conn))
        console.print(_render_tool_leaderboard(conn, top))
        console.print(_render_compaction_summary(conn))
