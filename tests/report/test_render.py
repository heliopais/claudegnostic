"""Smoke + empty-data tests for the report renderer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from claudegnostic.cli import app
from claudegnostic.report.filters import filter_by_cwd, parse_duration, since_from
from claudegnostic.report.render import default_out_path, render_report
from claudegnostic.schema import apply_schema


@pytest.fixture
def seeded_db_path(seeded_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> Path:
    """Copy the seeded in-memory DB to a real file the report can re-open read-only."""
    target = tmp_path / "stats.duckdb"
    # ATTACH does not accept bound parameters; the path is test-local, not user input.
    escaped = str(target).replace("'", "''")
    seeded_db.execute(f"ATTACH '{escaped}' AS dest")
    seeded_db.execute("COPY FROM DATABASE memory TO dest")
    seeded_db.execute("DETACH dest")
    return target


@pytest.fixture
def empty_db_path(tmp_path: Path) -> Path:
    target = tmp_path / "empty.duckdb"
    conn = duckdb.connect(str(target))
    try:
        apply_schema(conn)
    finally:
        conn.close()
    return target


def test_parse_duration_handles_units() -> None:
    assert parse_duration("30d").days == 30
    assert parse_duration("12h").total_seconds() == 12 * 3600
    assert parse_duration("2w").days == 14


def test_parse_duration_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_duration("nope")


def test_since_from_returns_none_when_unset() -> None:
    now = datetime(2026, 6, 12, tzinfo=timezone.utc)
    assert since_from(now, None) is None


def test_filter_by_cwd_skips_when_column_absent() -> None:
    import polars as pl

    df = pl.DataFrame({"a": [1, 2]})
    assert filter_by_cwd(df, "proj").equals(df)


def test_default_out_path_includes_date() -> None:
    now = datetime(2026, 6, 12, 10, 0, 0)
    out = default_out_path(now=now)
    assert out.name == "claudegnostic-report-2026-06-12.html"


def test_render_report_smoke(seeded_db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    written = render_report(seeded_db_path, out_path=out)

    assert written == out
    assert out.exists()
    html = out.read_text(encoding="utf-8")

    # Section headings rendered.
    assert "Cost" in html
    assert "Productivity" in html
    assert "Workflow archaeology" in html

    # At least one chart embedded as a base64 PNG.
    assert "data:image/png;base64," in html

    # Top-cost table rendered with at least one session row.
    assert "<table>" in html
    assert "<td>" in html


def test_render_report_on_empty_db(empty_db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "empty-report.html"
    render_report(empty_db_path, out_path=out)

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # No turns banner + all sections fall through to the "not enough data" notice.
    assert "No turns in the database" in html
    assert "Not enough data" in html
    # No base64 PNG was embedded.
    assert "data:image/png;base64," not in html


def test_render_report_with_project_filter(seeded_db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "filtered.html"
    render_report(seeded_db_path, out_path=out, project="/proj/a")

    html = out.read_text(encoding="utf-8")
    assert "/proj/a" in html
    # Project /proj/b's cwd should not appear in the top-cost table.
    assert "/proj/b" not in html


def test_render_report_with_since_zero_keeps_everything(
    seeded_db_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "since.html"
    # 9999d ago — should be effectively unbounded.
    render_report(seeded_db_path, out_path=out, since="9999d")
    assert "data:image/png;base64," in out.read_text(encoding="utf-8")


def test_report_cli_end_to_end(
    runner: CliRunner, seeded_db_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "cli-report.html"
    result = runner.invoke(
        app,
        [
            "report",
            "--db",
            str(seeded_db_path),
            "--out",
            str(out),
            "--top",
            "5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Report summary" in result.stdout
    assert out.exists()
    assert "data:image/png;base64," in out.read_text(encoding="utf-8")


def test_report_cli_on_empty_db(runner: CliRunner, empty_db_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "empty-cli.html"
    result = runner.invoke(
        app,
        ["report", "--db", str(empty_db_path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
