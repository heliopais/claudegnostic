"""Smoke AppTest for the dashboard Productivity page.

Exercises three modes against a real DuckDB file pointed at via
``CLAUDEGNOSTIC_DB_PATH``: no database, empty database, and a seeded
database with three sessions (the minimum to render charts).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from streamlit.testing.v1 import AppTest

from claudegnostic.dashboard.cli import DB_PATH_ENV
from claudegnostic.schema import apply_schema

PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "claudegnostic"
    / "dashboard"
    / "pages"
    / "03_productivity.py"
)


def _seed_three_sessions(db: Path) -> None:
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    # (session, idx, ts, cwd, model, tools, input, output, cache_read, text, wall_ms)
    rows = [
        ("S1", 0, datetime(2026, 6, 10, 9, 0), "/proj/a", "claude-opus-4-7",
         ["Read"], 1000, 200, 4000, 400, 2000),
        # Wasted turn: 20k input, 0 tools, 50 chars out.
        ("S1", 1, datetime(2026, 6, 10, 9, 5), "/proj/a", "claude-opus-4-7",
         [], 20_000, 10, 9000, 50, 8000),
        ("S2", 0, datetime(2026, 6, 11, 12, 0), "/proj/b", "claude-sonnet-4-6",
         ["Bash"], 2000, 500, 1000, 500, 3500),
        ("S3", 0, datetime(2026, 6, 12, 14, 0), "/proj/b", "claude-sonnet-4-6",
         ["Read", "Bash"], 3000, 800, 6000, 800, 4500),
    ]
    for sid, idx, ts, cwd, model, tools, inp, out, cr, txt, wall in rows:
        conn.execute(
            """
            INSERT INTO turns (
                session_id, turn_index, uuid, parent_uuid, timestamp, cwd,
                git_branch, model, is_sidechain, is_compact_summary,
                input_tokens, output_tokens, cache_creation_tokens,
                cache_read_tokens, service_tier, stop_reason, text_output_chars,
                thinking_chars, tool_call_count, tool_names, tool_input_bytes,
                tool_result_bytes, wall_duration_ms
            ) VALUES (?, ?, NULL, NULL, ?, ?, NULL, ?, FALSE, FALSE,
                      ?, ?, 0, ?, NULL, NULL, ?, 0, ?, ?, 0, 0, ?)
            """,
            [sid, idx, ts, cwd, model, inp, out, cr, txt, len(tools), tools, wall],
        )
    for sid, cwd, n in [("S1", "/proj/a", 2), ("S2", "/proj/b", 1), ("S3", "/proj/b", 1)]:
        conn.execute(
            "INSERT INTO sessions (session_id, cwd, turn_count) VALUES (?, ?, ?)",
            [sid, cwd, n],
        )
    conn.close()


@pytest.fixture
def empty_db(tmp_path: Path) -> Path:
    db = tmp_path / "stats.duckdb"
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    conn.close()
    return db


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "stats.duckdb"
    _seed_three_sessions(db)
    return db


def _run(db_path: Path | str, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv(DB_PATH_ENV, str(db_path))
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.run()
    return at


def test_renders_no_db_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(tmp_path / "missing.duckdb", monkeypatch)
    assert not at.exception
    infos = " ".join(el.value for el in at.info)
    assert "No database found" in infos


def test_renders_empty_db_message(empty_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _run(empty_db, monkeypatch)
    assert not at.exception
    infos = " ".join(el.value for el in at.info)
    assert "claudegnostic ingest" in infos


def test_renders_charts_with_three_sessions(
    seeded_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _run(seeded_db, monkeypatch)
    assert not at.exception
    headers = [el.value for el in at.subheader]
    assert "Per-tool wall time" in headers
    assert "Cache hit ratio vs session length" in headers
    assert "Wasted turns" in headers
