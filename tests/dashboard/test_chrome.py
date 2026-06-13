"""Smoke tests for the dashboard chrome: state, db helpers, empty-state.

These tests deliberately avoid Streamlit's runtime — they exercise the
plain helpers (``open_conn``, ``compute_filter_options``, ``is_db_empty``)
that the ``@st.cache_resource`` entry points delegate to. We don't try to
unit-test the cache or the sidebar widgets here; Streamlit's own
``AppTest`` would be the right tool for that and is out of scope for the
chrome bead.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from claudegnostic.analysis import archaeology
from claudegnostic.dashboard._db import (
    compute_filter_options,
    is_db_empty,
    open_conn,
)
from claudegnostic.dashboard._empty import DEFAULT_MESSAGE
from claudegnostic.dashboard._state import default_state
from claudegnostic.schema import apply_schema


@pytest.fixture
def empty_db_path(tmp_path: Path) -> Path:
    """Path to a fresh DuckDB file with schema but no rows."""
    db = tmp_path / "stats.duckdb"
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    conn.close()
    return db


@pytest.fixture
def seeded_db_path(tmp_path: Path) -> Path:
    """DuckDB file with two cwds, two models, two dates."""
    db = tmp_path / "stats.duckdb"
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    rows = [
        ("S1", 0, datetime(2026, 6, 10, 9, 0), "/proj/a", "claude-opus-4-7"),
        ("S1", 1, datetime(2026, 6, 10, 9, 5), "/proj/a", "claude-opus-4-7"),
        ("S2", 0, datetime(2026, 6, 11, 12, 0), "/proj/b", "claude-sonnet-4-6"),
    ]
    for session_id, idx, ts, cwd, model in rows:
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
                      0, 0, 0, 0, NULL, NULL, 0, 0, 0, [], 0, 0, NULL)
            """,
            [session_id, idx, ts, cwd, model],
        )
    conn.execute(
        "INSERT INTO sessions (session_id, cwd, turn_count) VALUES ('S1', '/proj/a', 2)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, cwd, turn_count) VALUES ('S2', '/proj/b', 1)"
    )
    conn.close()
    return db


def test_default_state_is_no_op() -> None:
    state = default_state()
    assert state["start_date"] is None
    assert state["end_date"] is None
    assert state["cwd_substr"] == ""
    assert state["models"] == []


def test_default_message_mentions_ingest() -> None:
    assert "claudegnostic ingest" in DEFAULT_MESSAGE


def test_is_db_empty_on_empty_db(empty_db_path: Path) -> None:
    with open_conn(str(empty_db_path)) as conn:
        assert is_db_empty(conn) is True


def test_is_db_empty_on_seeded_db(seeded_db_path: Path) -> None:
    with open_conn(str(seeded_db_path)) as conn:
        assert is_db_empty(conn) is False


def test_open_conn_is_read_only(seeded_db_path: Path) -> None:
    with open_conn(str(seeded_db_path)) as conn, pytest.raises(duckdb.Error):
        conn.execute("INSERT INTO sessions (session_id) VALUES ('X')")


def test_compute_filter_options_returns_distinct(seeded_db_path: Path) -> None:
    with open_conn(str(seeded_db_path)) as conn:
        opts = compute_filter_options(conn)
    assert opts["cwds"] == ["/proj/a", "/proj/b"]
    assert opts["models"] == ["claude-opus-4-7", "claude-sonnet-4-6"]
    assert opts["date_min"] == date(2026, 6, 10)
    assert opts["date_max"] == date(2026, 6, 11)


def test_compute_filter_options_on_empty_db(empty_db_path: Path) -> None:
    with open_conn(str(empty_db_path)) as conn:
        opts = compute_filter_options(conn)
    assert opts["cwds"] == []
    assert opts["models"] == []
    assert opts["date_min"] is None
    assert opts["date_max"] is None


def test_analysis_returns_polars_dataframe_via_dashboard_conn(
    seeded_db_path: Path,
) -> None:
    """The bead's 'DataFrame not Relation' acceptance check.

    Calling an analysis function through the dashboard's connection path
    must return a fully materialized polars DataFrame, not a
    ``duckdb.DuckDBPyRelation``.
    """
    with open_conn(str(seeded_db_path)) as conn:
        df = archaeology.session_length_distribution(conn)
    assert isinstance(df, pl.DataFrame)
    assert not isinstance(df, duckdb.DuckDBPyRelation)
