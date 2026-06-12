"""Connection adapter contract: accept a path, close what we open."""

from __future__ import annotations

from pathlib import Path

import duckdb

from claudegnostic.analysis import archaeology
from claudegnostic.analysis._connect import as_connection
from claudegnostic.schema import apply_schema


def test_as_connection_with_path_opens_and_closes(tmp_path: Path) -> None:
    db = tmp_path / "stats.duckdb"
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    conn.close()

    with as_connection(db) as opened:
        opened.execute("SELECT 1").fetchall()
    # Should be closed now: a second writer can open the file without contention.
    again = duckdb.connect(str(db))
    again.close()


def test_analysis_function_accepts_path(tmp_path: Path) -> None:
    db = tmp_path / "stats.duckdb"
    conn = duckdb.connect(str(db))
    apply_schema(conn)
    conn.close()

    df = archaeology.session_length_distribution(db)
    assert df.is_empty()
    assert df.columns == [
        "bucket",
        "count",
        "total_cost_usd",
        "cost_per_session_usd",
        "pct_total_cost",
    ]


def test_as_connection_with_connection_does_not_close(
    empty_db: duckdb.DuckDBPyConnection,
) -> None:
    with as_connection(empty_db) as same:
        assert same is empty_db
    # The fixture connection must still be usable after the context exits.
    empty_db.execute("SELECT 1").fetchall()
