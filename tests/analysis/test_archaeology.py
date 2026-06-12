"""Archaeology lens tests."""

from __future__ import annotations

from datetime import date

import duckdb

from claudegnostic.analysis import archaeology


def test_session_length_distribution_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.session_length_distribution(seeded_db)
    assert df.columns == [
        "bucket",
        "count",
        "total_cost_usd",
        "cost_per_session_usd",
        "pct_total_cost",
    ]
    rows = {r["bucket"]: r for r in df.to_dicts()}
    assert rows["1-5"]["count"] == 2  # both sessions have <=5 turns
    assert rows["6-20"]["count"] == 0
    assert rows["100+"]["count"] == 0
    # Empty buckets must report zeros for all cost metrics.
    assert rows["6-20"]["total_cost_usd"] == 0.0
    assert rows["6-20"]["cost_per_session_usd"] == 0.0
    assert rows["6-20"]["pct_total_cost"] == 0.0
    # Bucket order is preserved (numeric, not alphabetic).
    assert list(rows.keys()) == ["1-5", "6-20", "21-50", "51-100", "100+"]


def test_session_length_distribution_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.session_length_distribution(empty_db)
    assert df.is_empty()
    assert df.columns == [
        "bucket",
        "count",
        "total_cost_usd",
        "cost_per_session_usd",
        "pct_total_cost",
    ]


def test_tool_co_occurrence_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.tool_co_occurrence(seeded_db)
    assert set(df.columns) == {"tool_a", "tool_b", "pair_count"}
    pairs = {(r["tool_a"], r["tool_b"]): r["pair_count"] for r in df.to_dicts()}
    # Turn A0 has [Read, Edit] -> (Edit, Read). Turn B0 has [Read, Bash] -> (Bash, Read).
    assert pairs == {("Edit", "Read"): 1, ("Bash", "Read"): 1}


def test_tool_co_occurrence_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.tool_co_occurrence(empty_db)
    assert df.is_empty()
    assert df.columns == ["tool_a", "tool_b", "pair_count"]


def test_sidechain_ratio_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.sidechain_ratio_by_session(seeded_db)
    rows = {r["session_id"]: r for r in df.to_dicts()}
    assert rows["A"]["total_turns"] == 3
    assert rows["A"]["sidechain_turns"] == 1
    assert abs(rows["A"]["ratio"] - 1 / 3) < 1e-9
    assert rows["B"]["sidechain_turns"] == 0
    assert rows["B"]["ratio"] == 0.0


def test_sidechain_ratio_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.sidechain_ratio_by_session(empty_db)
    assert df.is_empty()
    assert df.columns == ["session_id", "total_turns", "sidechain_turns", "ratio"]


def test_project_activity_by_day_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.project_activity_by_day(seeded_db)
    assert set(df.columns) == {"date", "cwd", "turns", "tokens"}
    rows = {(r["date"], r["cwd"]): r for r in df.to_dicts()}
    # /proj/a on 2026-06-10: 2 turns. /proj/a on 2026-06-11: 1 sidechain turn.
    assert rows[(date(2026, 6, 10), "/proj/a")]["turns"] == 2
    assert rows[(date(2026, 6, 11), "/proj/a")]["turns"] == 1
    assert rows[(date(2026, 6, 11), "/proj/b")]["turns"] == 2


def test_project_activity_by_day_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = archaeology.project_activity_by_day(empty_db)
    assert df.is_empty()
    assert df.columns == ["date", "cwd", "turns", "tokens"]
