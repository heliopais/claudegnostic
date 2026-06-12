"""Productivity lens tests."""

from __future__ import annotations

import duckdb

from claudegnostic.analysis import productivity


def test_cache_hit_ratio_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.cache_hit_ratio_by_session(seeded_db)
    assert set(df.columns) == {"session_id", "ratio", "ratio_bucket"}
    rows = {r["session_id"]: r for r in df.to_dicts()}
    # Session A: cache_read 6000, input 21500 -> 6000 / 27500 ~ 0.218 (<0.25).
    assert abs(rows["A"]["ratio"] - 6_000 / 27_500) < 1e-9
    assert rows["A"]["ratio_bucket"] == "<0.25"
    # Session B: cache_read 9000, input 3500 -> 9000/12500 = 0.72 -> 0.5-0.75.
    assert abs(rows["B"]["ratio"] - 9_000 / 12_500) < 1e-9
    assert rows["B"]["ratio_bucket"] == "0.5-0.75"


def test_cache_hit_ratio_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.cache_hit_ratio_by_session(empty_db)
    assert df.is_empty()
    assert df.columns == ["session_id", "ratio", "ratio_bucket"]


def test_wall_time_per_tool_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.wall_time_per_tool(seeded_db)
    assert set(df.columns) == {"tool", "p50_ms", "p90_ms", "count"}
    counts = {r["tool"]: r["count"] for r in df.to_dicts()}
    # Read: turns A0, A2, B0 -> 3. Edit: A0 -> 1. Bash: B0, B1 -> 2.
    assert counts == {"Read": 3, "Edit": 1, "Bash": 2}


def test_wall_time_per_tool_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.wall_time_per_tool(empty_db)
    assert df.is_empty()
    assert df.columns == ["tool", "p50_ms", "p90_ms", "count"]


def test_wasted_turns_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.wasted_turns(seeded_db)
    rows = df.to_dicts()
    assert len(rows) == 1
    only = rows[0]
    assert only["session_id"] == "A"
    assert only["turn_index"] == 1
    assert only["input_tokens"] == 20_000
    assert only["tool_call_count"] == 0


def test_wasted_turns_thresholds(seeded_db: duckdb.DuckDBPyConnection) -> None:
    # Raise the input threshold above the only wasted-turn's input -> empty.
    df = productivity.wasted_turns(seeded_db, input_token_threshold=100_000)
    assert df.is_empty()
    assert df.columns == [
        "session_id",
        "turn_index",
        "input_tokens",
        "tool_call_count",
        "text_output_chars",
    ]


def test_wasted_turns_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = productivity.wasted_turns(empty_db)
    assert df.is_empty()
