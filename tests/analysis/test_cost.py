"""Cost lens tests."""

from __future__ import annotations

import math
from datetime import datetime

import duckdb

from claudegnostic.analysis import cost


def test_tokens_by_model_happy_path(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.tokens_by_model(seeded_db)
    assert set(df.columns) == {"model", "input", "output", "cache_creation", "cache_read", "total"}
    rows = {r["model"]: r for r in df.to_dicts()}
    assert set(rows) == {"claude-opus-4-7", "claude-sonnet-4-6"}
    opus = rows["claude-opus-4-7"]
    assert opus["input"] == 21_500
    assert opus["output"] == 310
    assert opus["cache_creation"] == 500
    assert opus["cache_read"] == 6_000
    assert opus["total"] == 21_500 + 310 + 500 + 6_000


def test_tokens_by_model_since_filter(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.tokens_by_model(seeded_db, since=datetime(2026, 6, 11, 0, 0, 0))
    rows = {r["model"]: r["input"] for r in df.to_dicts()}
    # Only the 6/11 opus turn (500) survives, and both sonnet turns (3500).
    assert rows["claude-opus-4-7"] == 500
    assert rows["claude-sonnet-4-6"] == 3_500


def test_tokens_by_model_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.tokens_by_model(empty_db)
    assert df.is_empty()
    assert df.columns == ["model", "input", "output", "cache_creation", "cache_read", "total"]


def test_estimated_cost_by_session_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.estimated_cost_by_session(seeded_db)
    assert set(df.columns) == {
        "session_id",
        "cwd",
        "model",
        "input_usd",
        "output_usd",
        "cache_creation_usd",
        "cache_read_usd",
        "est_usd",
    }
    # Session A on opus: 21500 input @ $15/M, 310 output @ $75/M,
    # 500 cache_creation @ $18.75/M, 6000 cache_read @ $1.50/M.
    row_a = next(r for r in df.to_dicts() if r["session_id"] == "A")
    assert math.isclose(row_a["input_usd"], 21_500 * 15 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(row_a["output_usd"], 310 * 75 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(row_a["cache_creation_usd"], 500 * 18.75 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(row_a["cache_read_usd"], 6_000 * 1.50 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(
        row_a["est_usd"],
        row_a["input_usd"]
        + row_a["output_usd"]
        + row_a["cache_creation_usd"]
        + row_a["cache_read_usd"],
        rel_tol=1e-9,
    )


def test_estimated_cost_unknown_model_is_zero(seeded_db: duckdb.DuckDBPyConnection) -> None:
    # Empty override price table -> every model is unknown -> zero cost.
    df = cost.estimated_cost_by_session(seeded_db, prices={})
    assert df["est_usd"].sum() == 0.0


def test_estimated_cost_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.estimated_cost_by_session(empty_db)
    assert df.is_empty()
    assert "est_usd" in df.columns


def test_cost_vs_context_by_turn_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.cost_vs_context_by_turn(seeded_db)
    assert set(df.columns) == {
        "session_id",
        "turn_index",
        "cwd",
        "model",
        "context_tokens",
        "est_usd",
    }
    # One row per turn. Session A has three turns (indices 0,1,2).
    a0 = next(
        r for r in df.to_dicts() if r["session_id"] == "A" and r["turn_index"] == 0
    )
    # Session A turn 0: input 1000, cache_read 4000, cache_creation 500
    # -> context_tokens = 5500.
    assert a0["context_tokens"] == 1_000 + 4_000 + 500
    # Opus pricing: 1000*15 + 200*75 + 500*18.75 + 4000*1.50 (per million).
    expected = (
        1_000 * 15 + 200 * 75 + 500 * 18.75 + 4_000 * 1.50
    ) / 1_000_000
    assert math.isclose(a0["est_usd"], expected, rel_tol=1e-9)


def test_cost_vs_context_by_turn_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.cost_vs_context_by_turn(empty_db)
    assert df.is_empty()
    assert df.columns == [
        "session_id",
        "turn_index",
        "cwd",
        "model",
        "context_tokens",
        "est_usd",
    ]


def test_cache_savings_happy(seeded_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.cache_savings(seeded_db)
    assert set(df.columns) == {
        "session_id",
        "cache_read_tokens",
        "would_have_paid_usd",
        "actually_paid_usd",
        "savings_usd",
    }
    rows = {r["session_id"]: r for r in df.to_dicts()}
    # Session A opus: cache_read 6000 -> would 6000*15/1M, actually 6000*1.50/1M.
    a = rows["A"]
    assert a["cache_read_tokens"] == 6_000
    assert math.isclose(a["would_have_paid_usd"], 6_000 * 15 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(a["actually_paid_usd"], 6_000 * 1.50 / 1_000_000, rel_tol=1e-9)
    assert math.isclose(a["savings_usd"], a["would_have_paid_usd"] - a["actually_paid_usd"])


def test_cache_savings_empty(empty_db: duckdb.DuckDBPyConnection) -> None:
    df = cost.cache_savings(empty_db)
    assert df.is_empty()
    assert "savings_usd" in df.columns
