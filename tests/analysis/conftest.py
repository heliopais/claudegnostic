"""Shared fixtures: in-memory DuckDB seeded with a tiny, hand-built corpus."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import duckdb
import pytest

from claudegnostic.schema import apply_schema


def _insert_turn(
    conn: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    turn_index: int,
    timestamp: datetime,
    cwd: str,
    model: str,
    is_sidechain: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    text_output_chars: int = 0,
    tool_call_count: int = 0,
    tool_names: list[str] | None = None,
    wall_duration_ms: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO turns (
            session_id, turn_index, uuid, parent_uuid, timestamp, cwd, git_branch,
            model, is_sidechain, is_compact_summary,
            input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
            service_tier, stop_reason, text_output_chars, thinking_chars,
            tool_call_count, tool_names, tool_input_bytes, tool_result_bytes,
            wall_duration_ms
        ) VALUES (?, ?, NULL, NULL, ?, ?, NULL, ?, ?, FALSE, ?, ?, ?, ?, NULL, NULL,
                  ?, 0, ?, ?, 0, 0, ?)
        """,
        [
            session_id,
            turn_index,
            timestamp,
            cwd,
            model,
            is_sidechain,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            text_output_chars,
            tool_call_count,
            tool_names or [],
            wall_duration_ms,
        ],
    )


def _insert_session(
    conn: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    cwd: str,
    turn_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, cwd, git_branch, started_at, ended_at,
            turn_count, sidechain_turn_count, compaction_count,
            total_input_tokens, total_output_tokens,
            total_cache_creation_tokens, total_cache_read_tokens,
            cache_hit_ratio, models_used, top_tools, total_wall_duration_ms
        ) VALUES (?, ?, NULL, NULL, NULL, ?, 0, 0, 0, 0, 0, 0, NULL, [], [], 0)
        """,
        [session_id, cwd, turn_count],
    )


@pytest.fixture
def empty_db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def seeded_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """Two sessions across two cwds, two models, mixed tool use and a sidechain."""
    conn = duckdb.connect(":memory:")
    apply_schema(conn)

    # Session A: opus, in /proj/a, 3 turns, one wasted turn, two tool turns.
    _insert_turn(
        conn,
        session_id="A",
        turn_index=0,
        timestamp=datetime(2026, 6, 10, 9, 0, 0),
        cwd="/proj/a",
        model="claude-opus-4-7",
        input_tokens=1_000,
        output_tokens=200,
        cache_creation_tokens=500,
        cache_read_tokens=4_000,
        text_output_chars=400,
        tool_call_count=2,
        tool_names=["Read", "Edit"],
        wall_duration_ms=2_000,
    )
    _insert_turn(
        conn,
        session_id="A",
        turn_index=1,
        timestamp=datetime(2026, 6, 10, 9, 5, 0),
        cwd="/proj/a",
        model="claude-opus-4-7",
        input_tokens=20_000,
        output_tokens=10,
        text_output_chars=50,  # wasted turn: big input, no tools, tiny output
        tool_call_count=0,
        tool_names=[],
        wall_duration_ms=8_000,
    )
    _insert_turn(
        conn,
        session_id="A",
        turn_index=2,
        timestamp=datetime(2026, 6, 11, 9, 0, 0),
        cwd="/proj/a",
        model="claude-opus-4-7",
        is_sidechain=True,
        input_tokens=500,
        output_tokens=100,
        cache_read_tokens=2_000,
        text_output_chars=300,
        tool_call_count=1,
        tool_names=["Read"],
        wall_duration_ms=1_000,
    )

    # Session B: sonnet, in /proj/b, 2 turns, both with tools.
    _insert_turn(
        conn,
        session_id="B",
        turn_index=0,
        timestamp=datetime(2026, 6, 11, 12, 0, 0),
        cwd="/proj/b",
        model="claude-sonnet-4-6",
        input_tokens=2_000,
        output_tokens=400,
        cache_read_tokens=6_000,
        text_output_chars=500,
        tool_call_count=2,
        tool_names=["Read", "Bash"],
        wall_duration_ms=3_000,
    )
    _insert_turn(
        conn,
        session_id="B",
        turn_index=1,
        timestamp=datetime(2026, 6, 11, 12, 10, 0),
        cwd="/proj/b",
        model="claude-sonnet-4-6",
        input_tokens=1_500,
        output_tokens=600,
        cache_read_tokens=3_000,
        text_output_chars=700,
        tool_call_count=1,
        tool_names=["Bash"],
        wall_duration_ms=5_000,
    )

    _insert_session(conn, session_id="A", cwd="/proj/a", turn_count=3)
    _insert_session(conn, session_id="B", cwd="/proj/b", turn_count=2)

    try:
        yield conn
    finally:
        conn.close()
