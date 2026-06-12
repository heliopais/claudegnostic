"""DuckDB schema definitions for turns and sessions tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

TURNS_DDL = """
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    uuid TEXT,
    parent_uuid TEXT,
    timestamp TIMESTAMP,
    cwd TEXT,
    git_branch TEXT,
    model TEXT,
    is_sidechain BOOLEAN,
    is_compact_summary BOOLEAN,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens INTEGER,
    service_tier TEXT,
    stop_reason TEXT,
    text_output_chars INTEGER,
    thinking_chars INTEGER,
    tool_call_count INTEGER,
    tool_names TEXT[],
    tool_input_bytes INTEGER,
    tool_result_bytes INTEGER,
    wall_duration_ms INTEGER,
    PRIMARY KEY (session_id, turn_index)
)
"""

SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    cwd TEXT,
    git_branch TEXT,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    turn_count INTEGER,
    sidechain_turn_count INTEGER,
    compaction_count INTEGER,
    total_input_tokens BIGINT,
    total_output_tokens BIGINT,
    total_cache_creation_tokens BIGINT,
    total_cache_read_tokens BIGINT,
    cache_hit_ratio DOUBLE,
    models_used TEXT[],
    top_tools TEXT[],
    total_wall_duration_ms BIGINT
)
"""


def apply_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create turns and sessions tables if they do not already exist."""
    conn.execute(TURNS_DDL)
    conn.execute(SESSIONS_DDL)
