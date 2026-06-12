"""Session discovery and idempotent ingest into DuckDB.

Walk ``~/.claude/projects/**/*.jsonl``, parse each file, and upsert into
``turns``. After turns are written, the ``sessions`` rows for affected
session ids are recomputed from a SQL aggregation over ``turns``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from claudegnostic.parser import TURNS_SCHEMA, parse_session

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)


_TURNS_COLUMNS: tuple[str, ...] = tuple(TURNS_SCHEMA.keys())
_TURNS_NON_KEY_COLUMNS: tuple[str, ...] = tuple(
    c for c in _TURNS_COLUMNS if c not in ("session_id", "turn_index")
)


@dataclass
class IngestReport:
    """Summary of an ingest run."""

    files_scanned: int = 0
    turns_added: int = 0
    turns_updated: int = 0
    sessions_touched: int = 0
    files_failed: list[Path] = field(default_factory=list)


def default_sessions_root() -> Path:
    """Default root for Claude Code session JSONL files."""
    return Path.home() / ".claude" / "projects"


def discover_sessions(root: Path) -> Iterator[Path]:
    """Yield every ``*.jsonl`` file under ``root`` (recursive)."""
    if not root.exists():
        return
    yield from root.rglob("*.jsonl")


def _upsert_turns(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> tuple[int, int, set[str]]:
    """Upsert a turns DataFrame; return (added, updated, session_ids_touched)."""
    if df.is_empty():
        return 0, 0, set()

    session_ids = {sid for sid in df["session_id"].to_list() if sid is not None}

    keys = df.select(["session_id", "turn_index"])
    conn.register("incoming_keys", keys)
    existing = conn.execute(
        "SELECT COUNT(*) FROM turns t "
        "JOIN incoming_keys k "
        "  ON t.session_id = k.session_id AND t.turn_index = k.turn_index"
    ).fetchone()
    conn.unregister("incoming_keys")
    existing_count = int(existing[0]) if existing else 0

    incoming_count = df.height
    turns_added = incoming_count - existing_count
    turns_updated = existing_count

    columns_csv = ", ".join(_TURNS_COLUMNS)
    update_csv = ", ".join(f"{c} = excluded.{c}" for c in _TURNS_NON_KEY_COLUMNS)
    conn.register("incoming_turns", df)
    conn.execute(
        f"INSERT INTO turns ({columns_csv}) "
        f"SELECT {columns_csv} FROM incoming_turns "
        f"ON CONFLICT (session_id, turn_index) DO UPDATE SET {update_csv}"
    )
    conn.unregister("incoming_turns")

    return turns_added, turns_updated, session_ids


_SESSIONS_AGG_SQL = """
WITH agg AS (
    SELECT
        session_id,
        any_value(cwd) AS cwd,
        any_value(git_branch) AS git_branch,
        min(timestamp) AS started_at,
        max(timestamp) AS ended_at,
        count(*) AS turn_count,
        sum(CASE WHEN is_sidechain THEN 1 ELSE 0 END) AS sidechain_turn_count,
        sum(CASE WHEN is_compact_summary THEN 1 ELSE 0 END) AS compaction_count,
        sum(coalesce(input_tokens, 0)) AS total_input_tokens,
        sum(coalesce(output_tokens, 0)) AS total_output_tokens,
        sum(coalesce(cache_creation_tokens, 0)) AS total_cache_creation_tokens,
        sum(coalesce(cache_read_tokens, 0)) AS total_cache_read_tokens,
        sum(coalesce(wall_duration_ms, 0)) AS total_wall_duration_ms,
        list_distinct(list(model) FILTER (WHERE model IS NOT NULL)) AS models_used,
        flatten(list(tool_names) FILTER (WHERE tool_names IS NOT NULL)) AS all_tools
    FROM turns
    WHERE session_id IN (SELECT session_id FROM affected_sessions)
    GROUP BY session_id
),
tool_counts AS (
    SELECT
        session_id,
        list(name ORDER BY n DESC)[1:5] AS top_tools
    FROM (
        SELECT session_id, name, count(*) AS n
        FROM (
            SELECT session_id, unnest(all_tools) AS name FROM agg
        )
        GROUP BY session_id, name
    )
    GROUP BY session_id
)
SELECT
    a.session_id,
    a.cwd,
    a.git_branch,
    a.started_at,
    a.ended_at,
    a.turn_count,
    a.sidechain_turn_count,
    a.compaction_count,
    a.total_input_tokens,
    a.total_output_tokens,
    a.total_cache_creation_tokens,
    a.total_cache_read_tokens,
    CASE
        WHEN (a.total_cache_read_tokens + a.total_cache_creation_tokens + a.total_input_tokens) > 0
        THEN a.total_cache_read_tokens::DOUBLE
             / (a.total_cache_read_tokens + a.total_cache_creation_tokens + a.total_input_tokens)
        ELSE NULL
    END AS cache_hit_ratio,
    a.models_used,
    coalesce(t.top_tools, []) AS top_tools,
    a.total_wall_duration_ms
FROM agg a
LEFT JOIN tool_counts t USING (session_id)
"""


def _recompute_sessions(conn: duckdb.DuckDBPyConnection, session_ids: Iterable[str]) -> int:
    """Recompute the ``sessions`` rows for the given ids from ``turns``."""
    ids = [sid for sid in session_ids if sid]
    if not ids:
        return 0

    affected = pl.DataFrame({"session_id": ids}, schema={"session_id": pl.Utf8})
    conn.register("affected_sessions", affected)
    try:
        conn.execute(
            "DELETE FROM sessions WHERE session_id IN (SELECT session_id FROM affected_sessions)"
        )
        conn.execute(f"INSERT INTO sessions {_SESSIONS_AGG_SQL}")
    finally:
        conn.unregister("affected_sessions")
    return len(ids)


def ingest_paths(conn: duckdb.DuckDBPyConnection, paths: Iterable[Path]) -> IngestReport:
    """Parse and upsert each path; recompute affected sessions; return a report."""
    report = IngestReport()
    touched: set[str] = set()

    for path in paths:
        report.files_scanned += 1
        try:
            df = parse_session(path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            report.files_failed.append(path)
            continue

        added, updated, session_ids = _upsert_turns(conn, df)
        report.turns_added += added
        report.turns_updated += updated
        touched.update(session_ids)

    report.sessions_touched = _recompute_sessions(conn, touched)
    return report


def ingest_root(conn: duckdb.DuckDBPyConnection, root: Path | None = None) -> IngestReport:
    """Discover and ingest every JSONL file under ``root``."""
    actual_root = root if root is not None else default_sessions_root()
    return ingest_paths(conn, discover_sessions(actual_root))
