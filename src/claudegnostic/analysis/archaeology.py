"""Workflow archaeology: shape and rhythm of sessions over time."""

from __future__ import annotations

import polars as pl

from claudegnostic.analysis._connect import ConnLike, as_connection
from claudegnostic.analysis.cost import cost_vs_turns_by_session

_SESSION_LENGTH_SCHEMA: dict[str, type[pl.DataType]] = {
    "bucket": pl.String,
    "count": pl.Int64,
    "total_cost_usd": pl.Float64,
    "cost_per_session_usd": pl.Float64,
    "pct_total_cost": pl.Float64,
}

_SESSION_LENGTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, None),
)

SESSION_LENGTH_BUCKET_ORDER: tuple[str, ...] = tuple(
    name for name, _lo, _hi in _SESSION_LENGTH_BUCKETS
)


def session_length_distribution(con_or_path: ConnLike) -> pl.DataFrame:
    """Histogram sessions by total turn count, with cost per bucket.

    Buckets: ``1-5``, ``6-20``, ``21-50``, ``51-100``, ``100+``.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``bucket, count, total_cost_usd,
        cost_per_session_usd, pct_total_cost`` in bucket order. Buckets
        with zero sessions are still present (count = 0, costs = 0.0).
        ``pct_total_cost`` is a percentage in [0, 100]. Returns an empty
        DataFrame with this schema only when the ``sessions`` table itself
        is empty.
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(
            "SELECT COALESCE(turn_count, 0) AS turn_count FROM sessions"
        ).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_SESSION_LENGTH_SCHEMA)

    cost_df = cost_vs_turns_by_session(con_or_path)
    grand_total_cost = (
        float(cost_df["est_usd"].sum()) if not cost_df.is_empty() else 0.0
    )

    rows = []
    for name, lo, hi in _SESSION_LENGTH_BUCKETS:
        mask = pl.col("turn_count") >= lo
        if hi is not None:
            mask = mask & (pl.col("turn_count") <= hi)
        bucket_count = int(df.filter(mask).height)

        if cost_df.is_empty():
            bucket_cost = 0.0
        else:
            cost_mask = pl.col("turn_count") >= lo
            if hi is not None:
                cost_mask = cost_mask & (pl.col("turn_count") <= hi)
            bucket_cost = float(cost_df.filter(cost_mask)["est_usd"].sum())

        cost_per_session = bucket_cost / bucket_count if bucket_count > 0 else 0.0
        pct = (bucket_cost / grand_total_cost * 100.0) if grand_total_cost > 0 else 0.0

        rows.append(
            {
                "bucket": name,
                "count": bucket_count,
                "total_cost_usd": bucket_cost,
                "cost_per_session_usd": cost_per_session,
                "pct_total_cost": pct,
            }
        )
    return pl.DataFrame(rows, schema=_SESSION_LENGTH_SCHEMA)


_TOOL_COOCCURRENCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "tool_a": pl.String,
    "tool_b": pl.String,
    "pair_count": pl.Int64,
}


def tool_co_occurrence(con_or_path: ConnLike) -> pl.DataFrame:
    """Count unordered pairs of tools that co-occur in the same turn.

    Each turn contributes one count per distinct ``(a, b)`` pair with
    ``a < b`` (alphabetical) drawn from ``tool_names``. Turns with fewer
    than two distinct tools contribute nothing.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``tool_a, tool_b, pair_count`` sorted by
        ``pair_count`` descending. Returns an empty DataFrame with this
        schema when no turn has two or more distinct tools.
    """
    sql = """
        WITH per_turn AS (
            SELECT session_id, turn_index, list_distinct(tool_names) AS tools
            FROM turns
            WHERE tool_names IS NOT NULL AND len(tool_names) >= 2
        ),
        pairs AS (
            SELECT
                LEAST(a, b)    AS tool_a,
                GREATEST(a, b) AS tool_b
            FROM per_turn,
                 UNNEST(tools) AS t1(a),
                 UNNEST(tools) AS t2(b)
            WHERE a < b
        )
        SELECT tool_a, tool_b, COUNT(*)::BIGINT AS pair_count
        FROM pairs
        GROUP BY tool_a, tool_b
        ORDER BY pair_count DESC, tool_a, tool_b
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_TOOL_COOCCURRENCE_SCHEMA)
    return df


_SIDECHAIN_RATIO_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "total_turns": pl.Int64,
    "sidechain_turns": pl.Int64,
    "ratio": pl.Float64,
}


def sidechain_ratio_by_session(con_or_path: ConnLike) -> pl.DataFrame:
    """Fraction of turns per session that ran on a sub-agent sidechain.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``session_id, total_turns, sidechain_turns,
        ratio``, sorted by ``ratio`` descending. Returns an empty DataFrame
        with this schema when no turns exist.
    """
    sql = """
        SELECT
            session_id,
            COUNT(*)::BIGINT                                  AS total_turns,
            SUM(CASE WHEN is_sidechain THEN 1 ELSE 0 END)::BIGINT AS sidechain_turns,
            (SUM(CASE WHEN is_sidechain THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) AS ratio
        FROM turns
        GROUP BY session_id
        ORDER BY ratio DESC, total_turns DESC
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_SIDECHAIN_RATIO_SCHEMA)
    return df


_PROJECT_ACTIVITY_SCHEMA: dict[str, type[pl.DataType]] = {
    "date": pl.Date,
    "cwd": pl.String,
    "turns": pl.Int64,
    "tokens": pl.Int64,
}


def project_activity_by_day(con_or_path: ConnLike) -> pl.DataFrame:
    """Turns and total tokens per (day, project) pair.

    Tokens = ``input + output + cache_creation + cache_read``. Turns with
    a null timestamp or null ``cwd`` are skipped.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``date, cwd, turns, tokens``, sorted by
        ``date`` ascending then ``cwd``. Returns an empty DataFrame with
        this schema when no datable turns exist.
    """
    sql = """
        SELECT
            CAST(timestamp AS DATE) AS date,
            cwd,
            COUNT(*)::BIGINT AS turns,
            (
                SUM(COALESCE(input_tokens, 0))
                + SUM(COALESCE(output_tokens, 0))
                + SUM(COALESCE(cache_creation_tokens, 0))
                + SUM(COALESCE(cache_read_tokens, 0))
            )::BIGINT AS tokens
        FROM turns
        WHERE timestamp IS NOT NULL AND cwd IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_PROJECT_ACTIVITY_SCHEMA)
    return df
