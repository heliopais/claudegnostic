"""Productivity lens: cache hit ratio, per-tool latency, wasted turns."""

from __future__ import annotations

import polars as pl

from claudegnostic.analysis._connect import ConnLike, as_connection

_CACHE_HIT_RATIO_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "ratio": pl.Float64,
    "ratio_bucket": pl.String,
}


def cache_hit_ratio_by_session(con_or_path: ConnLike) -> pl.DataFrame:
    """Compute the cache hit ratio for each session.

    Ratio = ``cache_read / (cache_read + input)``. Sessions where the
    denominator is zero are omitted (no prompt traffic to attribute).

    Buckets: ``<0.25``, ``0.25-0.5``, ``0.5-0.75``, ``>=0.75``.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``session_id, ratio, ratio_bucket``. Returns
        an empty DataFrame with this schema when no sessions match.
    """
    sql = """
        SELECT
            session_id,
            CAST(SUM(COALESCE(cache_read_tokens, 0)) AS DOUBLE)
                / NULLIF(
                    SUM(COALESCE(cache_read_tokens, 0)) + SUM(COALESCE(input_tokens, 0)),
                    0
                ) AS ratio
        FROM turns
        GROUP BY session_id
        HAVING ratio IS NOT NULL
        ORDER BY ratio DESC
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_CACHE_HIT_RATIO_SCHEMA)
    return df.with_columns(
        pl.when(pl.col("ratio") < 0.25)
        .then(pl.lit("<0.25"))
        .when(pl.col("ratio") < 0.5)
        .then(pl.lit("0.25-0.5"))
        .when(pl.col("ratio") < 0.75)
        .then(pl.lit("0.5-0.75"))
        .otherwise(pl.lit(">=0.75"))
        .alias("ratio_bucket")
    )


_WALL_TIME_PER_TOOL_SCHEMA: dict[str, type[pl.DataType]] = {
    "tool": pl.String,
    "p50_ms": pl.Float64,
    "p90_ms": pl.Float64,
    "count": pl.Int64,
}


def wall_time_per_tool(con_or_path: ConnLike) -> pl.DataFrame:
    """Approximate per-tool wall time using the turn's total duration.

    Each turn's ``wall_duration_ms`` is attributed in full to every tool
    named in that turn's ``tool_names`` array. This is a coarse heuristic
    (parallel tool calls share the turn duration), and is documented as
    such for downstream surfaces.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.

    Returns:
        DataFrame with columns ``tool, p50_ms, p90_ms, count``, sorted by
        ``count`` descending. Returns an empty DataFrame with this schema
        when no tool-bearing turns exist.
    """
    sql = """
        WITH expanded AS (
            SELECT UNNEST(tool_names) AS tool, wall_duration_ms
            FROM turns
            WHERE tool_names IS NOT NULL
              AND len(tool_names) > 0
              AND wall_duration_ms IS NOT NULL
        )
        SELECT
            tool,
            QUANTILE_CONT(wall_duration_ms, 0.5)::DOUBLE AS p50_ms,
            QUANTILE_CONT(wall_duration_ms, 0.9)::DOUBLE AS p90_ms,
            COUNT(*)::BIGINT AS count
        FROM expanded
        GROUP BY tool
        ORDER BY count DESC
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_WALL_TIME_PER_TOOL_SCHEMA)
    return df


_WASTED_TURNS_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "turn_index": pl.Int32,
    "input_tokens": pl.Int32,
    "tool_call_count": pl.Int32,
    "text_output_chars": pl.Int32,
}


def wasted_turns(
    con_or_path: ConnLike,
    input_token_threshold: int = 10_000,
    max_text_chars: int = 200,
) -> pl.DataFrame:
    """Flag turns that consumed a large prompt but produced nothing actionable.

    Heuristic: ``input_tokens >= input_token_threshold`` AND
    ``tool_call_count = 0`` AND ``text_output_chars <= max_text_chars``.
    The thresholds are tunable; defaults assume a turn that spent >=10k
    input tokens and emitted no tools plus <=200 chars of text is suspicious.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        input_token_threshold: Lower bound on ``input_tokens`` (inclusive).
        max_text_chars: Upper bound on ``text_output_chars`` (inclusive).

    Returns:
        DataFrame with columns ``session_id, turn_index, input_tokens,
        tool_call_count, text_output_chars``, sorted by ``input_tokens``
        descending. Returns an empty DataFrame with this schema when no
        turn meets the heuristic.
    """
    sql = """
        SELECT
            session_id,
            turn_index,
            COALESCE(input_tokens, 0)      AS input_tokens,
            COALESCE(tool_call_count, 0)   AS tool_call_count,
            COALESCE(text_output_chars, 0) AS text_output_chars
        FROM turns
        WHERE COALESCE(input_tokens, 0) >= ?
          AND COALESCE(tool_call_count, 0) = 0
          AND COALESCE(text_output_chars, 0) <= ?
        ORDER BY input_tokens DESC
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql, [input_token_threshold, max_text_chars]).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_WASTED_TURNS_SCHEMA)
    return df
