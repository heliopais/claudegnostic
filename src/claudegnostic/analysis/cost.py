"""Cost lens: token totals and USD estimates by model and session."""

from __future__ import annotations

import tomllib
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from typing import TypedDict

import polars as pl

from claudegnostic.analysis._connect import ConnLike, as_connection


class ModelPrice(TypedDict):
    input: float
    output: float
    cache_creation: float
    cache_read: float


PriceTable = dict[str, ModelPrice]


@lru_cache(maxsize=1)
def default_prices() -> PriceTable:
    """Load the packaged default price table (USD per million tokens)."""
    raw = files("claudegnostic.analysis").joinpath("prices.toml").read_text()
    data = tomllib.loads(raw)
    return dict(data.get("models", {}))


def _price_lookup_df(prices: PriceTable) -> pl.DataFrame:
    """Materialize the price table as a polars DataFrame for joins."""
    if not prices:
        return pl.DataFrame(
            schema={
                "model_key": pl.String,
                "input_price": pl.Float64,
                "output_price": pl.Float64,
                "cache_creation_price": pl.Float64,
                "cache_read_price": pl.Float64,
            }
        )
    return pl.DataFrame(
        {
            "model_key": list(prices.keys()),
            "input_price": [p["input"] for p in prices.values()],
            "output_price": [p["output"] for p in prices.values()],
            "cache_creation_price": [p["cache_creation"] for p in prices.values()],
            "cache_read_price": [p["cache_read"] for p in prices.values()],
        }
    )


def _attach_prices(turns: pl.DataFrame, prices: PriceTable) -> pl.DataFrame:
    """Attach the longest-prefix-matching price row to each turn.

    Returns the input DataFrame with five extra columns: `model_key`,
    `input_price`, `output_price`, `cache_creation_price`, `cache_read_price`.
    Turns whose model does not match any price key get nulls (zero cost
    contribution after fill).
    """
    price_df = _price_lookup_df(prices)
    if price_df.is_empty() or turns.is_empty():
        return turns.with_columns(
            pl.lit(None, dtype=pl.String).alias("model_key"),
            pl.lit(0.0).alias("input_price"),
            pl.lit(0.0).alias("output_price"),
            pl.lit(0.0).alias("cache_creation_price"),
            pl.lit(0.0).alias("cache_read_price"),
        )

    # Cross-join, keep rows where model starts with model_key, pick longest key.
    joined = (
        turns.join(price_df, how="cross")
        .filter(
            pl.col("model").is_not_null()
            & pl.col("model").str.starts_with(pl.col("model_key"))
        )
        .with_columns(pl.col("model_key").str.len_chars().alias("_key_len"))
        .sort("_key_len", descending=True)
        .unique(subset=["session_id", "turn_index"], keep="first")
        .drop("_key_len")
    )
    # Rejoin to keep turns with no price match.
    return turns.join(
        joined.select(
            "session_id",
            "turn_index",
            "model_key",
            "input_price",
            "output_price",
            "cache_creation_price",
            "cache_read_price",
        ),
        on=["session_id", "turn_index"],
        how="left",
    ).with_columns(
        pl.col("input_price").fill_null(0.0),
        pl.col("output_price").fill_null(0.0),
        pl.col("cache_creation_price").fill_null(0.0),
        pl.col("cache_read_price").fill_null(0.0),
    )


_TOKENS_BY_MODEL_SCHEMA: dict[str, type[pl.DataType]] = {
    "model": pl.String,
    "input": pl.Int64,
    "output": pl.Int64,
    "cache_creation": pl.Int64,
    "cache_read": pl.Int64,
    "total": pl.Int64,
}


def tokens_by_model(
    con_or_path: ConnLike,
    since: datetime | None = None,
) -> pl.DataFrame:
    """Sum token usage grouped by model.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        since: Optional lower bound (inclusive) on ``turns.timestamp``.

    Returns:
        DataFrame with columns ``model, input, output, cache_creation,
        cache_read, total``. Returns an empty DataFrame with this schema when
        no turns match.
    """
    where = "WHERE timestamp >= ?" if since is not None else ""
    params: list[object] = [since] if since is not None else []
    sql = f"""
        SELECT
            COALESCE(model, '<unknown>') AS model,
            SUM(COALESCE(input_tokens, 0))::BIGINT          AS input,
            SUM(COALESCE(output_tokens, 0))::BIGINT         AS output,
            SUM(COALESCE(cache_creation_tokens, 0))::BIGINT AS cache_creation,
            SUM(COALESCE(cache_read_tokens, 0))::BIGINT     AS cache_read,
            (
                SUM(COALESCE(input_tokens, 0))
                + SUM(COALESCE(output_tokens, 0))
                + SUM(COALESCE(cache_creation_tokens, 0))
                + SUM(COALESCE(cache_read_tokens, 0))
            )::BIGINT AS total
        FROM turns
        {where}
        GROUP BY 1
        ORDER BY total DESC
    """
    with as_connection(con_or_path) as conn:
        df = conn.execute(sql, params).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_TOKENS_BY_MODEL_SCHEMA)
    return df


_ESTIMATED_COST_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "cwd": pl.String,
    "model": pl.String,
    "input_usd": pl.Float64,
    "output_usd": pl.Float64,
    "cache_creation_usd": pl.Float64,
    "cache_read_usd": pl.Float64,
    "est_usd": pl.Float64,
}


def estimated_cost_by_session(
    con_or_path: ConnLike,
    prices: PriceTable | None = None,
) -> pl.DataFrame:
    """Estimate USD spent per (session, model) pair.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        prices: Optional override for the default packaged price table.

    Returns:
        DataFrame with columns ``session_id, cwd, model, input_usd,
        output_usd, cache_creation_usd, cache_read_usd, est_usd``. Prices are
        USD per million tokens; unknown models contribute 0 USD. Returns an
        empty DataFrame with this schema when no turns match.
    """
    table = prices if prices is not None else default_prices()
    with as_connection(con_or_path) as conn:
        turns = conn.execute(
            """
            SELECT
                session_id,
                turn_index,
                cwd,
                model,
                COALESCE(input_tokens, 0)          AS input_tokens,
                COALESCE(output_tokens, 0)         AS output_tokens,
                COALESCE(cache_creation_tokens, 0) AS cache_creation_tokens,
                COALESCE(cache_read_tokens, 0)     AS cache_read_tokens
            FROM turns
            """
        ).pl()
    if turns.is_empty():
        return pl.DataFrame(schema=_ESTIMATED_COST_SCHEMA)

    priced = _attach_prices(turns, table)
    out = (
        priced.with_columns(
            (pl.col("input_tokens") * pl.col("input_price") / 1_000_000).alias("input_usd"),
            (pl.col("output_tokens") * pl.col("output_price") / 1_000_000).alias("output_usd"),
            (
                pl.col("cache_creation_tokens") * pl.col("cache_creation_price") / 1_000_000
            ).alias("cache_creation_usd"),
            (pl.col("cache_read_tokens") * pl.col("cache_read_price") / 1_000_000).alias(
                "cache_read_usd"
            ),
        )
        .with_columns(
            (
                pl.col("input_usd")
                + pl.col("output_usd")
                + pl.col("cache_creation_usd")
                + pl.col("cache_read_usd")
            ).alias("est_usd")
        )
        .group_by(["session_id", "cwd", "model"])
        .agg(
            pl.col("input_usd").sum(),
            pl.col("output_usd").sum(),
            pl.col("cache_creation_usd").sum(),
            pl.col("cache_read_usd").sum(),
            pl.col("est_usd").sum(),
        )
        .sort("est_usd", descending=True)
        .select(list(_ESTIMATED_COST_SCHEMA.keys()))
    )
    return out


_COST_VS_TURNS_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "cwd": pl.String,
    "turn_count": pl.Int64,
    "est_usd": pl.Float64,
}


def cost_vs_turns_by_session(
    con_or_path: ConnLike,
    prices: PriceTable | None = None,
) -> pl.DataFrame:
    """Per-session total estimated USD alongside total turn count.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        prices: Optional override for the default packaged price table.

    Returns:
        DataFrame with columns ``session_id, cwd, turn_count, est_usd``.
        One row per session that has at least one turn. Sorted by
        ``est_usd`` descending. Returns an empty DataFrame with this
        schema when no sessions match.
    """
    by_session = estimated_cost_by_session(con_or_path, prices=prices)
    if by_session.is_empty():
        return pl.DataFrame(schema=_COST_VS_TURNS_SCHEMA)

    totals = by_session.group_by("session_id").agg(
        pl.col("cwd").first().alias("cwd"),
        pl.col("est_usd").sum().alias("est_usd"),
    )

    with as_connection(con_or_path) as conn:
        turn_counts = conn.execute(
            "SELECT session_id, COALESCE(turn_count, 0)::BIGINT AS turn_count "
            "FROM sessions"
        ).pl()

    if turn_counts.is_empty():
        return pl.DataFrame(schema=_COST_VS_TURNS_SCHEMA)

    return (
        totals.join(turn_counts, on="session_id", how="inner")
        .filter(pl.col("turn_count") > 0)
        .select(list(_COST_VS_TURNS_SCHEMA.keys()))
        .sort("est_usd", descending=True)
    )


_COST_VS_CONTEXT_PER_TURN_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "turn_index": pl.Int64,
    "cwd": pl.String,
    "model": pl.String,
    "context_tokens": pl.Int64,
    "est_usd": pl.Float64,
}


def cost_vs_context_by_turn(
    con_or_path: ConnLike,
    prices: PriceTable | None = None,
) -> pl.DataFrame:
    """Per-turn estimated USD alongside the context window sent to the model.

    Context window is approximated as the sum of tokens the model actually
    read at request time: ``input_tokens + cache_read_tokens +
    cache_creation_tokens``. Cost is the estimated USD for the same turn
    using the packaged price table.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        prices: Optional override for the default packaged price table.

    Returns:
        DataFrame with columns ``session_id, turn_index, cwd, model,
        context_tokens, est_usd``. One row per turn. Returns an empty
        DataFrame with this schema when no turns match.
    """
    table = prices if prices is not None else default_prices()
    with as_connection(con_or_path) as conn:
        turns = conn.execute(
            """
            SELECT
                session_id,
                turn_index,
                cwd,
                model,
                COALESCE(input_tokens, 0)          AS input_tokens,
                COALESCE(output_tokens, 0)         AS output_tokens,
                COALESCE(cache_creation_tokens, 0) AS cache_creation_tokens,
                COALESCE(cache_read_tokens, 0)     AS cache_read_tokens
            FROM turns
            """
        ).pl()
    if turns.is_empty():
        return pl.DataFrame(schema=_COST_VS_CONTEXT_PER_TURN_SCHEMA)

    priced = _attach_prices(turns, table)
    return (
        priced.with_columns(
            (
                pl.col("input_tokens")
                + pl.col("cache_read_tokens")
                + pl.col("cache_creation_tokens")
            )
            .cast(pl.Int64)
            .alias("context_tokens"),
            (
                pl.col("input_tokens") * pl.col("input_price") / 1_000_000
                + pl.col("output_tokens") * pl.col("output_price") / 1_000_000
                + pl.col("cache_creation_tokens")
                * pl.col("cache_creation_price")
                / 1_000_000
                + pl.col("cache_read_tokens") * pl.col("cache_read_price") / 1_000_000
            ).alias("est_usd"),
        )
        .select(list(_COST_VS_CONTEXT_PER_TURN_SCHEMA.keys()))
        .sort(["session_id", "turn_index"])
    )


_CACHE_SAVINGS_SCHEMA: dict[str, type[pl.DataType]] = {
    "session_id": pl.String,
    "cache_read_tokens": pl.Int64,
    "would_have_paid_usd": pl.Float64,
    "actually_paid_usd": pl.Float64,
    "savings_usd": pl.Float64,
}


def cache_savings(
    con_or_path: ConnLike,
    prices: PriceTable | None = None,
) -> pl.DataFrame:
    """Estimate USD saved per session by serving cached prompt tokens.

    Savings = ``cache_read_tokens * (input_price - cache_read_price)``,
    valued at the price of each turn's own model.

    Args:
        con_or_path: A DuckDB connection or a path to the stats DB.
        prices: Optional override for the default packaged price table.

    Returns:
        DataFrame with columns ``session_id, cache_read_tokens,
        would_have_paid_usd, actually_paid_usd, savings_usd``. Sessions with
        no cache reads are omitted. Returns an empty DataFrame with this
        schema when no turns match.
    """
    table = prices if prices is not None else default_prices()
    with as_connection(con_or_path) as conn:
        turns = conn.execute(
            """
            SELECT
                session_id,
                turn_index,
                model,
                COALESCE(cache_read_tokens, 0) AS cache_read_tokens
            FROM turns
            WHERE COALESCE(cache_read_tokens, 0) > 0
            """
        ).pl()
    if turns.is_empty():
        return pl.DataFrame(schema=_CACHE_SAVINGS_SCHEMA)

    priced = _attach_prices(turns, table)
    out = (
        priced.with_columns(
            (pl.col("cache_read_tokens") * pl.col("input_price") / 1_000_000).alias(
                "would_have_paid_usd"
            ),
            (pl.col("cache_read_tokens") * pl.col("cache_read_price") / 1_000_000).alias(
                "actually_paid_usd"
            ),
        )
        .with_columns(
            (pl.col("would_have_paid_usd") - pl.col("actually_paid_usd")).alias("savings_usd")
        )
        .group_by("session_id")
        .agg(
            pl.col("cache_read_tokens").sum(),
            pl.col("would_have_paid_usd").sum(),
            pl.col("actually_paid_usd").sum(),
            pl.col("savings_usd").sum(),
        )
        .sort("savings_usd", descending=True)
        .select(list(_CACHE_SAVINGS_SCHEMA.keys()))
    )
    return out
