"""Productivity page: cache hits, per-tool wall time, wasted turns.

Renders the three productivity lenses against the live DuckDB connection,
gated by the standard empty-state. Charts are plotnine (per project
convention); each handles its own sparse-data case so a brand-new user
with a handful of sessions sees a clear note instead of a broken plot.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import duckdb
import polars as pl
import streamlit as st
from plotnine import (
    aes,
    element_text,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_discrete,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

from claudegnostic.analysis.productivity import (
    cache_hit_ratio_by_session,
    wall_time_per_tool,
    wasted_turns,
)
from claudegnostic.dashboard._db import (
    db_exists,
    get_conn,
    get_filter_options,
    is_db_empty,
)
from claudegnostic.dashboard._empty import empty_state
from claudegnostic.dashboard._filters import render_sidebar
from claudegnostic.dashboard.cli import DB_PATH_ENV
from claudegnostic.storage import default_db_path

_MIN_SESSIONS_FOR_CHARTS = 3
_TOP_N_TOOLS = 12
_BUCKET_ORDER = ["<0.25", "0.25-0.5", "0.5-0.75", ">=0.75"]
_BUCKET_COLORS = {
    "<0.25": "#C44536",
    "0.25-0.5": "#E08E45",
    "0.5-0.75": "#5B8C5A",
    ">=0.75": "#22577A",
}


def _resolve_db_path() -> Path:
    raw = os.environ.get(DB_PATH_ENV)
    return Path(raw) if raw else default_db_path()


def _wall_time_plot(df: pl.DataFrame, top_n: int = _TOP_N_TOOLS) -> ggplot | None:
    """Range plot (p50 -> p90) per tool, sorted by call count.

    The underlying analysis returns aggregated p50/p90/count rather than
    raw durations, so this is a lollipop-style range chart instead of a
    true ridge. The shape is still the obvious "which tools dominate wall
    time" read.
    """
    if df.is_empty():
        return None
    sub = df.head(top_n).with_columns(
        pl.col("p50_ms").cast(pl.Float64),
        pl.col("p90_ms").cast(pl.Float64),
    )
    tool_order = list(sub.sort("p90_ms", descending=False)["tool"].to_list())
    plot: ggplot = (
        ggplot(sub, aes(x="tool"))
        + geom_segment(aes(xend="tool", y="p50_ms", yend="p90_ms"), color="#22577A", size=2)
        + geom_point(aes(y="p50_ms"), color="#22577A", size=3)
        + geom_point(aes(y="p90_ms"), color="#C44536", size=3)
        + scale_x_discrete(limits=tool_order)
        + scale_y_log10()
        + labs(
            title=f"Per-tool wall time (top {top_n} by call count)",
            subtitle="Blue = p50, red = p90 (log scale, ms). Coarse heuristic: turn duration "
            "attributed in full to every tool in the turn.",
            x="",
            y="Wall duration (ms)",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    return plot


def _cache_vs_length_plot(
    conn: duckdb.DuckDBPyConnection, ratios: pl.DataFrame
) -> ggplot | None:
    """Scatter: cache hit ratio vs session length, colored by bucket."""
    if ratios.is_empty():
        return None
    lengths = conn.execute(
        "SELECT session_id, turn_count FROM sessions WHERE turn_count IS NOT NULL"
    ).pl()
    if lengths.is_empty():
        return None
    joined = ratios.join(lengths, on="session_id", how="inner").filter(
        pl.col("turn_count") > 0
    )
    if joined.is_empty():
        return None
    plot: ggplot = (
        ggplot(joined, aes(x="turn_count", y="ratio", color="ratio_bucket"))
        + geom_point(size=3, alpha=0.75)
        + scale_x_log10()
        + scale_color_manual(values=_BUCKET_COLORS, breaks=_BUCKET_ORDER)
        + labs(
            title="Cache hit ratio vs session length",
            x="Turns per session (log)",
            y="Cache hit ratio",
            color="Bucket",
        )
        + theme_minimal()
    )
    return plot


def _wasted_turns_table(
    conn: duckdb.DuckDBPyConnection, wasted: pl.DataFrame
) -> pl.DataFrame:
    """Join wasted-turn rows with each turn's cwd so outliers are findable."""
    if wasted.is_empty():
        return wasted
    cwds = conn.execute(
        "SELECT session_id, turn_index, cwd FROM turns"
    ).pl()
    return wasted.join(cwds, on=["session_id", "turn_index"], how="left").select(
        "session_id",
        "cwd",
        "turn_index",
        "input_tokens",
        "tool_call_count",
        "text_output_chars",
    )


def _render(plot: ggplot | None, *, sparse_msg: str) -> None:
    if plot is None:
        empty_state(sparse_msg)
        return
    st.pyplot(plot.draw(), clear_figure=True)


st.title("Productivity")

db_path = _resolve_db_path()
st.caption(f"Database: `{db_path}`")

if not db_exists(db_path):
    empty_state("No database found. Run `claudegnostic ingest` first.")
else:
    conn = get_conn(str(db_path))
    if is_db_empty(conn):
        empty_state()
    else:
        options = get_filter_options(str(db_path))
        render_sidebar(options)

        session_count_row = conn.execute("SELECT COUNT(*)::BIGINT FROM sessions").fetchone()
        session_count = int(session_count_row[0]) if session_count_row else 0

        if session_count < _MIN_SESSIONS_FOR_CHARTS:
            empty_state(
                f"Need at least {_MIN_SESSIONS_FOR_CHARTS} sessions to draw "
                f"productivity charts — found {session_count}. "
                "Run more sessions or `claudegnostic ingest` again."
            )
        else:
            st.subheader("Per-tool wall time")
            _render(
                _wall_time_plot(wall_time_per_tool(conn)),
                sparse_msg="No tool-bearing turns with a recorded wall duration yet.",
            )

            st.subheader("Cache hit ratio vs session length")
            _render(
                _cache_vs_length_plot(conn, cache_hit_ratio_by_session(conn)),
                sparse_msg="No sessions with prompt traffic to compute cache ratios yet.",
            )

            st.subheader("Wasted turns")
            wasted = _wasted_turns_table(conn, wasted_turns(conn))
            if wasted.is_empty():
                empty_state(
                    "No wasted turns detected — every high-input turn either ran a tool "
                    "or produced meaningful text."
                )
            else:
                st.dataframe(wasted.to_pandas(), use_container_width=True)
