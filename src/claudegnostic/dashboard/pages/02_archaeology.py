"""Archaeology page: shape and rhythm of sessions over time.

Renders the four archaeology lenses against the live DuckDB connection,
gated by the standard empty-state. Charts are plotnine (per project
convention); each handles its own sparse-data case so a brand-new user
with a handful of sessions sees a clear note instead of a broken plot.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl
import streamlit as st
from plotnine import (
    aes,
    element_text,
    geom_bar,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from claudegnostic.analysis.archaeology import (
    SESSION_LENGTH_BUCKET_ORDER,
    project_activity_by_day,
    session_length_distribution,
    sidechain_ratio_by_session,
    tool_co_occurrence,
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
_TOP_N_SIDECHAIN = 20


def _resolve_db_path() -> Path:
    raw = os.environ.get(DB_PATH_ENV)
    return Path(raw) if raw else default_db_path()


def _session_length_plot(df: pl.DataFrame) -> ggplot | None:
    if df.is_empty() or df["count"].sum() == 0:
        return None
    labelled = df.with_columns(
        pl.format(
            "${}/session\n${} ({}%)",
            pl.col("cost_per_session_usd").round(2),
            pl.col("total_cost_usd").round(0).cast(pl.Int64),
            pl.col("pct_total_cost").round(0).cast(pl.Int64),
        ).alias("label")
    )
    raw_max = labelled["count"].cast(pl.Float64).max()
    y_max = float(raw_max) if isinstance(raw_max, (int, float)) else 0.0
    headroom = y_max * 1.22 if y_max > 0 else 1.0
    return (
        ggplot(labelled, aes(x="bucket", y="count"))
        + geom_bar(stat="identity", fill="#22577A")
        + geom_text(
            aes(label="label"),
            va="bottom",
            size=8,
            nudge_y=y_max * 0.02 if y_max > 0 else 0.05,
        )
        + scale_x_discrete(limits=list(SESSION_LENGTH_BUCKET_ORDER))
        + scale_y_continuous(limits=(0, headroom))
        + labs(title="Session length distribution", x="Turns per session", y="Sessions")
        + theme_minimal()
    )


def _tool_cooccurrence_plot(df: pl.DataFrame, top_n: int = _TOP_N_TOOLS) -> ggplot | None:
    if df.is_empty():
        return None
    counts = (
        pl.concat(
            [
                df.select(pl.col("tool_a").alias("tool"), pl.col("pair_count")),
                df.select(pl.col("tool_b").alias("tool"), pl.col("pair_count")),
            ]
        )
        .group_by("tool")
        .agg(pl.col("pair_count").sum().alias("total"))
        .sort("total", descending=True)
        .head(top_n)
    )
    if counts.is_empty():
        return None
    keep = set(counts["tool"].to_list())
    sub = df.filter(pl.col("tool_a").is_in(keep) & pl.col("tool_b").is_in(keep))
    if sub.is_empty():
        return None
    plot: ggplot = (
        ggplot(sub, aes(x="tool_a", y="tool_b", fill="pair_count"))
        + geom_tile()
        + scale_fill_gradient(low="#EEEEEE", high="#22577A")
        + labs(title=f"Tool co-occurrence (top {top_n} tools)", x="", y="", fill="Pair count")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    return plot


def _sidechain_plot(df: pl.DataFrame, top_n: int = _TOP_N_SIDECHAIN) -> ggplot | None:
    if df.is_empty() or df["sidechain_turns"].sum() == 0:
        return None
    sub = df.filter(pl.col("sidechain_turns") > 0).head(top_n)
    if sub.is_empty():
        return None
    ordered = sub.sort("ratio", descending=False)
    session_order = list(ordered["session_id"].to_list())
    plot: ggplot = (
        ggplot(ordered, aes(x="session_id", y="ratio"))
        + geom_bar(stat="identity", fill="#22577A")
        + scale_x_discrete(limits=session_order)
        + labs(
            title=f"Sidechain ratio (top {top_n} sessions)",
            x="Session",
            y="Sidechain turns / total",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    return plot


def _project_activity_plot(df: pl.DataFrame) -> ggplot | None:
    if df.is_empty():
        return None
    plot: ggplot = (
        ggplot(df, aes(x="date", y="cwd", fill="turns"))
        + geom_tile()
        + scale_fill_gradient(low="#EEEEEE", high="#22577A")
        + labs(title="Project activity by day", x="Date", y="Project (cwd)", fill="Turns")
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    return plot


def _render(plot: ggplot | None, *, sparse_msg: str) -> None:
    if plot is None:
        empty_state(sparse_msg)
        return
    st.pyplot(plot.draw(), clear_figure=True)


st.title("Archaeology")

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
                f"archaeology charts — found {session_count}. "
                "Run more sessions or `claudegnostic ingest` again."
            )
        else:
            st.subheader("Session length distribution")
            _render(
                _session_length_plot(session_length_distribution(conn)),
                sparse_msg="Not enough session-length data yet.",
            )

            st.subheader("Tool co-occurrence")
            _render(
                _tool_cooccurrence_plot(tool_co_occurrence(conn)),
                sparse_msg="No turn used two or more distinct tools yet.",
            )

            st.subheader("Sidechain ratio by session")
            _render(
                _sidechain_plot(sidechain_ratio_by_session(conn)),
                sparse_msg="No sidechain turns recorded yet.",
            )

            st.subheader("Project activity by day")
            _render(
                _project_activity_plot(project_activity_by_day(conn)),
                sparse_msg="No dated turns with a cwd yet.",
            )
