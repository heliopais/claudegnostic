"""Overview page: a high-level look at the ingested data.

The page beads (cost, productivity, archaeology) will follow the same
shape: resolve db_path, get a cached connection, gate on empty, render
the sidebar, then draw charts.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

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


def _resolve_db_path() -> Path:
    raw = os.environ.get(DB_PATH_ENV)
    return Path(raw) if raw else default_db_path()


st.title("Overview")

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

        totals = conn.execute(
            """
            SELECT
                COUNT(*)::BIGINT                            AS session_count,
                COALESCE(SUM(turn_count), 0)::BIGINT        AS turn_count,
                COALESCE(SUM(total_input_tokens), 0)::BIGINT  AS input_tokens,
                COALESCE(SUM(total_output_tokens), 0)::BIGINT AS output_tokens
            FROM sessions
            """
        ).fetchone()
        sc, tc, it, ot = totals if totals else (0, 0, 0, 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions", f"{sc:,}")
        c2.metric("Turns", f"{tc:,}")
        c3.metric("Input tokens", f"{it:,}")
        c4.metric("Output tokens", f"{ot:,}")
