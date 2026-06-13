"""Streamlit entry point for the optional dashboard surface.

Run via ``claudegnostic dashboard`` rather than invoking ``streamlit run``
directly; the launcher sets the DB-path environment variable for us.
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


def main() -> None:
    st.set_page_config(page_title="claudegnostic", layout="wide")
    st.title("claudegnostic dashboard")

    db_path = _resolve_db_path()
    st.caption(f"Database: `{db_path}`")

    if not db_exists(db_path):
        st.warning(
            "No database found. Run `claudegnostic ingest` to populate it."
        )
        return

    conn = get_conn(str(db_path))

    if is_db_empty(conn):
        empty_state()
        return

    options = get_filter_options(str(db_path))
    render_sidebar(options)

    st.markdown(
        "Pick a page from the sidebar to drill in. The landing view below "
        "shows your highest-output sessions as a sanity check."
    )

    sessions = conn.execute(
        """
        SELECT
            session_id,
            cwd,
            turn_count,
            total_input_tokens,
            total_output_tokens,
            cache_hit_ratio
        FROM sessions
        ORDER BY total_output_tokens DESC NULLS LAST
        LIMIT 100
        """
    ).fetchdf()

    st.subheader("Top sessions by output tokens")
    st.dataframe(sessions, use_container_width=True)


if __name__ == "__main__":
    main()
