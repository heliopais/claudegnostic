"""Streamlit entry point for the optional dashboard surface.

Run via ``claudegnostic dashboard`` rather than invoking this file directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from claudegnostic.dashboard.cli import DB_PATH_ENV
from claudegnostic.storage import connect, default_db_path


def _resolve_db_path() -> Path:
    raw = os.environ.get(DB_PATH_ENV)
    return Path(raw) if raw else default_db_path()


def main() -> None:
    st.set_page_config(page_title="claudegnostic", layout="wide")
    st.title("claudegnostic dashboard")

    db_path = _resolve_db_path()
    st.caption(f"Database: `{db_path}`")

    if not db_path.exists():
        st.warning(
            "No database found. Run `claudegnostic ingest` first to populate it."
        )
        return

    with connect(db_path) as conn:
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
            """,
        ).fetchdf()

    if sessions.empty:
        st.info("Database is empty. Ingest some session files to see data here.")
        return

    st.subheader("Top sessions by output tokens")
    st.dataframe(sessions, use_container_width=True)


if __name__ == "__main__":
    main()
