"""DuckDB connection and filter-option caches for the dashboard.

The dashboard owns one read-only DuckDB connection per db_path, cached via
``@st.cache_resource`` so it survives reruns. Analysis functions accept this
connection directly through ``ConnLike``.

The cached entry points (``get_conn``, ``get_filter_options``) delegate to
plain helpers (``open_conn``, ``compute_filter_options``) so unit tests can
exercise the same logic without a live Streamlit runtime.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypedDict

import duckdb
import streamlit as st


class FilterOptions(TypedDict):
    """Distinct values used to populate sidebar widgets."""

    cwds: list[str]
    models: list[str]
    date_min: date | None
    date_max: date | None


def open_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection for ``db_path``."""
    return duckdb.connect(db_path, read_only=True)


def compute_filter_options(conn: duckdb.DuckDBPyConnection) -> FilterOptions:
    """Distinct cwds, models, and date bounds available in ``conn``."""
    cwds = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT cwd FROM turns WHERE cwd IS NOT NULL ORDER BY cwd"
        ).fetchall()
    ]
    models = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT model FROM turns WHERE model IS NOT NULL ORDER BY model"
        ).fetchall()
    ]
    bounds = conn.execute(
        "SELECT MIN(CAST(timestamp AS DATE)), MAX(CAST(timestamp AS DATE)) FROM turns"
    ).fetchone()
    date_min, date_max = (bounds[0], bounds[1]) if bounds is not None else (None, None)
    return FilterOptions(
        cwds=cwds,
        models=models,
        date_min=date_min,
        date_max=date_max,
    )


@st.cache_resource(show_spinner=False)
def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    """Process-wide read-only DuckDB connection for ``db_path``.

    Cached on ``db_path`` (string for hashability). The connection is reused
    across Streamlit reruns; Streamlit closes it on cache eviction or app
    shutdown.
    """
    return open_conn(db_path)


@st.cache_resource(show_spinner=False)
def get_filter_options(db_path: str) -> FilterOptions:
    """Cached wrapper around ``compute_filter_options`` keyed on db_path."""
    return compute_filter_options(get_conn(db_path))


def db_exists(db_path: Path) -> bool:
    """Whether the configured DuckDB file is present on disk."""
    return db_path.exists()


def is_db_empty(conn: duckdb.DuckDBPyConnection) -> bool:
    """True if neither sessions nor turns has any rows."""
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()
    s = sessions[0] if sessions else 0
    t = turns[0] if turns else 0
    return s == 0 and t == 0
