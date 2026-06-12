"""Connection adapter so analysis functions accept a connection or a path."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

ConnLike = duckdb.DuckDBPyConnection | str | Path


@contextmanager
def as_connection(con_or_path: ConnLike) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection, closing only what we opened.

    A path is opened read-only and closed on exit; a connection is yielded
    untouched. Analysis modules use this so callers can pass either shape.
    """
    if isinstance(con_or_path, duckdb.DuckDBPyConnection):
        yield con_or_path
        return
    conn = duckdb.connect(str(con_or_path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()
