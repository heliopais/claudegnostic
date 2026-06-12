"""DuckDB connection lifecycle and on-disk location."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from claudegnostic.schema import apply_schema


def default_db_path() -> Path:
    """Return the default on-disk path for the stats database.

    Honors XDG_DATA_HOME; falls back to ~/.local/share.
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "claudegnostic" / "stats.duckdb"


@contextmanager
def connect(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the stats DuckDB, ensure schema exists, yield a connection."""
    db_path = path if path is not None else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        apply_schema(conn)
        yield conn
    finally:
        conn.close()
