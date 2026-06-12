"""Tests for storage module: connection lifecycle and schema bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from claudegnostic import storage


def _table_names(conn: object) -> set[str]:
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {row[0] for row in rows}


def test_connect_creates_db_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "stats.duckdb"
    with storage.connect(db_path) as conn:
        assert _table_names(conn) >= {"turns", "sessions"}
    assert db_path.exists()


def test_reopen_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "stats.duckdb"
    with storage.connect(db_path) as conn:
        conn.execute("INSERT INTO sessions (session_id, turn_count) VALUES ('s1', 3)")
    with storage.connect(db_path) as conn:
        assert _table_names(conn) >= {"turns", "sessions"}
        row = conn.execute(
            "SELECT session_id, turn_count FROM sessions WHERE session_id = 's1'"
        ).fetchone()
        assert row == ("s1", 3)


def test_default_db_path_under_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_db_path() == tmp_path / "claudegnostic" / "stats.duckdb"


def test_default_db_path_without_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = tmp_path / ".local" / "share" / "claudegnostic" / "stats.duckdb"
    assert storage.default_db_path() == expected
