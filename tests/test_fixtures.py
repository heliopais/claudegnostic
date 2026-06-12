"""Tests driven by on-disk JSONL fixtures under tests/fixtures/sessions/.

These complement the inline-built fixtures in test_parser.py / test_ingest.py
by exercising the parser and ingest pipeline against persisted files that
mirror real Claude Code session shapes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from claudegnostic import storage
from claudegnostic.ingest import ingest_paths, ingest_root
from claudegnostic.parser import TURNS_SCHEMA, parse_session

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"


def test_fixtures_dir_present() -> None:
    expected = {"normal", "sidechain", "compaction", "interrupted", "malformed"}
    found = {p.stem for p in FIXTURES.glob("*.jsonl")}
    assert expected <= found


def test_normal_fixture_shape_and_tool_result_bytes() -> None:
    df = parse_session(FIXTURES / "normal.jsonl").sort("turn_index")
    assert dict(df.schema) == TURNS_SCHEMA
    assert df.height == 2

    first = df.row(0, named=True)
    assert first["session_id"] == "sess-normal"
    assert first["uuid"] == "a1"
    assert first["stop_reason"] == "end_turn"
    assert first["tool_call_count"] == 0
    assert first["tool_result_bytes"] == 0

    second = df.row(1, named=True)
    assert second["uuid"] == "a2"
    assert second["tool_call_count"] == 1
    assert second["tool_names"] == ["Read"]
    # Forward-pair attribution: tool_result for toolu_1 in the next user event
    # contributes a non-zero byte count.
    assert second["tool_result_bytes"] > 0
    assert second["wall_duration_ms"] == 1000


def test_sidechain_fixture_flags_propagate() -> None:
    df = parse_session(FIXTURES / "sidechain.jsonl").sort("turn_index")
    assert df.height == 2
    flags = df["is_sidechain"].to_list()
    assert flags == [False, True]
    assert df["session_id"].unique().to_list() == ["sess-side"]


def test_compaction_fixture_flag_propagates() -> None:
    df = parse_session(FIXTURES / "compaction.jsonl").sort("turn_index")
    assert df.height == 2
    compact_flags = df["is_compact_summary"].to_list()
    assert compact_flags == [True, False]


def test_interrupted_fixture_stop_reason() -> None:
    df = parse_session(FIXTURES / "interrupted.jsonl")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["stop_reason"] == "max_tokens"
    assert row["stop_reason"] != "end_turn"


def test_malformed_fixture_logs_and_recovers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="claudegnostic.parser"):
        df = parse_session(FIXTURES / "malformed.jsonl")

    assert df.height == 1
    assert df.row(0, named=True)["uuid"] == "a1"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_ingest_normal_fixture_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "stats.duckdb"
    fixture = FIXTURES / "normal.jsonl"

    with storage.connect(db_path) as conn:
        first = ingest_paths(conn, [fixture])
        assert first.files_scanned == 1
        assert first.turns_added == 2
        assert first.turns_updated == 0

        turns_after_first = conn.execute("SELECT count(*) FROM turns").fetchone()[0]
        sessions_after_first = conn.execute(
            "SELECT count(*) FROM sessions"
        ).fetchone()[0]

        second = ingest_paths(conn, [fixture])
        assert second.turns_added == 0
        assert second.turns_updated == 2

        assert conn.execute("SELECT count(*) FROM turns").fetchone()[0] == turns_after_first
        assert (
            conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
            == sessions_after_first
        )


def test_ingest_root_picks_up_all_fixtures(tmp_path: Path) -> None:
    staged = tmp_path / "sessions"
    shutil.copytree(FIXTURES, staged)
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        report = ingest_root(conn, staged)
        assert report.files_scanned == 5
        session_ids = {
            row[0] for row in conn.execute("SELECT session_id FROM sessions").fetchall()
        }
        # malformed.jsonl still yields one valid assistant turn for sess-bad.
        assert session_ids == {
            "sess-normal",
            "sess-side",
            "sess-compact",
            "sess-int",
            "sess-bad",
        }


def test_resumed_session_adds_one_turn(tmp_path: Path) -> None:
    """Append a new assistant turn to a copy of normal.jsonl and re-ingest."""
    staged = tmp_path / "normal.jsonl"
    shutil.copyfile(FIXTURES / "normal.jsonl", staged)
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        ingest_paths(conn, [staged])
        before = conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = 'sess-normal'"
        ).fetchone()[0]
        assert before == 2

        appended_event = (
            '{"type": "assistant", "sessionId": "sess-normal", "uuid": "a3", '
            '"parentUuid": "u3", "timestamp": "2026-06-12T10:00:06Z", '
            '"cwd": "/repo", "gitBranch": "main", "isSidechain": false, '
            '"message": {"role": "assistant", "model": "claude-sonnet-4-6", '
            '"content": [{"type": "text", "text": "follow up"}], '
            '"stop_reason": "end_turn", '
            '"usage": {"input_tokens": 5, "output_tokens": 2}}}\n'
        )
        with staged.open("a", encoding="utf-8") as fh:
            fh.write(appended_event)

        report = ingest_paths(conn, [staged])
        assert report.turns_added == 1
        assert report.turns_updated == 2
        assert report.sessions_touched == 1

        after = conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = 'sess-normal'"
        ).fetchone()[0]
        assert after == 3
