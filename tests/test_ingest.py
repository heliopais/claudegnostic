"""Tests for ingest: discovery, idempotent upsert, sessions recompute."""

from __future__ import annotations

import json
from pathlib import Path

from claudegnostic import storage
from claudegnostic.ingest import (
    default_sessions_root,
    discover_sessions,
    ingest_paths,
    ingest_root,
)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _assistant(
    *,
    uuid: str,
    timestamp: str,
    text: str,
    session_id: str = "sess-1",
    cwd: str = "/tmp",
    git_branch: str = "main",
    model: str = "claude-sonnet-4-6",
    usage: dict | None = None,
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": timestamp,
        "cwd": cwd,
        "gitBranch": git_branch,
        "sessionId": session_id,
        "isSidechain": False,
        "message": {
            "model": model,
            "stop_reason": "end_turn",
            "usage": usage
            or {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "content": [{"type": "text", "text": text}],
        },
    }


def _user(*, uuid: str, timestamp: str, text: str, session_id: str = "sess-1") -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": None,
        "timestamp": timestamp,
        "sessionId": session_id,
        "isSidechain": False,
        "message": {"role": "user", "content": text},
    }


def _session_one(session_id: str = "sess-1") -> list[dict]:
    return [
        _user(uuid="u1", timestamp="2026-06-12T10:00:00Z", text="hi", session_id=session_id),
        _assistant(
            uuid="a1",
            timestamp="2026-06-12T10:00:01Z",
            text="one",
            session_id=session_id,
        ),
        _user(uuid="u2", timestamp="2026-06-12T10:00:02Z", text="again", session_id=session_id),
        _assistant(
            uuid="a2",
            timestamp="2026-06-12T10:00:03Z",
            text="two",
            session_id=session_id,
        ),
        _user(uuid="u3", timestamp="2026-06-12T10:00:04Z", text="bye", session_id=session_id),
    ]


def test_discover_sessions_recursive(tmp_path: Path) -> None:
    a = tmp_path / "proj-a" / "s1.jsonl"
    b = tmp_path / "proj-b" / "nested" / "s2.jsonl"
    other = tmp_path / "proj-a" / "ignore.txt"
    for p in (a, b, other):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")

    found = sorted(discover_sessions(tmp_path))
    assert found == sorted([a, b])


def test_discover_sessions_missing_root(tmp_path: Path) -> None:
    assert list(discover_sessions(tmp_path / "does-not-exist")) == []


def test_default_sessions_root_under_home(monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/x")))
    assert default_sessions_root() == Path("/x/.claude/projects")


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    fixture = tmp_path / "sessions" / "s1.jsonl"
    _write_jsonl(fixture, _session_one())
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        first = ingest_paths(conn, [fixture])
        assert first.files_scanned == 1
        assert first.turns_added == 2
        assert first.turns_updated == 0
        assert first.sessions_touched == 1

        turns_after_first = conn.execute("SELECT count(*) FROM turns").fetchone()[0]
        sessions_after_first = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        assert turns_after_first == 2
        assert sessions_after_first == 1

        second = ingest_paths(conn, [fixture])
        assert second.turns_added == 0
        assert second.turns_updated == 2
        assert second.sessions_touched == 1

        turns_after_second = conn.execute("SELECT count(*) FROM turns").fetchone()[0]
        sessions_after_second = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        assert turns_after_second == turns_after_first
        assert sessions_after_second == sessions_after_first


def test_resumed_session_adds_one_turn(tmp_path: Path) -> None:
    fixture = tmp_path / "s1.jsonl"
    _write_jsonl(fixture, _session_one())
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        ingest_paths(conn, [fixture])

        # Resume: append one more assistant turn to the same file.
        appended = [
            *_session_one(),
            _assistant(
                uuid="a3",
                timestamp="2026-06-12T10:00:05Z",
                text="three",
            ),
        ]
        _write_jsonl(fixture, appended)

        report = ingest_paths(conn, [fixture])
        assert report.turns_added == 1
        assert report.turns_updated == 2
        assert report.sessions_touched == 1

        turn_count = conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()[0]
        assert turn_count == 3


def test_sessions_aggregate_matches_turns(tmp_path: Path) -> None:
    fixture = tmp_path / "s1.jsonl"
    _write_jsonl(fixture, _session_one())
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        ingest_paths(conn, [fixture])
        row = conn.execute(
            "SELECT turn_count, total_input_tokens, total_output_tokens, "
            "started_at, ended_at, models_used "
            "FROM sessions WHERE session_id = 'sess-1'"
        ).fetchone()
        agg = conn.execute(
            "SELECT count(*), sum(input_tokens), sum(output_tokens), "
            "min(timestamp), max(timestamp) "
            "FROM turns WHERE session_id = 'sess-1'"
        ).fetchone()

        assert row[0] == agg[0]
        assert row[1] == agg[1]
        assert row[2] == agg[2]
        assert row[3] == agg[3]
        assert row[4] == agg[4]
        assert row[5] == ["claude-sonnet-4-6"]


def test_ingest_root_walks_tree(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "proj-a" / "s1.jsonl",
        _session_one(session_id="sess-a"),
    )
    _write_jsonl(
        tmp_path / "proj-b" / "s2.jsonl",
        _session_one(session_id="sess-b"),
    )
    db_path = tmp_path / "stats.duckdb"

    with storage.connect(db_path) as conn:
        report = ingest_root(conn, tmp_path)
        assert report.files_scanned == 2
        assert report.turns_added == 4
        assert report.sessions_touched == 2

        ids = {row[0] for row in conn.execute("SELECT session_id FROM sessions").fetchall()}
        assert ids == {"sess-a", "sess-b"}


def test_ingest_skips_unparseable_file(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text("not even json\n")
    db_path = tmp_path / "stats.duckdb"

    from claudegnostic import ingest as ingest_mod

    def boom(_path):
        raise RuntimeError("simulated parse failure")

    monkeypatch.setattr(ingest_mod, "parse_session", boom)

    with storage.connect(db_path) as conn:
        report = ingest_paths(conn, [fixture])
        assert report.files_scanned == 1
        assert report.turns_added == 0
        assert report.files_failed == [fixture]
