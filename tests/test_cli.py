"""Smoke and end-to-end tests for the CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from claudegnostic import __version__
from claudegnostic.cli import app


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "claudegnostic" in result.stdout.lower()


def test_ingest_and_stats_help_documented(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "stats" in result.stdout


def _write_session(path: Path, session_id: str = "sess-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-06-12T10:00:00Z",
            "sessionId": session_id,
            "isSidechain": False,
            "message": {"role": "user", "content": "hi"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-06-12T10:00:01Z",
            "cwd": "/tmp/work",
            "gitBranch": "main",
            "sessionId": session_id,
            "isSidechain": False,
            "message": {
                "model": "claude-sonnet-4-6",
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 200,
                },
                "content": [
                    {"type": "text", "text": "calling a tool"},
                    {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "ls"}},
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "timestamp": "2026-06-12T10:00:02Z",
            "sessionId": session_id,
            "isSidechain": False,
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def test_ingest_end_to_end(runner: CliRunner, tmp_path: Path) -> None:
    sessions_root = tmp_path / "projects"
    _write_session(sessions_root / "proj-a" / "sess-1.jsonl", session_id="sess-1")
    db_path = tmp_path / "stats.duckdb"

    result = runner.invoke(app, ["ingest", "--root", str(sessions_root), "--db", str(db_path)])

    assert result.exit_code == 0, result.stdout
    assert "Ingest summary" in result.stdout
    assert "Files scanned" in result.stdout
    assert db_path.exists()


def test_stats_renders_all_tables(runner: CliRunner, tmp_path: Path) -> None:
    sessions_root = tmp_path / "projects"
    _write_session(sessions_root / "proj-a" / "sess-1.jsonl", session_id="sess-1")
    _write_session(sessions_root / "proj-b" / "sess-2.jsonl", session_id="sess-2")
    db_path = tmp_path / "stats.duckdb"

    ingest_res = runner.invoke(
        app, ["ingest", "--root", str(sessions_root), "--db", str(db_path)]
    )
    assert ingest_res.exit_code == 0, ingest_res.stdout

    result = runner.invoke(app, ["stats", "--db", str(db_path), "--top", "5"])

    assert result.exit_code == 0, result.stdout
    assert "Top 5 sessions by output tokens" in result.stdout
    assert "Cache hit ratio" in result.stdout
    assert "Top 5 tools by call count" in result.stdout
    assert "Compaction summary" in result.stdout


def test_stats_on_empty_db(runner: CliRunner, tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    result = runner.invoke(app, ["stats", "--db", str(db_path)])
    assert result.exit_code == 0, result.stdout
    assert "Cache hit ratio" in result.stdout
