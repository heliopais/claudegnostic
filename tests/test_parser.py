"""Tests for session JSONL parser."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from claudegnostic.parser import TURNS_SCHEMA, parse_session


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _assistant(
    *,
    uuid: str,
    timestamp: str,
    content: list[dict],
    usage: dict | None = None,
    parent_uuid: str | None = None,
    is_sidechain: bool = False,
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-6",
    cwd: str = "/tmp",
    git_branch: str = "main",
    session_id: str = "sess-1",
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
        "cwd": cwd,
        "gitBranch": git_branch,
        "sessionId": session_id,
        "isSidechain": is_sidechain,
        "message": {
            "model": model,
            "stop_reason": stop_reason,
            "usage": usage or {},
            "content": content,
        },
    }


def _user(
    *,
    uuid: str,
    timestamp: str,
    content,
    parent_uuid: str | None = None,
    is_sidechain: bool = False,
    is_compact_summary: bool = False,
    session_id: str = "sess-1",
) -> dict:
    event = {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "timestamp": timestamp,
        "sessionId": session_id,
        "isSidechain": is_sidechain,
        "message": {"role": "user", "content": content},
    }
    if is_compact_summary:
        event["isCompactSummary"] = True
    return event


def test_schema_matches_polars_dtypes(tmp_path: Path) -> None:
    fixture = tmp_path / "empty.jsonl"
    _write_jsonl(fixture, [])
    df = parse_session(fixture)
    assert df.is_empty()
    assert dict(df.schema) == TURNS_SCHEMA


def test_basic_turn_extraction(tmp_path: Path) -> None:
    fixture = tmp_path / "basic.jsonl"
    events = [
        _user(uuid="u1", timestamp="2026-06-12T10:00:00Z", content="hi"),
        _assistant(
            uuid="a1",
            timestamp="2026-06-12T10:00:01Z",
            content=[
                {"type": "thinking", "thinking": "let me think"},
                {"type": "text", "text": "hello world"},
            ],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 200,
                "service_tier": "standard",
            },
            stop_reason="end_turn",
        ),
        _user(
            uuid="u2",
            timestamp="2026-06-12T10:00:03Z",
            content="next",
        ),
    ]
    _write_jsonl(fixture, events)

    df = parse_session(fixture)
    assert df.shape == (1, len(TURNS_SCHEMA))
    row = df.row(0, named=True)
    assert row["turn_index"] == 0
    assert row["uuid"] == "a1"
    assert row["model"] == "claude-sonnet-4-6"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["cache_creation_tokens"] == 50
    assert row["cache_read_tokens"] == 200
    assert row["service_tier"] == "standard"
    assert row["stop_reason"] == "end_turn"
    assert row["text_output_chars"] == len("hello world")
    assert row["thinking_chars"] == len("let me think")
    assert row["tool_call_count"] == 0
    assert row["tool_names"] == []
    assert row["tool_input_bytes"] == 0
    assert row["tool_result_bytes"] == 0
    assert row["wall_duration_ms"] == 2000
    assert row["is_sidechain"] is False
    assert row["is_compact_summary"] is False


def test_forward_pair_tool_result_bytes(tmp_path: Path) -> None:
    fixture = tmp_path / "tools.jsonl"
    tool_input = {"path": "/tmp/x"}
    tool_result_payload = "ABCDE" * 4
    events = [
        _assistant(
            uuid="a1",
            timestamp="2026-06-12T10:00:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": tool_input,
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "Grep",
                    "input": {"pattern": "foo"},
                },
            ],
            stop_reason="tool_use",
        ),
        _user(
            uuid="u1",
            timestamp="2026-06-12T10:00:01Z",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": tool_result_payload,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "x",
                },
            ],
        ),
    ]
    _write_jsonl(fixture, events)

    df = parse_session(fixture)
    row = df.row(0, named=True)
    assert row["tool_call_count"] == 2
    assert row["tool_names"] == ["Read", "Grep"]
    assert row["tool_input_bytes"] == (
        len(json.dumps(tool_input, separators=(",", ":")))
        + len(json.dumps({"pattern": "foo"}, separators=(",", ":")))
    )
    # tool_result_bytes is the JSON byte-length of the content payload.
    expected = (
        len(json.dumps(tool_result_payload, separators=(",", ":")))
        + len(json.dumps("x", separators=(",", ":")))
    )
    assert row["tool_result_bytes"] == expected
    assert row["wall_duration_ms"] == 1000


def test_sidechain_and_compaction_flags(tmp_path: Path) -> None:
    fixture = tmp_path / "flags.jsonl"
    events = [
        _user(
            uuid="u-compact",
            timestamp="2026-06-12T09:00:00Z",
            content="summary",
            is_compact_summary=True,
        ),
        _assistant(
            uuid="a1",
            timestamp="2026-06-12T10:00:00Z",
            content=[{"type": "text", "text": "side"}],
            is_sidechain=True,
        ),
        _assistant(
            uuid="a2",
            timestamp="2026-06-12T10:00:01Z",
            content=[{"type": "text", "text": "main"}],
            is_sidechain=False,
        ),
    ]
    # Tag the second assistant as compact-summary directly to verify the flag
    # is read at the top-level of the assistant event itself.
    events[2]["isCompactSummary"] = True
    _write_jsonl(fixture, events)

    df = parse_session(fixture).sort("turn_index")
    # Three rows now: the type=="user" compact-summary event also emits a turn.
    # Order: compact-summary user, sidechain assistant, compact-flagged assistant.
    assert df["is_sidechain"].to_list() == [False, True, False]
    assert df["is_compact_summary"].to_list() == [True, False, True]
    # The user-event-shaped compact-summary row has no model and no tokens.
    summary_row = df.row(0, named=True)
    assert summary_row["model"] is None
    assert summary_row["input_tokens"] is None
    assert summary_row["tool_call_count"] == 0


def test_malformed_lines_logged_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = tmp_path / "broken.jsonl"
    good = _assistant(
        uuid="a1",
        timestamp="2026-06-12T10:00:00Z",
        content=[{"type": "text", "text": "ok"}],
    )
    with fixture.open("w", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")
        fh.write(json.dumps(good) + "\n")
        fh.write("also not json}\n")

    with caplog.at_level(logging.WARNING, logger="claudegnostic.parser"):
        df = parse_session(fixture)

    assert df.shape == (1, len(TURNS_SCHEMA))
    assert df.row(0, named=True)["uuid"] == "a1"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
