"""Streaming parser for Claude Code session JSONL files.

Produces a :class:`polars.DataFrame` of turn rows matching the ``turns`` schema
in :mod:`claudegnostic.schema`. One row per ``type=="assistant"`` event in the
file. Tool result bytes are attributed via a forward-pair pass over the
following user events, matched on ``tool_use_id``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


TURNS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "session_id": pl.Utf8,
    "turn_index": pl.Int64,
    "uuid": pl.Utf8,
    "parent_uuid": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "cwd": pl.Utf8,
    "git_branch": pl.Utf8,
    "model": pl.Utf8,
    "is_sidechain": pl.Boolean,
    "is_compact_summary": pl.Boolean,
    "input_tokens": pl.Int64,
    "output_tokens": pl.Int64,
    "cache_creation_tokens": pl.Int64,
    "cache_read_tokens": pl.Int64,
    "service_tier": pl.Utf8,
    "stop_reason": pl.Utf8,
    "text_output_chars": pl.Int64,
    "thinking_chars": pl.Int64,
    "tool_call_count": pl.Int64,
    "tool_names": pl.List(pl.Utf8),
    "tool_input_bytes": pl.Int64,
    "tool_result_bytes": pl.Int64,
    "wall_duration_ms": pl.Int64,
}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _json_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def _iter_events(path: Path) -> list[dict[str, Any]]:
    """Stream-load a session JSONL file. Malformed lines are logged and skipped."""
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("Malformed JSON on line %d of %s: %s", lineno, path, exc)
    return events


def _extract_tool_results(user_event: dict[str, Any]) -> dict[str, int]:
    """Return {tool_use_id: byte_length} for tool_result blocks in a user event."""
    out: dict[str, int] = {}
    message = user_event.get("message")
    if not isinstance(message, dict):
        return out
    content = message.get("content")
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        use_id = block.get("tool_use_id")
        if not isinstance(use_id, str):
            continue
        out[use_id] = _json_bytes(block.get("content"))
    return out


def _summarize_assistant_content(content: Any) -> dict[str, Any]:
    text_chars = 0
    thinking_chars = 0
    tool_names: list[str] = []
    tool_input_bytes = 0
    tool_use_ids: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_chars += len(block.get("text") or "")
            elif btype == "thinking":
                thinking_chars += len(block.get("thinking") or "")
            elif btype == "tool_use":
                name = block.get("name")
                if isinstance(name, str):
                    tool_names.append(name)
                tool_input_bytes += _json_bytes(block.get("input"))
                use_id = block.get("id")
                if isinstance(use_id, str):
                    tool_use_ids.append(use_id)
    return {
        "text_chars": text_chars,
        "thinking_chars": thinking_chars,
        "tool_names": tool_names,
        "tool_input_bytes": tool_input_bytes,
        "tool_use_ids": tool_use_ids,
    }


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def parse_session(path: str | Path) -> pl.DataFrame:
    """Parse one Claude Code session JSONL into a DataFrame matching ``turns``.

    One row per assistant event. ``tool_result_bytes`` is sourced from the
    next user event by matching ``tool_use_id``. ``wall_duration_ms`` is the
    delta between this assistant's timestamp and the timestamp of the next
    user event in the file (or null if none follows).
    """
    path = Path(path)
    events = _iter_events(path)

    rows: list[dict[str, Any]] = []
    turn_index = 0

    for i, event in enumerate(events):
        etype = event.get("type")
        is_compact_summary = bool(event.get("isCompactSummary", False))

        # Real Claude Code session JSONLs put the compact-summary marker on a
        # type=="user" event (with isVisibleInTranscriptOnly + isCompactSummary
        # at the top level). Surface it as a turn row so downstream aggregations
        # see the compaction. Assistant-only fields stay null.
        if etype == "user" and is_compact_summary:
            message = event.get("message") or {}
            if not isinstance(message, dict):
                message = {}
            content = message.get("content")
            text_chars = len(content) if isinstance(content, str) else 0

            ts = _parse_timestamp(event.get("timestamp"))
            rows.append(
                {
                    "session_id": event.get("sessionId"),
                    "turn_index": turn_index,
                    "uuid": event.get("uuid"),
                    "parent_uuid": event.get("parentUuid"),
                    "timestamp": ts,
                    "cwd": event.get("cwd"),
                    "git_branch": event.get("gitBranch"),
                    "model": None,
                    "is_sidechain": bool(event.get("isSidechain", False)),
                    "is_compact_summary": True,
                    "input_tokens": None,
                    "output_tokens": None,
                    "cache_creation_tokens": None,
                    "cache_read_tokens": None,
                    "service_tier": None,
                    "stop_reason": None,
                    "text_output_chars": text_chars,
                    "thinking_chars": 0,
                    "tool_call_count": 0,
                    "tool_names": [],
                    "tool_input_bytes": 0,
                    "tool_result_bytes": 0,
                    "wall_duration_ms": None,
                }
            )
            turn_index += 1
            continue

        if etype != "assistant":
            continue

        message = event.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        usage = message.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        summary = _summarize_assistant_content(message.get("content"))

        next_user: dict[str, Any] | None = None
        for forward in events[i + 1 :]:
            if forward.get("type") == "user":
                next_user = forward
                break

        tool_result_map: dict[str, int] = {}
        if next_user is not None:
            tool_result_map = _extract_tool_results(next_user)

        tool_result_bytes = sum(
            tool_result_map.get(use_id, 0) for use_id in summary["tool_use_ids"]
        )

        ts = _parse_timestamp(event.get("timestamp"))
        wall_duration_ms: int | None = None
        if ts is not None and next_user is not None:
            next_ts = _parse_timestamp(next_user.get("timestamp"))
            if next_ts is not None:
                wall_duration_ms = int((next_ts - ts).total_seconds() * 1000)

        service_tier = usage.get("service_tier")
        if not isinstance(service_tier, str):
            service_tier = None

        rows.append(
            {
                "session_id": event.get("sessionId"),
                "turn_index": turn_index,
                "uuid": event.get("uuid"),
                "parent_uuid": event.get("parentUuid"),
                "timestamp": ts,
                "cwd": event.get("cwd"),
                "git_branch": event.get("gitBranch"),
                "model": message.get("model"),
                "is_sidechain": bool(event.get("isSidechain", False)),
                "is_compact_summary": bool(event.get("isCompactSummary", False)),
                "input_tokens": _coerce_int(usage.get("input_tokens")),
                "output_tokens": _coerce_int(usage.get("output_tokens")),
                "cache_creation_tokens": _coerce_int(
                    usage.get("cache_creation_input_tokens")
                ),
                "cache_read_tokens": _coerce_int(usage.get("cache_read_input_tokens")),
                "service_tier": service_tier,
                "stop_reason": message.get("stop_reason"),
                "text_output_chars": summary["text_chars"],
                "thinking_chars": summary["thinking_chars"],
                "tool_call_count": len(summary["tool_names"]),
                "tool_names": summary["tool_names"],
                "tool_input_bytes": summary["tool_input_bytes"],
                "tool_result_bytes": tool_result_bytes,
                "wall_duration_ms": wall_duration_ms,
            }
        )
        turn_index += 1

    return pl.DataFrame(rows, schema=TURNS_SCHEMA, orient="row")
