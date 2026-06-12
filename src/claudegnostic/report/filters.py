"""Duration parsing and project filtering helpers for the report surface."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import polars as pl

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 7 * 86_400}


def parse_duration(text: str) -> timedelta:
    """Parse a short duration string like ``30d``, ``12h``, ``2w``.

    Accepts an integer count followed by a unit letter (``s``, ``m``, ``h``,
    ``d``, ``w``). Raises ``ValueError`` on anything else.
    """
    match = _DURATION_RE.match(text)
    if not match:
        raise ValueError(
            f"Invalid duration {text!r}: expected like '30d', '12h', '2w'."
        )
    count = int(match.group(1))
    unit = match.group(2).lower()
    return timedelta(seconds=count * _UNIT_SECONDS[unit])


def since_from(now: datetime, duration: str | None) -> datetime | None:
    """Return the lower-bound timestamp for ``--since``, or ``None`` when unbounded."""
    if duration is None:
        return None
    return now - parse_duration(duration)


def filter_by_cwd(df: pl.DataFrame, substring: str | None) -> pl.DataFrame:
    """Return rows whose ``cwd`` contains ``substring`` (case-insensitive).

    A null ``substring`` returns the DataFrame unchanged. A DataFrame without
    a ``cwd`` column also returns unchanged (callers may pre-aggregate).
    """
    if substring is None or "cwd" not in df.columns:
        return df
    return df.filter(pl.col("cwd").str.contains(substring, literal=True))
