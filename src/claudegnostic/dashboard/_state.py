"""Typed filter state shared by the sidebar and page bodies."""

from __future__ import annotations

from datetime import date
from typing import TypedDict


class FilterState(TypedDict):
    """The sidebar's snapshot of what the user wants to look at.

    Page beads consume this dict; analysis functions don't accept it yet.
    A field at its default value means 'no filter'.
    """

    start_date: date | None
    end_date: date | None
    cwd_substr: str
    models: list[str]


def default_state() -> FilterState:
    """A FilterState that applies no filtering."""
    return FilterState(
        start_date=None,
        end_date=None,
        cwd_substr="",
        models=[],
    )
