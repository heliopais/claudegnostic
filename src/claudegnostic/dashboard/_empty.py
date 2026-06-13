"""Uniform empty-state helper for dashboard pages."""

from __future__ import annotations

import streamlit as st

DEFAULT_MESSAGE = (
    "Not enough data — run `claudegnostic ingest` to populate the database first."
)


def empty_state(message: str | None = None) -> None:
    """Render the canonical empty-state notice.

    Pages call this instead of inlining ``st.info`` so the wording stays
    consistent and one place to change it later.
    """
    st.info(message or DEFAULT_MESSAGE)
