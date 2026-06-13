"""Sidebar filter widgets.

Renders the sidebar and returns a ``FilterState`` snapshot. Widgets are
keyed off ``st.session_state`` so values round-trip across Streamlit reruns.
The page beads will consume the returned state; analysis functions do not
read it yet.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import streamlit as st

from claudegnostic.dashboard._db import FilterOptions
from claudegnostic.dashboard._state import FilterState, default_state


def render_sidebar(options: FilterOptions) -> FilterState:
    """Draw the filter sidebar and return the current selection."""
    st.sidebar.header("Filters")

    if options["date_min"] is None or options["date_max"] is None:
        st.sidebar.caption("No date range available — database is empty.")
        return default_state()

    raw_range = st.sidebar.date_input(
        "Date range",
        value=(options["date_min"], options["date_max"]),
        min_value=options["date_min"],
        max_value=options["date_max"],
        key="filter_date_range",
    )
    # Streamlit returns a tuple for ranges; a single date if the user is
    # mid-selection. Normalize to a (start, end) pair.
    if isinstance(raw_range, tuple) and len(raw_range) == 2:
        start, end = raw_range
    else:
        single = cast(date, raw_range)
        start, end = single, single

    cwd_substr = st.sidebar.text_input(
        "Project (cwd substring)",
        value="",
        key="filter_cwd_substr",
        placeholder="e.g. claudegnostic",
    )

    models = st.sidebar.multiselect(
        "Models",
        options=options["models"],
        default=[],
        key="filter_models",
    )

    return FilterState(
        start_date=start,
        end_date=end,
        cwd_substr=cwd_substr,
        models=list(models),
    )
