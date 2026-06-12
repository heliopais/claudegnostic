"""Reuse the seeded-DB fixture from the analysis test tree."""

from __future__ import annotations

# Re-export the seeded/empty in-memory DuckDB fixtures so report tests can
# build on them without duplicating the corpus.
from tests.analysis.conftest import empty_db, seeded_db  # noqa: F401
