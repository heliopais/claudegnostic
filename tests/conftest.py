"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Typer/Click CliRunner for invoking the CLI in-process."""
    return CliRunner()
