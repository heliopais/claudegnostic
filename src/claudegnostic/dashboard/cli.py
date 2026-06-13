"""Launcher that hands off to ``streamlit run`` for the dashboard app."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DB_PATH_ENV = "CLAUDEGNOSTIC_DB_PATH"
APP_FILE = Path(__file__).with_name("app.py")


def launch(db_path: Path, port: int) -> int:
    """Spawn ``streamlit run`` for the dashboard app and return its exit code.

    The DB path is passed through an environment variable because Streamlit
    swallows positional CLI args before the app sees them.
    """
    env = os.environ.copy()
    env[DB_PATH_ENV] = str(db_path)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(cmd, env=env)
