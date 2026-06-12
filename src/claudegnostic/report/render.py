"""Orchestrate the report: pull DataFrames, render plots, fill the template, write."""

from __future__ import annotations

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from claudegnostic import __version__
from claudegnostic.analysis._connect import ConnLike, as_connection
from claudegnostic.analysis.archaeology import (
    session_length_distribution,
    tool_co_occurrence,
)
from claudegnostic.analysis.cost import (
    cost_vs_context_by_turn,
    cost_vs_turns_by_session,
    estimated_cost_by_session,
    tokens_by_model,
)
from claudegnostic.analysis.productivity import (
    cache_hit_ratio_by_session,
    wall_time_per_tool,
)
from claudegnostic.report.filters import filter_by_cwd, since_from


def _templates_dir() -> Path:
    """Return the on-disk path to the packaged templates directory."""
    return Path(str(files("claudegnostic.report").joinpath("templates")))


def _read_totals(
    con_or_path: ConnLike,
    since: datetime | None,
    project: str | None,
) -> dict[str, int]:
    sql = """
        SELECT COUNT(DISTINCT session_id)::BIGINT AS sessions,
               COUNT(*)::BIGINT                   AS turns
        FROM turns
        WHERE (? IS NULL OR timestamp >= ?)
          AND (? IS NULL OR cwd LIKE ?)
    """
    like = f"%{project}%" if project else None
    with as_connection(con_or_path) as conn:
        row = conn.execute(sql, [since, since, project, like]).fetchone()
    if row is None:
        return {"sessions": 0, "turns": 0}
    return {"sessions": int(row[0] or 0), "turns": int(row[1] or 0)}


def _top_cost_sessions(df: pl.DataFrame, top_n: int) -> list[dict[str, object]]:
    if df.is_empty():
        return []
    return df.head(top_n).to_dicts()


def render_report(
    db_path: Path,
    *,
    out_path: Path,
    since: str | None = None,
    project: str | None = None,
    now: datetime | None = None,
    top_n: int = 10,
) -> Path:
    """Render the HTML report and write it to ``out_path``.

    Args:
        db_path: Path to the claudegnostic DuckDB file.
        out_path: Destination ``.html`` file. Overwritten if it exists.
        since: Optional duration string (e.g. ``30d``, ``12h``) for the
            lower-bound timestamp.
        project: Optional case-sensitive substring matched against ``cwd``.
        now: Override "now" for ``--since`` arithmetic (test seam).
        top_n: Row count for top-N tables.

    Returns:
        The ``out_path`` that was written.
    """
    from claudegnostic.report import plots  # local import to keep Agg-setup contained

    actual_now = now if now is not None else datetime.now().astimezone()
    since_dt = since_from(actual_now, since)

    totals = _read_totals(db_path, since_dt, project)

    # Pull DataFrames once, then filter in-Python for project/since where the
    # analysis API doesn't accept them. Single read pass per query.
    with as_connection(db_path) as conn:
        tokens_df = tokens_by_model(conn, since=since_dt)
        cost_df = estimated_cost_by_session(conn)
        cost_vs_turns_df = cost_vs_turns_by_session(conn)
        cost_vs_context_df = cost_vs_context_by_turn(conn)
        wall_df = wall_time_per_tool(conn)
        cache_df = cache_hit_ratio_by_session(conn)
        cooc_df = tool_co_occurrence(conn)
        length_df = session_length_distribution(conn)

    cost_df = filter_by_cwd(cost_df, project)
    cost_vs_turns_df = filter_by_cwd(cost_vs_turns_df, project)
    cost_vs_context_df = filter_by_cwd(cost_vs_context_df, project)

    charts = {
        "tokens_by_model": plots.tokens_by_model_chart(tokens_df),
        "cost_vs_turns": plots.cost_vs_turns_chart(cost_vs_turns_df),
        "cost_vs_context_per_turn": plots.cost_vs_context_per_turn_chart(cost_vs_context_df),
        "wall_time_per_tool": plots.wall_time_per_tool_chart(wall_df),
        "cache_hit_ratio": plots.cache_hit_ratio_chart(cache_df),
        "tool_co_occurrence": plots.tool_co_occurrence_chart(cooc_df),
        "session_length": plots.session_length_chart(length_df),
    }
    tables = {
        "top_cost_sessions": _top_cost_sessions(cost_df, top_n),
    }

    templates = _templates_dir()
    env = Environment(
        loader=FileSystemLoader(str(templates)),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        version=__version__,
        generated_at_human=actual_now.strftime("%Y-%m-%d %H:%M %Z").strip(),
        db_path=str(db_path),
        totals=totals,
        since_label=since,
        project_label=project,
        charts=charts,
        tables=tables,
        styles=(templates / "styles.css").read_text(encoding="utf-8"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def default_out_path(now: datetime | None = None) -> Path:
    """Default output path: ``./claudegnostic-report-YYYY-MM-DD.html``."""
    actual = now if now is not None else datetime.now().astimezone()
    return Path.cwd() / f"claudegnostic-report-{actual:%Y-%m-%d}.html"
