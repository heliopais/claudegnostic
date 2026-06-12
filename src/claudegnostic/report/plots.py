"""Chart builders for the HTML report.

Each public function takes a DataFrame and returns a base64-encoded PNG data
URI ready to drop into an ``<img src="...">``, or ``None`` when the input
has too little data to plot. Callers render a "not enough data" note for
``None``.

The matplotlib ``Agg`` backend is forced at import time so plotnine can render
without a display. Any other module that imports matplotlib MUST be loaded
after this one (or set Agg itself first).
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede any plotnine/pyplot import

import polars as pl  # noqa: E402
from plotnine import (  # noqa: E402
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_bar,
    geom_errorbar,
    geom_histogram,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    position_dodge,
    scale_color_gradient,
    scale_fill_gradient,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

_DPI = 100
_WIDTH_IN = 9.6
_HEIGHT_IN = 4.8


def _to_data_uri(plot: ggplot, *, height: float = _HEIGHT_IN) -> str:
    """Render a plotnine plot to a base64 PNG data URI."""
    buf = io.BytesIO()
    plot.save(
        buf,
        format="png",
        dpi=_DPI,
        width=_WIDTH_IN,
        height=height,
        units="in",
        verbose=False,
    )
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def tokens_by_model_chart(df: pl.DataFrame) -> str | None:
    """Stacked bar of tokens by model and category."""
    if df.is_empty():
        return None
    long = df.unpivot(
        index="model",
        on=["input", "output", "cache_creation", "cache_read"],
        variable_name="category",
        value_name="tokens",
    ).filter(pl.col("tokens") > 0)
    if long.is_empty():
        return None
    plot = (
        ggplot(long, aes(x="model", y="tokens", fill="category"))
        + geom_bar(stat="identity")
        + coord_flip()
        + labs(
            title="Tokens by model",
            x="",
            y="Tokens",
            fill="Category",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)


def wall_time_per_tool_chart(df: pl.DataFrame, top_n: int = 12) -> str | None:
    """Bar of p50 wall time per tool with an error bar reaching up to p90.

    We do not store raw observations per tool, so a true box plot is out of
    reach; this conveys the same p50/p90 summary that the analysis layer
    exposes. Log-scale y axis keeps a 10x range legible.
    """
    if df.is_empty():
        return None
    sub = df.head(top_n).filter(pl.col("p50_ms") > 0)
    if sub.is_empty():
        return None
    plot = (
        ggplot(sub, aes(x="tool", y="p50_ms"))
        + geom_bar(stat="identity", fill="#4C78A8")
        + geom_errorbar(
            aes(ymin="p50_ms", ymax="p90_ms"),
            width=0.3,
            color="#1f2933",
            position=position_dodge(0.9),
        )
        + scale_y_log10()
        + coord_flip()
        + labs(
            title="Wall time per tool (bar = p50, whisker = p90, log scale)",
            x="",
            y="Milliseconds",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)


def cache_hit_ratio_chart(df: pl.DataFrame) -> str | None:
    """Histogram of per-session cache hit ratios."""
    if df.is_empty():
        return None
    plot = (
        ggplot(df, aes(x="ratio"))
        + geom_histogram(bins=20, fill="#4C78A8", color="white")
        + labs(
            title="Cache hit ratio across sessions",
            x="cache_read / (cache_read + input)",
            y="Sessions",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)


def tool_co_occurrence_chart(df: pl.DataFrame, top_n_tools: int = 12) -> str | None:
    """Heatmap of unordered tool-pair counts, restricted to the top-N tools."""
    if df.is_empty():
        return None
    counts = (
        pl.concat(
            [
                df.select(pl.col("tool_a").alias("tool"), pl.col("pair_count")),
                df.select(pl.col("tool_b").alias("tool"), pl.col("pair_count")),
            ]
        )
        .group_by("tool")
        .agg(pl.col("pair_count").sum().alias("total"))
        .sort("total", descending=True)
        .head(top_n_tools)
    )
    if counts.is_empty():
        return None
    keep = set(counts["tool"].to_list())
    sub = df.filter(pl.col("tool_a").is_in(keep) & pl.col("tool_b").is_in(keep))
    if sub.is_empty():
        return None
    plot = (
        ggplot(sub, aes(x="tool_a", y="tool_b", fill="pair_count"))
        + geom_tile()
        + scale_fill_gradient(low="#EEEEEE", high="#22577A")
        + labs(
            title=f"Tool co-occurrence (top {top_n_tools} tools)",
            x="",
            y="",
            fill="Pair count",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
    return _to_data_uri(plot, height=5.4)


def session_length_chart(df: pl.DataFrame) -> str | None:
    """Bar chart of session-count by turn-count bucket."""
    if df.is_empty() or df["count"].sum() == 0:
        return None
    plot = (
        ggplot(df, aes(x="bucket", y="count"))
        + geom_bar(stat="identity", fill="#22577A")
        + labs(
            title="Session length distribution",
            x="Turns per session",
            y="Sessions",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)


def cost_vs_context_per_turn_chart(df: pl.DataFrame) -> str | None:
    """Scatter of per-turn estimated USD vs context window size, faceted by model.

    One dot per turn, one panel per model. ``context_tokens`` (input +
    cache_read + cache_creation) and per-turn cost both span multiple
    orders of magnitude, so both axes are log-scaled. Panels share axes so
    cross-model comparison stays apples-to-apples.
    """
    if df.is_empty():
        return None
    sub = df.filter((pl.col("context_tokens") > 0) & (pl.col("est_usd") > 0)).with_columns(
        pl.col("model").fill_null("<unknown>").alias("model"),
    )
    if sub.is_empty():
        return None
    n_models = sub["model"].n_unique()
    # ~2.4in per row of panels; plotnine picks the wrap width.
    ncol = min(n_models, 2)
    nrow = (n_models + ncol - 1) // ncol
    height = max(_HEIGHT_IN, 2.4 * nrow + 1.2)
    has_share = "cache_read_share" in sub.columns
    point_layer = (
        geom_point(aes(color="cache_read_share"), alpha=0.5, size=1.5)
        if has_share
        else geom_point(color="#22577A", alpha=0.4, size=1.5)
    )
    plot = (
        ggplot(sub, aes(x="context_tokens", y="est_usd"))
        + point_layer
        + facet_wrap("model", ncol=ncol)
        + scale_x_log10()
        + scale_y_log10()
        + labs(
            title="Estimated cost vs context window per turn (log-log, by model)",
            x="Context tokens (input + cache_read + cache_creation)",
            y="Estimated USD",
            color="Cache-read share" if has_share else None,
        )
        + theme_minimal()
    )
    if has_share:
        # Low share (cold / fresh writes) = warm color; high share (warm) = cool color.
        plot = plot + scale_color_gradient(low="#D62728", high="#1F77B4", limits=(0.0, 1.0))
    return _to_data_uri(plot, height=height)


def cost_vs_turns_chart(df: pl.DataFrame) -> str | None:
    """Scatter of per-session estimated USD vs turn count.

    Log-scale both axes — turn counts and per-session cost both span
    multiple orders of magnitude in real usage.
    """
    if df.is_empty():
        return None
    sub = df.filter((pl.col("turn_count") > 0) & (pl.col("est_usd") > 0))
    if sub.is_empty():
        return None
    plot = (
        ggplot(sub, aes(x="turn_count", y="est_usd"))
        + geom_point(color="#4C78A8", alpha=0.6, size=2)
        + scale_x_log10()
        + scale_y_log10()
        + labs(
            title="Estimated cost vs turns per session (log-log)",
            x="Turns per session",
            y="Estimated USD",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)
