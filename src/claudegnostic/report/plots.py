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
    geom_bar,
    geom_errorbar,
    geom_histogram,
    geom_tile,
    ggplot,
    labs,
    position_dodge,
    scale_fill_gradient,
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
        ggplot(long.to_pandas(), aes(x="model", y="tokens", fill="category"))
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
        ggplot(sub.to_pandas(), aes(x="tool", y="p50_ms"))
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
        ggplot(df.to_pandas(), aes(x="ratio"))
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
        ggplot(sub.to_pandas(), aes(x="tool_a", y="tool_b", fill="pair_count"))
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
        ggplot(df.to_pandas(), aes(x="bucket", y="count"))
        + geom_bar(stat="identity", fill="#22577A")
        + labs(
            title="Session length distribution",
            x="Turns per session",
            y="Sessions",
        )
        + theme_minimal()
    )
    return _to_data_uri(plot)
