"""Small, deterministic Matplotlib charts for the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#17324d"
BLUE = "#2563eb"
GREEN = "#16835f"
PURPLE = "#6d4bc3"
ORANGE = "#c47c09"
RED = "#b84a4a"
GRID = "#dce4ec"

MODEL_COLORS = {
    "Lagged Logistic Regression": "#2563eb",
    "Random Forest": "#16835f",
    "XGBoost": "#6d4bc3",
    "Compact LSTM": "#c47c09",
    "EMA8/EMA13 State": "#0e7490",
    "SMA10/SMA20 State": "#64748b",
    "Buy-and-Hold": "#111827",
}

SERIES_COLORS = {
    "Mean outer-fold Balanced Accuracy": BLUE,
    "Audit Balanced Accuracy": ORANGE,
}


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors="#526273", labelsize=9)
    ax.title.set_color(NAVY)
    ax.title.set_fontweight("semibold")


def grouped_percentage_bars(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str = "Percentage",
    reference: float | None = None,
) -> plt.Figure:
    """Create grouped bars from proportions and annotate exact percentages."""

    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(categories))
    width = 0.72 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (label, values) in zip(offsets, series.items(), strict=True):
        numeric = np.asarray(values, dtype=float) * 100
        bars = axis.bar(
            x + offset,
            numeric,
            width=width * 0.92,
            label=label,
            color=SERIES_COLORS.get(label, MODEL_COLORS.get(label, BLUE)),
            alpha=0.92,
        )
        axis.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3, color=NAVY)
    if reference is not None:
        axis.axhline(
            reference * 100,
            color="#94a3b8",
            linestyle="--",
            linewidth=1.1,
            label=f"{reference * 100:.0f}% reference",
        )
    axis.set_xticks(x, categories)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", pad=14)
    axis.legend(frameon=False, fontsize=9, ncols=2)
    axis.margins(y=0.18)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def horizontal_percentage_bars(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    reference: float | None = None,
) -> plt.Figure:
    ordered = frame.sort_values(value, ascending=True)
    numeric = ordered[value].astype(float).to_numpy() * 100
    figure, axis = plt.subplots(figsize=(9.4, 4.5))
    colors = [
        GREEN if "Technical" in str(label) else BLUE
        for label in ordered[category]
    ]
    bars = axis.barh(
        ordered[category],
        numeric,
        color=colors,
        alpha=0.9,
        height=0.62,
    )
    axis.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=4, color=NAVY)
    if reference is not None:
        axis.axvline(
            reference * 100,
            color="#94a3b8",
            linestyle="--",
            linewidth=1.1,
        )
    axis.set_title(title, loc="left", pad=14)
    axis.set_xlabel("Accuracy")
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.14)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def annotated_heatmap(
    matrix: pd.DataFrame,
    title: str,
    unit: str = "%",
    value_range: tuple[float, float] | None = None,
) -> plt.Figure:
    values = matrix.astype(float).to_numpy() * 100
    if value_range is not None:
        lower, upper = (float(value) * 100 for value in value_range)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("value_range must contain two finite increasing values.")
        cmap = "Blues"
    elif np.nanmin(values) < 0 < np.nanmax(values):
        max_abs = max(float(np.nanmax(np.abs(values))), 1.0)
        cmap = "RdYlGn"
        lower, upper = -max_abs, max_abs
    else:
        cmap = "Blues"
        lower, upper = min(0.0, float(np.nanmin(values))), float(np.nanmax(values))
    figure, axis = plt.subplots(
        figsize=(max(7.5, matrix.shape[1] * 2.15), max(3.2, matrix.shape[0] * 1.05))
    )
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=lower, vmax=upper)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column,
                row,
                f"{value:,.1f}{unit}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if luminance < 0.55 else NAVY,
                fontweight="semibold",
            )
    axis.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    axis.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    axis.set_title(title, loc="left", pad=14)
    axis.tick_params(axis="x", rotation=0)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.colorbar(image, ax=axis, shrink=0.72, pad=0.02)
    figure.tight_layout()
    return figure


def equity_curve_chart(frame: pd.DataFrame, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10.4, 5.0))
    for strategy, group in frame.groupby("Strategy", sort=False):
        group = group.sort_values("Open Date")
        axis.plot(
            group["Open Date"],
            group["Net NAV"],
            label=strategy,
            linewidth=2.0 if strategy in MODEL_COLORS else 1.5,
            color=MODEL_COLORS.get(strategy),
        )
    axis.axhline(1.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel("Net NAV (log scale)")
    axis.set_title(title, loc="left", pad=14)
    axis.legend(frameon=False, fontsize=8, ncols=2)
    _style_axis(axis)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
