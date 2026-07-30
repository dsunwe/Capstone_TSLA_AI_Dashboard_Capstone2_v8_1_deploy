from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def main_oof_uncertainty_chart(
    classification_uncertainty: pd.DataFrame,
    return_separation_uncertainty: pd.DataFrame,
):
    """Plot pointwise OOF accuracy and gross-return-separation intervals."""

    classification_required = {
        "Analysis",
        "Model",
        "Metric",
        "Point Estimate",
        "CI Lower",
        "CI Upper",
    }
    return_required = {
        "Model",
        "Point Estimate",
        "CI Lower",
        "CI Upper",
    }
    if (
        classification_uncertainty.empty
        or return_separation_uncertainty.empty
        or not classification_required.issubset(classification_uncertainty.columns)
        or not return_required.issubset(return_separation_uncertainty.columns)
    ):
        return None
    accuracy = classification_uncertainty.loc[
        classification_uncertainty["Analysis"].eq("Absolute OOF metric")
        & classification_uncertainty["Metric"].eq("Accuracy")
    ].copy()
    separation = return_separation_uncertainty.copy()
    if accuracy.empty or separation.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, table, title, reference, scale, x_label in [
        (
            axes[0],
            accuracy,
            "Pooled OOF Direction Accuracy",
            0.50,
            100.0,
            "Accuracy (%)",
        ),
        (
            axes[1],
            separation,
            "Long-minus-Cash Gross Return",
            0.0,
            100.0,
            "Mean daily return separation (%)",
        ),
    ]:
        table = table.reset_index(drop=True)
        point = pd.to_numeric(table["Point Estimate"], errors="coerce").to_numpy()
        lower = pd.to_numeric(table["CI Lower"], errors="coerce").to_numpy()
        upper = pd.to_numeric(table["CI Upper"], errors="coerce").to_numpy()
        y = np.arange(len(table))
        ax.errorbar(
            point * scale,
            y,
            xerr=np.vstack([(point - lower) * scale, (upper - point) * scale]),
            fmt="o",
            capsize=4,
            color="#2563eb",
            ecolor="#64748b",
        )
        ax.axvline(reference * scale, color="#dc2626", linestyle="--", linewidth=1.4)
        ax.set_yticks(y, table["Model"].astype(str))
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "TSLA Main-Study Pointwise 95% Moving-Block Intervals "
        "(20-Day Blocks, 2,000 Replicates)"
    )
    fig.tight_layout()
    return fig


def chronological_split_chart(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
):
    """Visualise the non-overlapping chronological holdout design."""
    periods = [
        ("Training", pd.Timestamp(train_start), pd.Timestamp(train_end), "#2563eb"),
        ("Held-out test", pd.Timestamp(test_start), pd.Timestamp(test_end), "#f59e0b"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for y, (label, start, end, colour) in enumerate(reversed(periods)):
        start_number = mdates.date2num(start)
        duration = max(mdates.date2num(end) - start_number, 1)
        ax.barh(y, duration, left=start_number, height=0.48, color=colour, label=label)
        ax.text(
            start_number + duration / 2,
            y,
            f"{start:%Y-%m} to {end:%Y-%m}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_yticks([0, 1], ["Held-out test", "Training"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Chronological Train/Test Split (No Random Shuffle)")
    ax.set_xlabel("Calendar time")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def walk_forward_metric_chart(walk_forward: pd.DataFrame):
    """Compare OOF accuracy by fold, supporting one or several model levels."""
    required = {"Fold", "Accuracy", "Majority Baseline Accuracy"}
    if walk_forward.empty or not required.issubset(walk_forward.columns):
        return None
    if "Model" in walk_forward.columns:
        chart = walk_forward[["Model", "Fold", "Accuracy"]].copy()
        fig, ax = plt.subplots(figsize=(10.5, 5))
        sns.lineplot(
            data=chart,
            x="Fold",
            y="Accuracy",
            hue="Model",
            marker="o",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Walk-Forward OOF Accuracy by Model and Fold")
        ax.legend(
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            borderaxespad=0,
        )
        fig.tight_layout()
        return fig
    chart = walk_forward[["Fold", "Accuracy", "Majority Baseline Accuracy"]].melt(
        id_vars="Fold",
        var_name="Series",
        value_name="Score",
    )
    chart["Series"] = chart["Series"].replace(
        {
            "Accuracy": "Selected model",
            "Majority Baseline Accuracy": "Majority baseline",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(
        data=chart,
        x="Fold",
        y="Score",
        hue="Series",
        marker="o",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-Forward Accuracy by Fold")
    ax.legend(title="")
    fig.tight_layout()
    return fig


def line_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y_columns: list[str],
    title: str,
    y_label: str = "Value",
):
    available = [column for column in y_columns if column in df.columns]
    if not available:
        return None
    plot_data = df[[x] + available].dropna(subset=[x]).melt(
        id_vars=x,
        value_vars=available,
        var_name="Feature",
        value_name="Value",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=plot_data, x=x, y="Value", hue="Feature", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(title="Feature", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def price_moving_average_chart(
    df: pd.DataFrame,
    *,
    date_column: str = "Date",
    price_column: str = "Price",
    short_window: int = 10,
    long_window: int = 20,
    title: str | None = None,
):
    """Plot price with short- and long-window simple moving averages.

    Existing ``SMA{window}`` columns are used when available; otherwise the
    helper computes them on a copy, leaving the caller's DataFrame unchanged.
    """
    if (
        df.empty
        or date_column not in df.columns
        or price_column not in df.columns
        or short_window < 1
        or long_window < 1
    ):
        return None

    short_column = f"SMA{short_window}"
    long_column = f"SMA{long_window}"
    columns = [date_column, price_column]
    columns.extend(
        column
        for column in [short_column, long_column]
        if column in df.columns and column not in columns
    )
    chart = df[columns].copy()
    chart[date_column] = pd.to_datetime(chart[date_column], errors="coerce")
    chart[price_column] = pd.to_numeric(chart[price_column], errors="coerce")
    chart = chart.dropna(subset=[date_column, price_column]).sort_values(
        date_column,
        kind="stable",
    )
    if chart.empty:
        return None
    for window, column in [
        (short_window, short_column),
        (long_window, long_column),
    ]:
        if column not in chart.columns:
            chart[column] = chart[price_column].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        chart[date_column],
        chart[price_column],
        label="Price",
        color="#111827",
        linewidth=1.5,
    )
    ax.plot(
        chart[date_column],
        chart[short_column],
        label=f"MA{short_window}",
        color="#2563eb",
        linewidth=1.25,
    )
    ax.plot(
        chart[date_column],
        chart[long_column],
        label=f"MA{long_window}",
        color="#f59e0b",
        linewidth=1.25,
    )
    ax.set_title(title or f"Price with MA{short_window} and MA{long_window}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(title="")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def return_distribution(df: pd.DataFrame, column: str = "Daily_Return"):
    if column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[column].dropna(), bins=60, kde=True, ax=ax)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str]):
    available = [column for column in columns if column in df.columns]
    if len(available) < 2:
        return None
    corr = df[available].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Selected Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def confusion_matrix_heatmap(confusion: pd.DataFrame):
    if confusion.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 13):
    if feature_importance.empty:
        return None
    value_column = "Importance" if "Importance" in feature_importance.columns else "Absolute Coefficient"
    chart = feature_importance.head(top_n).sort_values(value_column, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=chart, x=value_column, y="Feature", ax=ax)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def model_line_chart(equity_df: pd.DataFrame, y_column: str, title: str):
    if (
        equity_df.empty
        or "Date" not in equity_df.columns
        or y_column not in equity_df.columns
    ):
        return None
    chart = equity_df.copy()
    chart["Date"] = pd.to_datetime(chart["Date"], errors="coerce")
    chart = chart.dropna(subset=["Date", y_column])
    if chart.empty:
        return None
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    sns.lineplot(data=chart, x="Date", y=y_column, hue="Model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_column)
    ax.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        borderaxespad=0,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def sensitivity_bar_chart(df: pd.DataFrame, y_column: str, title: str):
    if df.empty or y_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Model", y=y_column, hue="Type", dodge=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def cross_asset_model_chart(
    benchmark: pd.DataFrame,
    metric: str = "Macro F1",
):
    if benchmark.empty or metric not in benchmark.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=benchmark, x="Asset", y=metric, hue="Model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-Asset {metric} by Model")
    ax.legend(title="Model", loc="best")
    fig.tight_layout()
    return fig


def feature_stability_chart(feature_stability: pd.DataFrame, top_n: int = 15):
    if feature_stability.empty or "Selection Rate" not in feature_stability.columns:
        return None
    chart = feature_stability.head(top_n).sort_values("Selection Rate", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=chart, x="Selection Rate", y="Feature", hue="Feature Group", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Training-Selected Feature Stability Across Assets")
    ax.legend(
        title="Feature Group",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    fig.tight_layout()
    return fig


def ablation_metric_chart(ablation: pd.DataFrame, metric: str = "Macro F1"):
    if ablation.empty or metric not in ablation.columns:
        return None
    chart = ablation.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=chart, x=metric, y="Experiment", ax=ax, color="#2563eb")
    ax.set_xlim(0, 1)
    ax.set_title(f"Feature Ablation: Held-Out {metric}")
    fig.tight_layout()
    return fig


def normalised_asset_price_chart(engineered_by_asset: dict[str, pd.DataFrame]):
    rows: list[pd.DataFrame] = []
    for asset, frame in engineered_by_asset.items():
        values = frame[["Date", "Price"]].dropna().copy()
        if values.empty:
            continue
        values["Normalised Price"] = values["Price"] / values["Price"].iloc[0]
        values["Asset"] = asset
        rows.append(values[["Date", "Normalised Price", "Asset"]])
    if not rows:
        return None
    chart = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=chart, x="Date", y="Normalised Price", hue="Asset", ax=ax)
    ax.set_title("Common-Window Normalised Asset Prices")
    ax.set_ylabel("Growth of 1.0")
    fig.tight_layout()
    return fig
