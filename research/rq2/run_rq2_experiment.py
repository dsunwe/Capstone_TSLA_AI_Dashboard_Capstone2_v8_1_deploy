"""Run the focused v8.2 RQ2 EMA-aware feature-family experiment.

This script intentionally evaluates only RQ2. Every standalone feature family
uses the same regularised Logistic Regression, target, dates, chronological
folds, purge, threshold, and metrics. RQ3 model comparisons and RQ4 economic
backtests are not run here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "TSLA_features_common_window.csv"
CPI_DATA = ROOT / "data" / "CPIAUCSL.csv"
UNRATE_DATA = ROOT / "data" / "UNRATE.csv"
PROTOCOL = ROOT / "protocol.json"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
OUTPUTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

AUDIT_START = pd.Timestamp("2024-11-26")
WARMUP_START = pd.Timestamp("2016-10-03")
HORIZON = 5
PURGE_ROWS = 5
N_SPLITS = 5
TEST_SIZE = 250
BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 5_000
RANDOM_STATE = 42

CORE_5 = [
    "Daily_Return",
    "Momentum_5",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]

EMA_TECHNICAL = [
    "Daily_Return",
    "Momentum_5",
    "EMA8_13_State",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]

SMA_TECHNICAL = [
    "Daily_Return",
    "Momentum_5",
    "SMA10_20_State",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]

LEGACY_TECHNICAL = [
    "Daily_Return",
    "Momentum_5",
    "SMA50_Ratio",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]

MARKET_FEATURES = [
    "SP500_Return",
    "QQQ_Excess_Return",
    "QQQ_Momentum_5",
    "QQQ_Volatility_20",
    "QQQ_Volatility_Change_5",
]

MACRO_FEATURES = [
    "Treasury_10Y_Change_5",
    "Yield_Curve_10Y2Y",
    "Yield_Curve_Change_5",
    "Fed_Funds_Change_20",
    "Latest_CPI_YoY_Release_Change",
    "Latest_Unemployment_Release_Change",
]

SENTIMENT_FEATURES = [
    "VIX_Level",
    "VIX_Change_5",
    "Consumer_Sentiment",
    "Consumer_Sentiment_Release_Change",
]

FEATURE_FAMILIES = {
    "Technical — EMA-aware compact": EMA_TECHNICAL,
    "Cross-market": MARKET_FEATURES,
    "Macroeconomic": MACRO_FEATURES,
    "Broad sentiment and risk proxies": SENTIMENT_FEATURES,
}

TECHNICAL_DESIGNS = {
    "Primary EMA-aware compact": EMA_TECHNICAL,
    "Core-5 without moving-average state": CORE_5,
    "SMA10/20-state sensitivity": SMA_TECHNICAL,
    "Legacy SMA50-ratio comparator": LEGACY_TECHNICAL,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monthly_release_changes(
    path: Path,
    series_id: str,
) -> pd.DataFrame:
    """Compute a monthly release change before converting it to daily as-of data."""
    source = pd.read_csv(path, parse_dates=["Date"])
    source[series_id] = pd.to_numeric(source[series_id], errors="coerce")
    source = (
        source.dropna(subset=["Date", series_id])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if series_id == "CPIAUCSL":
        source["CPI_YoY"] = source[series_id].pct_change(12) * 100
        source["Latest_CPI_YoY_Release_Change"] = source["CPI_YoY"].diff()
        source["Available_Date"] = (
            source["Date"]
            + pd.offsets.MonthBegin(1)
            + pd.Timedelta(days=14)
        )
        value_column = "Latest_CPI_YoY_Release_Change"
    elif series_id == "UNRATE":
        source["Latest_Unemployment_Release_Change"] = source[series_id].diff()
        source["Available_Date"] = (
            source["Date"]
            + pd.offsets.MonthBegin(1)
            + pd.Timedelta(days=7)
        )
        value_column = "Latest_Unemployment_Release_Change"
    else:  # pragma: no cover - closed protocol
        raise ValueError(f"Unsupported monthly series: {series_id}")
    return source[["Available_Date", value_column]].dropna(
        subset=[value_column]
    )


def add_release_aware_changes(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    qqq_growth = 1 + data["QQQ_Return"].fillna(0)
    data["QQQ_Excess_Return"] = data["QQQ_Return"] - data["SP500_Return"]
    data["QQQ_Momentum_5"] = (
        qqq_growth.rolling(5).apply(np.prod, raw=True) - 1
    )
    data["QQQ_Volatility_Change_5"] = (
        data["QQQ_Volatility_20"] - data["QQQ_Volatility_20"].shift(5)
    )
    data["Treasury_10Y_Change_5"] = (
        data["Treasury_10Y_Yield"] - data["Treasury_10Y_Yield"].shift(5)
    )
    data["Yield_Curve_Change_5"] = (
        data["Yield_Curve_10Y2Y"] - data["Yield_Curve_10Y2Y"].shift(5)
    )
    data["Fed_Funds_Change_20"] = (
        data["Fed_Funds_Rate"] - data["Fed_Funds_Rate"].shift(20)
    )

    for source in [
        monthly_release_changes(CPI_DATA, "CPIAUCSL"),
        monthly_release_changes(UNRATE_DATA, "UNRATE"),
    ]:
        data = pd.merge_asof(
            data.sort_values("Date"),
            source.sort_values("Available_Date"),
            left_on="Date",
            right_on="Available_Date",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["Available_Date"])
    data["Consumer_Sentiment_Release_Change"] = data[
        "Consumer_Sentiment_Change"
    ]
    return data


def add_moving_average_states(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    ema_available = data[["EMA8", "EMA13"]].notna().all(axis=1)
    sma_available = data[["SMA10", "SMA20"]].notna().all(axis=1)
    data["EMA8_13_State"] = np.where(
        ema_available,
        (data["EMA8"] > data["EMA13"]).astype(int),
        np.nan,
    )
    data["SMA10_20_State"] = np.where(
        sma_available,
        (data["SMA10"] > data["SMA20"]).astype(int),
        np.nan,
    )
    return data


def add_weekly_target(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("Date").reset_index(drop=True)
    current_week_mean = data["Close"].rolling(HORIZON).mean()
    future_week_mean = sum(
        data["Close"].shift(-offset)
        for offset in range(1, HORIZON + 1)
    ) / HORIZON
    previous_week_mean = current_week_mean.shift(HORIZON)

    data["Current_Week_Mean_Close"] = current_week_mean
    data["Future_Week_Mean_Close"] = future_week_mean
    data["Weekly_Trend_Return"] = future_week_mean / current_week_mean - 1
    data["Weekly_Trend_Direction"] = np.where(
        data["Weekly_Trend_Return"].notna(),
        (data["Weekly_Trend_Return"] > 0).astype(int),
        np.nan,
    )
    data["Weekly_Persistence"] = np.where(
        previous_week_mean.notna(),
        (current_week_mean > previous_week_mean).astype(int),
        np.nan,
    )
    data["Target_Date"] = data["Date"].shift(-HORIZON)
    return data


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA, parse_dates=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data = add_release_aware_changes(data)
    data = add_moving_average_states(data)
    data = add_weekly_target(data)

    required = [
        "Date",
        "Target_Date",
        "Weekly_Trend_Direction",
        "Weekly_Persistence",
        *EMA_TECHNICAL,
        "SMA10_20_State",
        "SMA50_Ratio",
    ]
    data = data.dropna(subset=required)
    data = data.loc[data["Date"] >= WARMUP_START].reset_index(drop=True)
    data["Eligible_Row_ID"] = np.arange(len(data), dtype=int)
    data["Phase_Offset"] = data["Eligible_Row_ID"] % HORIZON
    return data


def development_and_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = frame.loc[frame["Target_Date"] < AUDIT_START].copy()
    audit = frame.loc[frame["Date"] >= AUDIT_START].copy()
    if development.empty or audit.empty:
        raise AssertionError("Development or audit split is empty.")
    if development["Target_Date"].max() >= audit["Date"].min():
        raise AssertionError("Development labels overlap the audit feature period.")
    return development, audit


def logistic_regression() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def classification_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(actual, prediction),
        "Balanced Accuracy": balanced_accuracy_score(actual, prediction),
        "Macro F1": f1_score(actual, prediction, average="macro"),
        "MCC": matthews_corrcoef(actual, prediction),
        "Recall Down": recall_score(actual, prediction, pos_label=0),
        "Recall Up": recall_score(actual, prediction, pos_label=1),
        "ROC-AUC": (
            roc_auc_score(actual, probability)
            if len(np.unique(actual)) == 2
            else np.nan
        ),
        "Predicted Positive Share": float(np.mean(prediction)),
    }


def walk_forward_oof(
    development: pd.DataFrame,
    features: list[str],
    design: str,
    design_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=PURGE_ROWS,
    )
    x = development[features]
    y = development["Weekly_Trend_Direction"].astype(int)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        train = development.iloc[train_index]
        test = development.iloc[test_index]
        y_train = y.iloc[train_index]
        actual = y.iloc[test_index].to_numpy()
        model = logistic_regression()
        model.fit(train[features], y_train)
        probability = model.predict_proba(test[features])[:, 1]
        prediction = (probability >= 0.5).astype(int)

        train_target_max = train["Target_Date"].max()
        test_feature_min = test["Date"].min()
        if train_target_max >= test_feature_min:
            raise AssertionError(
                f"Fold {fold} has label overlap: "
                f"{train_target_max} >= {test_feature_min}"
            )

        row: dict[str, object] = {
            design_column: design,
            "Fold": fold,
            "Train Start": train["Date"].min(),
            "Train End": train["Date"].max(),
            "Train Target End": train_target_max,
            "Test Start": test_feature_min,
            "Test End": test["Date"].max(),
            "Train Rows": len(train),
            "Test Rows": len(test),
            "Feature Count": len(features),
            "Features": " | ".join(features),
            "Actual Positive Share": float(actual.mean()),
        }
        row.update(classification_metrics(actual, prediction, probability))
        fold_rows.append(row)

        output = test[["Date", "Target_Date"]].copy()
        output[design_column] = design
        output["Fold"] = fold
        output["Actual"] = actual
        output["Probability"] = probability
        output["Prediction"] = prediction
        prediction_frames.append(output)

    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def fit_audit(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    features: list[str],
    design: str,
    design_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_train = development["Weekly_Trend_Direction"].astype(int)
    actual = audit["Weekly_Trend_Direction"].astype(int).to_numpy()
    model = logistic_regression()
    model.fit(development[features], y_train)
    probability = model.predict_proba(audit[features])[:, 1]
    prediction = (probability >= 0.5).astype(int)

    row: dict[str, object] = {
        design_column: design,
        "Model": "Regularised Logistic Regression",
        "Audit Start": audit["Date"].min(),
        "Audit End": audit["Date"].max(),
        "Development Rows": len(development),
        "Audit Rows": len(audit),
        "Feature Count": len(features),
        "Features": " | ".join(features),
        "Actual Positive Share": float(actual.mean()),
    }
    row.update(classification_metrics(actual, prediction, probability))

    output = audit[["Date", "Target_Date"]].copy()
    output[design_column] = design
    output["Actual"] = actual
    output["Probability"] = probability
    output["Prediction"] = prediction
    return row, output


def evaluate_collection(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    designs: dict[str, list[str]],
    design_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    audit_frames: list[pd.DataFrame] = []
    for design, features in designs.items():
        folds, oof = walk_forward_oof(
            development,
            features,
            design,
            design_column,
        )
        audit_row, audit_predictions = fit_audit(
            development,
            audit,
            features,
            design,
            design_column,
        )
        fold_frames.append(folds)
        oof_frames.append(oof)
        audit_rows.append(audit_row)
        audit_frames.append(audit_predictions)
    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(oof_frames, ignore_index=True),
        pd.DataFrame(audit_rows),
        pd.concat(audit_frames, ignore_index=True),
    )


def circular_block_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block_count = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=block_count)
    blocks = [
        (start + np.arange(block_length)) % length
        for start in starts
    ]
    return np.concatenate(blocks)[:length]


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    group_column: str,
    group_a: str,
    group_b: str,
    period: str,
) -> dict[str, object]:
    pivot = predictions.pivot_table(
        index=["Date", "Actual"],
        columns=group_column,
        values="Prediction",
        aggfunc="first",
    ).reset_index()
    if group_a not in pivot or group_b not in pivot:
        raise AssertionError(f"Missing comparison columns: {group_a}, {group_b}")
    pivot = pivot.sort_values("Date").dropna(subset=[group_a, group_b])
    actual = pivot["Actual"].astype(int).to_numpy()
    correct_a = (pivot[group_a].astype(int).to_numpy() == actual).astype(float)
    correct_b = (pivot[group_b].astype(int).to_numpy() == actual).astype(float)
    paired_difference = correct_a - correct_b

    rng = np.random.default_rng(RANDOM_STATE)
    distribution = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = circular_block_indices(len(pivot), BLOCK_LENGTH, rng)
        distribution[replicate] = paired_difference[index].mean()

    return {
        "Period": period,
        "Group A": group_a,
        "Group B": group_b,
        "Rows": len(pivot),
        "Point Difference": float(paired_difference.mean()),
        "CI Lower 95%": float(np.quantile(distribution, 0.025)),
        "CI Upper 95%": float(np.quantile(distribution, 0.975)),
        "Share Difference <= 0": float(np.mean(distribution <= 0)),
        "Share Difference >= 0": float(np.mean(distribution >= 0)),
        "Two-sided Bootstrap p": float(
            min(
                1.0,
                2
                * min(
                    (1 + np.sum(distribution <= 0))
                    / (BOOTSTRAP_REPLICATES + 1),
                    (1 + np.sum(distribution >= 0))
                    / (BOOTSTRAP_REPLICATES + 1),
                ),
            )
        ),
        "Bootstrap Replicates": BOOTSTRAP_REPLICATES,
        "Block Length": BLOCK_LENGTH,
    }


def bootstrap_comparisons(
    predictions: pd.DataFrame,
    group_column: str,
    primary: str,
    alternatives: Iterable[str],
    period: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            paired_block_bootstrap(
                predictions,
                group_column,
                primary,
                alternative,
                period,
            )
            for alternative in alternatives
        ]
    )


def add_holm_adjustment(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Add Holm-adjusted p-values within each saved comparison period."""
    adjusted_frames: list[pd.DataFrame] = []
    for _, period_frame in comparisons.groupby("Period", sort=False):
        frame = period_frame.copy()
        p_values = frame["Two-sided Bootstrap p"].to_numpy()
        order = np.argsort(p_values)
        adjusted = np.empty(len(frame), dtype=float)
        running_max = 0.0
        count = len(frame)
        for rank, position in enumerate(order):
            candidate = min(1.0, (count - rank) * p_values[position])
            running_max = max(running_max, candidate)
            adjusted[position] = running_max
        frame["Holm-adjusted p"] = adjusted
        adjusted_frames.append(frame)
    return pd.concat(adjusted_frames, ignore_index=True)


def phase_refit_evaluation(
    development: pd.DataFrame,
    audit: pd.DataFrame,
    designs: dict[str, list[str]],
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit each design independently within five every-fifth-row phases."""
    fold_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    audit_prediction_frames: list[pd.DataFrame] = []

    for group, features in designs.items():
        for phase in range(HORIZON):
            phase_development = development.loc[
                development["Phase_Offset"].eq(phase)
            ].copy()
            phase_audit = audit.loc[audit["Phase_Offset"].eq(phase)].copy()
            splitter = TimeSeriesSplit(
                n_splits=N_SPLITS,
                test_size=50,
                gap=1,
            )
            for fold, (train_index, test_index) in enumerate(
                splitter.split(phase_development),
                start=1,
            ):
                train = phase_development.iloc[train_index]
                test = phase_development.iloc[test_index]
                if train["Target_Date"].max() >= test["Date"].min():
                    raise AssertionError(
                        f"Phase {phase} fold {fold} has label overlap."
                    )
                model = logistic_regression()
                y_train = train["Weekly_Trend_Direction"].astype(int)
                actual = test["Weekly_Trend_Direction"].astype(int).to_numpy()
                model.fit(train[features], y_train)
                probability = model.predict_proba(test[features])[:, 1]
                prediction = (probability >= 0.5).astype(int)
                fold_row: dict[str, object] = {
                    group_column: group,
                    "Phase Offset": phase,
                    "Fold": fold,
                    "Train Start": train["Date"].min(),
                    "Train End": train["Date"].max(),
                    "Train Target End": train["Target_Date"].max(),
                    "Test Start": test["Date"].min(),
                    "Test End": test["Date"].max(),
                    "Train Rows": len(train),
                    "Test Rows": len(test),
                }
                fold_row.update(
                    classification_metrics(actual, prediction, probability)
                )
                fold_rows.append(fold_row)

            model = logistic_regression()
            y_train = phase_development[
                "Weekly_Trend_Direction"
            ].astype(int)
            actual = phase_audit["Weekly_Trend_Direction"].astype(int).to_numpy()
            model.fit(phase_development[features], y_train)
            probability = model.predict_proba(phase_audit[features])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            audit_row: dict[str, object] = {
                group_column: group,
                "Phase Offset": phase,
                "Development Rows": len(phase_development),
                "Rows": len(phase_audit),
                "Start": phase_audit["Date"].min(),
                "End": phase_audit["Date"].max(),
            }
            audit_row.update(
                classification_metrics(actual, prediction, probability)
            )
            audit_rows.append(audit_row)

            output = phase_audit[
                ["Date", "Target_Date", "Eligible_Row_ID", "Phase_Offset"]
            ].copy()
            output[group_column] = group
            output["Actual"] = actual
            output["Probability"] = probability
            output["Prediction"] = prediction
            audit_prediction_frames.append(output)

    fold_metrics = pd.DataFrame(fold_rows)
    audit_metrics = pd.DataFrame(audit_rows)
    audit_predictions = pd.concat(audit_prediction_frames, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for group, group_metrics in audit_metrics.groupby(group_column, sort=False):
        group_predictions = audit_predictions.loc[
            audit_predictions[group_column].eq(group)
        ].sort_values("Date")
        actual = group_predictions["Actual"].astype(int).to_numpy()
        prediction = group_predictions["Prediction"].astype(int).to_numpy()
        probability = group_predictions["Probability"].to_numpy()
        summary_row: dict[str, object] = {
            group_column: group,
            "Phase_Count": group_metrics["Phase Offset"].nunique(),
            "Total_Rows": int(group_metrics["Rows"].sum()),
            "Mean_Accuracy": group_metrics["Accuracy"].mean(),
            "Minimum_Accuracy": group_metrics["Accuracy"].min(),
            "Maximum_Accuracy": group_metrics["Accuracy"].max(),
            "Accuracy_SD": group_metrics["Accuracy"].std(),
            "Mean_Balanced_Accuracy": group_metrics[
                "Balanced Accuracy"
            ].mean(),
            "Minimum_Balanced_Accuracy": group_metrics[
                "Balanced Accuracy"
            ].min(),
            "Maximum_Balanced_Accuracy": group_metrics[
                "Balanced Accuracy"
            ].max(),
        }
        pooled = classification_metrics(actual, prediction, probability)
        summary_row.update(
            {f"Pooled_{key.replace(' ', '_')}": value for key, value in pooled.items()}
        )
        summary_rows.append(summary_row)
    return (
        fold_metrics,
        audit_metrics,
        audit_predictions,
        pd.DataFrame(summary_rows),
    )


def baseline_audit(
    development: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = audit["Weekly_Trend_Direction"].astype(int).to_numpy()
    positive_share = float(
        development["Weekly_Trend_Direction"].astype(int).mean()
    )
    baselines = {
        "Training-majority": (
            np.full(len(audit), int(positive_share >= 0.5), dtype=int),
            np.full(len(audit), positive_share),
        ),
        "Weekly-trend persistence": (
            audit["Weekly_Persistence"].astype(int).to_numpy(),
            audit["Weekly_Persistence"].astype(float).to_numpy(),
        ),
    }
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for baseline, (prediction, probability) in baselines.items():
        row: dict[str, object] = {
            "Baseline": baseline,
            "Audit Rows": len(audit),
            "Audit Start": audit["Date"].min(),
            "Audit End": audit["Date"].max(),
        }
        row.update(classification_metrics(actual, prediction, probability))
        rows.append(row)
        frame = audit[["Date", "Target_Date"]].copy()
        frame["Baseline"] = baseline
        frame["Actual"] = actual
        frame["Prediction"] = prediction
        frame["Probability"] = probability
        frames.append(frame)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def save_figures(
    family_audit: pd.DataFrame,
    technical_folds: pd.DataFrame,
    technical_audit: pd.DataFrame,
) -> None:
    family_plot = family_audit.sort_values("Accuracy", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(family_plot))
    axis.barh(
        positions - 0.16,
        family_plot["Accuracy"] * 100,
        height=0.32,
        label="Accuracy",
        color="#2563EB",
    )
    axis.barh(
        positions + 0.16,
        family_plot["Balanced Accuracy"] * 100,
        height=0.32,
        label="Balanced Accuracy",
        color="#14B8A6",
    )
    axis.set_yticks(positions, family_plot["Feature Family"])
    axis.set_xlabel("Retrospective audit performance (%)")
    axis.set_title("RQ2 standalone feature families under common Logistic Regression")
    axis.set_xlim(45, 70)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "rq2_feature_family_audit.png", dpi=180)
    plt.close(fig)

    development = (
        technical_folds.groupby("Technical Design", as_index=False)["Accuracy"]
        .mean()
        .rename(columns={"Accuracy": "Development OOF"})
    )
    audit = technical_audit[
        ["Technical Design", "Accuracy"]
    ].rename(columns={"Accuracy": "Retrospective audit"})
    comparison = development.merge(audit, on="Technical Design")
    x = np.arange(len(comparison))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        x - width / 2,
        comparison["Development OOF"] * 100,
        width,
        label="Development OOF mean",
        color="#7C3AED",
    )
    axis.bar(
        x + width / 2,
        comparison["Retrospective audit"] * 100,
        width,
        label="Retrospective audit",
        color="#F59E0B",
    )
    axis.set_xticks(x, comparison["Technical Design"], rotation=15, ha="right")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Technical moving-average expression sensitivity")
    axis.set_ylim(55, 75)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rq2_technical_design_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_data_hash = protocol["input_snapshot"]["sha256"]
    actual_data_hash = file_sha256(DATA)
    if actual_data_hash != expected_data_hash:
        raise AssertionError(
            f"Input snapshot hash mismatch: {actual_data_hash}"
        )
    macro_hashes = {
        "CPIAUCSL": file_sha256(CPI_DATA),
        "UNRATE": file_sha256(UNRATE_DATA),
    }
    for series_id, actual_hash in macro_hashes.items():
        expected_hash = protocol["macro_source_snapshots"][series_id]["sha256"]
        if actual_hash != expected_hash:
            raise AssertionError(
                f"{series_id} source hash mismatch: {actual_hash}"
            )

    data = load_data()
    development, audit = development_and_audit(data)

    (
        family_folds,
        family_oof,
        family_audit,
        family_audit_predictions,
    ) = evaluate_collection(
        development,
        audit,
        FEATURE_FAMILIES,
        "Feature Family",
    )
    (
        technical_folds,
        technical_oof,
        technical_audit,
        technical_audit_predictions,
    ) = evaluate_collection(
        development,
        audit,
        TECHNICAL_DESIGNS,
        "Technical Design",
    )

    family_folds.to_csv(
        OUTPUTS / "rq2_feature_family_fold_metrics.csv",
        index=False,
    )
    family_oof.to_csv(
        OUTPUTS / "rq2_feature_family_oof_predictions.csv",
        index=False,
    )
    family_audit.to_csv(
        OUTPUTS / "rq2_feature_family_audit_metrics.csv",
        index=False,
    )
    family_audit_predictions.to_csv(
        OUTPUTS / "rq2_feature_family_audit_predictions.csv",
        index=False,
    )
    technical_folds.to_csv(
        OUTPUTS / "rq2_technical_design_fold_metrics.csv",
        index=False,
    )
    technical_oof.to_csv(
        OUTPUTS / "rq2_technical_design_oof_predictions.csv",
        index=False,
    )
    technical_audit.to_csv(
        OUTPUTS / "rq2_technical_design_audit_metrics.csv",
        index=False,
    )
    technical_audit_predictions.to_csv(
        OUTPUTS / "rq2_technical_design_audit_predictions.csv",
        index=False,
    )

    family_development_summary = (
        family_folds.groupby("Feature Family", as_index=False)
        .agg(
            Development_Mean_Accuracy=("Accuracy", "mean"),
            Development_Accuracy_SD=("Accuracy", "std"),
            Development_Min_Accuracy=("Accuracy", "min"),
            Development_Mean_Balanced_Accuracy=(
                "Balanced Accuracy",
                "mean",
            ),
            Development_Mean_Macro_F1=("Macro F1", "mean"),
            Development_Mean_ROC_AUC=("ROC-AUC", "mean"),
        )
    )
    family_summary = family_development_summary.merge(
        family_audit[
            [
                "Feature Family",
                "Audit Rows",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "MCC",
                "Recall Down",
                "Recall Up",
                "ROC-AUC",
            ]
        ].rename(
            columns={
                "Accuracy": "Audit_Accuracy",
                "Balanced Accuracy": "Audit_Balanced_Accuracy",
                "Macro F1": "Audit_Macro_F1",
                "MCC": "Audit_MCC",
                "Recall Down": "Audit_Recall_Down",
                "Recall Up": "Audit_Recall_Up",
                "ROC-AUC": "Audit_ROC_AUC",
            }
        ),
        on="Feature Family",
        validate="one_to_one",
    )
    family_summary.to_csv(
        OUTPUTS / "rq2_feature_family_summary.csv",
        index=False,
    )

    technical_development_summary = (
        technical_folds.groupby("Technical Design", as_index=False)
        .agg(
            Development_Mean_Accuracy=("Accuracy", "mean"),
            Development_Accuracy_SD=("Accuracy", "std"),
            Development_Min_Accuracy=("Accuracy", "min"),
            Development_Mean_Balanced_Accuracy=(
                "Balanced Accuracy",
                "mean",
            ),
            Development_Mean_Macro_F1=("Macro F1", "mean"),
            Development_Mean_ROC_AUC=("ROC-AUC", "mean"),
        )
    )
    technical_summary = technical_development_summary.merge(
        technical_audit[
            [
                "Technical Design",
                "Audit Rows",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "MCC",
                "Recall Down",
                "Recall Up",
                "ROC-AUC",
            ]
        ].rename(
            columns={
                "Accuracy": "Audit_Accuracy",
                "Balanced Accuracy": "Audit_Balanced_Accuracy",
                "Macro F1": "Audit_Macro_F1",
                "MCC": "Audit_MCC",
                "Recall Down": "Audit_Recall_Down",
                "Recall Up": "Audit_Recall_Up",
                "ROC-AUC": "Audit_ROC_AUC",
            }
        ),
        on="Technical Design",
        validate="one_to_one",
    )
    technical_summary.to_csv(
        OUTPUTS / "rq2_technical_design_summary.csv",
        index=False,
    )

    family_alternatives = [
        name
        for name in FEATURE_FAMILIES
        if name != "Technical — EMA-aware compact"
    ]
    family_dev_bootstrap = bootstrap_comparisons(
        family_oof,
        "Feature Family",
        "Technical — EMA-aware compact",
        family_alternatives,
        "Development OOF",
    )
    family_audit_bootstrap = bootstrap_comparisons(
        family_audit_predictions,
        "Feature Family",
        "Technical — EMA-aware compact",
        family_alternatives,
        "Retrospective audit",
    )
    family_bootstrap = add_holm_adjustment(pd.concat(
        [family_dev_bootstrap, family_audit_bootstrap],
        ignore_index=True,
    ))
    family_bootstrap.to_csv(
        OUTPUTS / "rq2_feature_family_paired_block_bootstrap.csv",
        index=False,
    )

    technical_alternatives = [
        name
        for name in TECHNICAL_DESIGNS
        if name != "Primary EMA-aware compact"
    ]
    technical_dev_bootstrap = bootstrap_comparisons(
        technical_oof,
        "Technical Design",
        "Primary EMA-aware compact",
        technical_alternatives,
        "Development OOF",
    )
    technical_audit_bootstrap = bootstrap_comparisons(
        technical_audit_predictions,
        "Technical Design",
        "Primary EMA-aware compact",
        technical_alternatives,
        "Retrospective audit",
    )
    pd.concat(
        [technical_dev_bootstrap, technical_audit_bootstrap],
        ignore_index=True,
    ).to_csv(
        OUTPUTS / "rq2_technical_design_paired_block_bootstrap.csv",
        index=False,
    )

    (
        family_phase_folds,
        family_phase,
        family_phase_predictions,
        family_phase_summary,
    ) = phase_refit_evaluation(
        development,
        audit,
        FEATURE_FAMILIES,
        "Feature Family",
    )
    family_phase_folds.to_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_fold_metrics.csv",
        index=False,
    )
    family_phase.to_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_metrics.csv",
        index=False,
    )
    family_phase_predictions.to_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_predictions.csv",
        index=False,
    )
    family_phase_summary.to_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_summary.csv",
        index=False,
    )
    (
        technical_phase_folds,
        technical_phase,
        technical_phase_predictions,
        technical_phase_summary,
    ) = phase_refit_evaluation(
        development,
        audit,
        TECHNICAL_DESIGNS,
        "Technical Design",
    )
    technical_phase_folds.to_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_fold_metrics.csv",
        index=False,
    )
    technical_phase.to_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_metrics.csv",
        index=False,
    )
    technical_phase_predictions.to_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_predictions.csv",
        index=False,
    )
    technical_phase_summary.to_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_summary.csv",
        index=False,
    )

    baseline_metrics, baseline_predictions = baseline_audit(
        development,
        audit,
    )
    baseline_metrics.to_csv(
        OUTPUTS / "rq2_baseline_audit_metrics.csv",
        index=False,
    )
    baseline_predictions.to_csv(
        OUTPUTS / "rq2_baseline_audit_predictions.csv",
        index=False,
    )
    primary_predictions = family_audit_predictions.loc[
        family_audit_predictions["Feature Family"].eq(
            "Technical — EMA-aware compact"
        )
    ].copy()
    primary_predictions["Comparator"] = "Technical — EMA-aware compact"
    baseline_for_compare = baseline_predictions.copy()
    baseline_for_compare["Comparator"] = baseline_for_compare["Baseline"]
    combined_baseline_predictions = pd.concat(
        [
            primary_predictions[
                [
                    "Date",
                    "Target_Date",
                    "Comparator",
                    "Actual",
                    "Probability",
                    "Prediction",
                ]
            ],
            baseline_for_compare[
                [
                    "Date",
                    "Target_Date",
                    "Comparator",
                    "Actual",
                    "Probability",
                    "Prediction",
                ]
            ],
        ],
        ignore_index=True,
    )
    bootstrap_comparisons(
        combined_baseline_predictions,
        "Comparator",
        "Technical — EMA-aware compact",
        ["Training-majority", "Weekly-trend persistence"],
        "Retrospective audit",
    ).to_csv(
        OUTPUTS / "rq2_primary_vs_baseline_paired_block_bootstrap.csv",
        index=False,
    )

    correlations = (
        development[EMA_TECHNICAL]
        .corr()
        .rename_axis(index="Feature A", columns="Feature B")
    )
    correlations.to_csv(
        OUTPUTS / "rq2_primary_technical_correlations.csv"
    )
    upper = correlations.abs().where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)
    )
    max_abs_correlation = float(upper.max().max())

    save_figures(family_audit, technical_folds, technical_audit)

    metadata = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "input_sha256": actual_data_hash,
        "macro_source_sha256": macro_hashes,
        "status": protocol["status"],
        "target": protocol["target"]["formula"],
        "data_rows_after_warmup": len(data),
        "development_rows": len(development),
        "development_start": development["Date"].min().strftime("%Y-%m-%d"),
        "development_end": development["Date"].max().strftime("%Y-%m-%d"),
        "development_target_end": development["Target_Date"]
        .max()
        .strftime("%Y-%m-%d"),
        "audit_rows": len(audit),
        "audit_start": audit["Date"].min().strftime("%Y-%m-%d"),
        "audit_end": audit["Date"].max().strftime("%Y-%m-%d"),
        "feature_families": FEATURE_FAMILIES,
        "technical_designs": TECHNICAL_DESIGNS,
        "common_model": protocol["common_model"],
        "maximum_absolute_primary_technical_correlation_development": (
            max_abs_correlation
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_block_length": BLOCK_LENGTH,
        "phase_offsets": list(range(HORIZON)),
        "claim_boundary": protocol["claim_boundary"],
    }
    (OUTPUTS / "rq2_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("RQ2 feature-family summary")
    print(
        family_summary[
            [
                "Feature Family",
                "Development_Mean_Accuracy",
                "Development_Accuracy_SD",
                "Audit_Accuracy",
                "Audit_Balanced_Accuracy",
                "Audit_Macro_F1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nTechnical design summary")
    print(
        technical_summary[
            [
                "Technical Design",
                "Development_Mean_Accuracy",
                "Development_Accuracy_SD",
                "Audit_Accuracy",
                "Audit_Balanced_Accuracy",
                "Audit_Macro_F1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
