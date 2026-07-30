"""Fold-aware uncertainty analysis for the retrospective TSLA OOF study.

These intervals quantify sampling uncertainty within the historical OOF
window.  They do not turn an iteratively inspected development period into a
prospectively untouched confirmatory test.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .backtesting import ML_STRATEGY_SPEC, RULE_STRATEGY_SPECS
from .config import MAJORITY_MODEL_NAME, TARGET_RETURN_COLUMN
from .statistics import fold_aware_moving_block_indices


DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BLOCK_LENGTH = 20
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_STATE = 42


def _ordered_prediction_groups(
    predictions: pd.DataFrame,
) -> tuple[list[str], dict[str, pd.DataFrame], np.ndarray]:
    required = {
        "Model",
        "Date",
        "Fold",
        "Actual Direction",
        "Predicted Direction",
        "Signal",
        TARGET_RETURN_COLUMN,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    if predictions.empty:
        raise ValueError("OOF predictions must not be empty.")

    prepared = predictions.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="raise")
    if prepared.duplicated(["Model", "Date"]).any():
        raise ValueError("Each model must have exactly one row per OOF date.")

    model_order = prepared["Model"].drop_duplicates().astype(str).tolist()
    groups = {
        model: (
            prepared.loc[prepared["Model"].astype(str).eq(model)]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        for model in model_order
    }
    reference = groups[model_order[0]]
    reference_dates = pd.DatetimeIndex(reference["Date"])
    reference_folds = reference["Fold"].to_numpy()
    reference_actual = pd.to_numeric(
        reference["Actual Direction"], errors="raise"
    ).astype(int).to_numpy()
    reference_returns = pd.to_numeric(
        reference[TARGET_RETURN_COLUMN], errors="raise"
    ).astype(float).to_numpy()

    for model, group in groups.items():
        if not pd.DatetimeIndex(group["Date"]).equals(reference_dates):
            raise ValueError(f"OOF dates are not aligned for {model}.")
        if not np.array_equal(group["Fold"].to_numpy(), reference_folds):
            raise ValueError(f"OOF folds are not aligned for {model}.")
        actual = pd.to_numeric(
            group["Actual Direction"], errors="raise"
        ).astype(int).to_numpy()
        if not np.array_equal(actual, reference_actual):
            raise ValueError(f"Actual directions are not aligned for {model}.")
        returns = pd.to_numeric(
            group[TARGET_RETURN_COLUMN], errors="raise"
        ).astype(float).to_numpy()
        if not np.allclose(returns, reference_returns, rtol=0.0, atol=1e-15):
            raise ValueError(f"Target returns are not aligned for {model}.")
        signal = pd.to_numeric(group["Signal"], errors="raise").astype(int)
        predicted = pd.to_numeric(
            group["Predicted Direction"], errors="raise"
        ).astype(int)
        if not signal.equals(predicted):
            raise ValueError(f"Signal and predicted direction differ for {model}.")
        if not set(signal.unique()).issubset({0, 1}):
            raise ValueError(f"Signals must be binary for {model}.")
    return model_order, groups, reference_folds


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    true_positive = np.sum((actual == 1) & (predicted == 1))
    true_negative = np.sum((actual == 0) & (predicted == 0))
    false_positive = np.sum((actual == 0) & (predicted == 1))
    false_negative = np.sum((actual == 1) & (predicted == 0))
    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = (
        2 * true_positive / positive_denominator
        if positive_denominator
        else 0.0
    )
    negative_f1 = (
        2 * true_negative / negative_denominator
        if negative_denominator
        else 0.0
    )
    return float((positive_f1 + negative_f1) / 2.0)


def _macro_f1_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> np.ndarray:
    sampled_actual = actual[bootstrap_indices]
    sampled_predicted = predicted[bootstrap_indices]
    true_positive = np.sum(
        (sampled_actual == 1) & (sampled_predicted == 1),
        axis=1,
    )
    true_negative = np.sum(
        (sampled_actual == 0) & (sampled_predicted == 0),
        axis=1,
    )
    false_positive = np.sum(
        (sampled_actual == 0) & (sampled_predicted == 1),
        axis=1,
    )
    false_negative = np.sum(
        (sampled_actual == 1) & (sampled_predicted == 0),
        axis=1,
    )
    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = np.divide(
        2 * true_positive,
        positive_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=positive_denominator != 0,
    )
    negative_f1 = np.divide(
        2 * true_negative,
        negative_denominator,
        out=np.zeros_like(true_negative, dtype=float),
        where=negative_denominator != 0,
    )
    return (positive_f1 + negative_f1) / 2.0


def _interval(distribution: Iterable[float], confidence_level: float) -> tuple[float, float]:
    values = np.asarray(list(distribution), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite bootstrap replicates were produced.")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _direction_label(lower: float, upper: float, reference: float) -> str:
    if lower > reference:
        return "Entire interval above reference"
    if upper < reference:
        return "Entire interval below reference"
    return "Interval includes reference"


def build_classification_uncertainty(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Build absolute and Majority-paired OOF classification intervals."""

    model_order, groups, folds = _ordered_prediction_groups(predictions)
    if MAJORITY_MODEL_NAME not in groups:
        raise ValueError("The Majority baseline is required for paired inference.")
    indices = fold_aware_moving_block_indices(
        folds,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        random_state=random_state,
    )
    actual = pd.to_numeric(
        groups[model_order[0]]["Actual Direction"], errors="raise"
    ).astype(int).to_numpy()

    points: dict[str, dict[str, float]] = {}
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for model in model_order:
        predicted = pd.to_numeric(
            groups[model]["Predicted Direction"], errors="raise"
        ).astype(int).to_numpy()
        points[model] = {
            "Accuracy": float(np.mean(predicted == actual)),
            "Macro F1": _macro_f1(actual, predicted),
        }
        distributions[model] = {
            "Accuracy": np.mean(
                predicted[indices] == actual[indices],
                axis=1,
            ),
            "Macro F1": _macro_f1_distribution(actual, predicted, indices),
        }

    common = {
        "Confidence Level": confidence_level,
        "Requested Replicates": n_bootstrap,
        "Block Length": block_length,
        "Fold Count": int(pd.Series(folds).nunique()),
        "Random State": random_state,
    }
    rows: list[dict[str, object]] = []
    for model in model_order:
        for metric in ("Accuracy", "Macro F1"):
            distribution = distributions[model][metric]
            lower, upper = _interval(distribution, confidence_level)
            reference = 0.50 if metric == "Accuracy" else np.nan
            rows.append(
                {
                    "Analysis": "Absolute OOF metric",
                    "Model": model,
                    "Reference Model": "",
                    "Metric": metric,
                    "Point Estimate": points[model][metric],
                    "Reference Value": reference,
                    "Difference": (
                        points[model][metric] - reference
                        if np.isfinite(reference)
                        else np.nan
                    ),
                    "CI Lower": lower,
                    "CI Upper": upper,
                    "Bootstrap Share Above Reference": (
                        float(np.mean(distribution > reference))
                        if np.isfinite(reference)
                        else np.nan
                    ),
                    "CI Excludes Reference": (
                        bool(lower > reference or upper < reference)
                        if np.isfinite(reference)
                        else False
                    ),
                    "Interpretation": (
                        _direction_label(lower, upper, reference)
                        if np.isfinite(reference)
                        else "Absolute pointwise interval; no null reference assigned"
                    ),
                    "Valid Replicates": int(np.isfinite(distribution).sum()),
                    **common,
                }
            )

    majority_points = points[MAJORITY_MODEL_NAME]
    majority_distributions = distributions[MAJORITY_MODEL_NAME]
    for model in model_order:
        if model == MAJORITY_MODEL_NAME:
            continue
        for metric in ("Accuracy", "Macro F1"):
            distribution = (
                distributions[model][metric] - majority_distributions[metric]
            )
            lower, upper = _interval(distribution, confidence_level)
            rows.append(
                {
                    "Analysis": "Paired difference vs Majority",
                    "Model": model,
                    "Reference Model": MAJORITY_MODEL_NAME,
                    "Metric": metric,
                    "Point Estimate": points[model][metric],
                    "Reference Value": majority_points[metric],
                    "Difference": points[model][metric] - majority_points[metric],
                    "CI Lower": lower,
                    "CI Upper": upper,
                    "Bootstrap Share Above Reference": float(
                        np.mean(distribution > 0.0)
                    ),
                    "CI Excludes Reference": bool(lower > 0.0 or upper < 0.0),
                    "Interpretation": _direction_label(lower, upper, 0.0),
                    "Valid Replicates": int(np.isfinite(distribution).sum()),
                    **common,
                }
            )
    return pd.DataFrame(rows)


def build_return_separation_uncertainty(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Estimate uncertainty in long-signal minus cash-signal gross returns."""

    model_order, groups, folds = _ordered_prediction_groups(predictions)
    indices = fold_aware_moving_block_indices(
        folds,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        random_state=random_state,
    )
    rows: list[dict[str, object]] = []
    for model in model_order:
        group = groups[model]
        signal = pd.to_numeric(group["Signal"], errors="raise").astype(int).to_numpy()
        returns = pd.to_numeric(
            group[TARGET_RETURN_COLUMN], errors="raise"
        ).astype(float).to_numpy()
        if np.unique(signal).size < 2:
            raise ValueError(f"Return separation requires both signal states for {model}.")
        long_mean = float(returns[signal == 1].mean())
        cash_mean = float(returns[signal == 0].mean())

        sampled_signal = signal[indices]
        sampled_returns = returns[indices]
        long_count = np.sum(sampled_signal == 1, axis=1)
        cash_count = np.sum(sampled_signal == 0, axis=1)
        valid = (long_count > 0) & (cash_count > 0)
        long_bootstrap = np.divide(
            np.sum(sampled_returns * (sampled_signal == 1), axis=1),
            long_count,
            out=np.full(len(indices), np.nan),
            where=long_count > 0,
        )
        cash_bootstrap = np.divide(
            np.sum(sampled_returns * (sampled_signal == 0), axis=1),
            cash_count,
            out=np.full(len(indices), np.nan),
            where=cash_count > 0,
        )
        distribution = (long_bootstrap - cash_bootstrap)[valid]
        lower, upper = _interval(distribution, confidence_level)
        rows.append(
            {
                "Model": model,
                "Metric": "Gross Return Separation",
                "Long-Signal Mean Gross Return": long_mean,
                "Cash-Signal Market Mean Gross Return": cash_mean,
                "Point Estimate": long_mean - cash_mean,
                "Reference Value": 0.0,
                "CI Lower": lower,
                "CI Upper": upper,
                "Confidence Level": confidence_level,
                "Bootstrap Share Above 0": float(np.mean(distribution > 0.0)),
                "CI Excludes Zero": bool(lower > 0.0 or upper < 0.0),
                "Interpretation": _direction_label(lower, upper, 0.0),
                "Requested Replicates": n_bootstrap,
                "Valid Replicates": len(distribution),
                "Block Length": block_length,
                "Fold Count": int(pd.Series(folds).nunique()),
                "Random State": random_state,
            }
        )
    return pd.DataFrame(rows)


def build_strategy_benchmark_uncertainty(
    predictions: pd.DataFrame,
    execution: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Compare each model's net daily return with all eight rule benchmarks."""

    model_order, groups, folds = _ordered_prediction_groups(predictions)
    required = {
        "Signal Model",
        "Signal Date",
        ML_STRATEGY_SPEC[1],
        *(return_column for _, return_column, _ in RULE_STRATEGY_SPECS),
    }
    missing = sorted(required.difference(execution.columns))
    if missing:
        raise ValueError(f"OOF execution is missing columns: {missing}")
    prepared = execution.copy()
    prepared["Signal Date"] = pd.to_datetime(
        prepared["Signal Date"], errors="raise"
    )
    if prepared.duplicated(["Signal Model", "Signal Date"]).any():
        raise ValueError("Execution rows must be unique by signal model and date.")

    execution_groups = {
        model: (
            prepared.loc[prepared["Signal Model"].astype(str).eq(model)]
            .sort_values("Signal Date")
            .reset_index(drop=True)
        )
        for model in model_order
    }
    reference = execution_groups[model_order[0]]
    expected_dates = pd.DatetimeIndex(groups[model_order[0]]["Date"])
    if not pd.DatetimeIndex(reference["Signal Date"]).equals(expected_dates):
        raise ValueError("Prediction and execution signal dates are not aligned.")
    for model, group in execution_groups.items():
        if group.empty:
            raise ValueError(f"OOF execution has no rows for {model}.")
        if not pd.DatetimeIndex(group["Signal Date"]).equals(expected_dates):
            raise ValueError(f"Execution dates are not aligned for {model}.")
        for _, return_column, _ in RULE_STRATEGY_SPECS:
            values = pd.to_numeric(group[return_column], errors="raise").to_numpy()
            reference_values = pd.to_numeric(
                reference[return_column], errors="raise"
            ).to_numpy()
            if not np.allclose(values, reference_values, rtol=0.0, atol=1e-15):
                raise ValueError(
                    f"Benchmark return {return_column} differs across model copies."
                )

    indices = fold_aware_moving_block_indices(
        folds,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        random_state=random_state,
    )
    rows: list[dict[str, object]] = []
    for model in model_order:
        group = execution_groups[model]
        model_returns = pd.to_numeric(
            group[ML_STRATEGY_SPEC[1]], errors="raise"
        ).astype(float).to_numpy()
        for benchmark, return_column, _ in RULE_STRATEGY_SPECS:
            benchmark_returns = pd.to_numeric(
                group[return_column], errors="raise"
            ).astype(float).to_numpy()
            daily_difference = model_returns - benchmark_returns
            distribution = daily_difference[indices].mean(axis=1)
            lower, upper = _interval(distribution, confidence_level)
            rows.append(
                {
                    "Model": model,
                    "Benchmark": benchmark,
                    "Metric": "Mean Daily Net Return Difference",
                    "Model Mean Daily Net Return": float(model_returns.mean()),
                    "Benchmark Mean Daily Net Return": float(
                        benchmark_returns.mean()
                    ),
                    "Point Estimate": float(daily_difference.mean()),
                    "Reference Value": 0.0,
                    "CI Lower": lower,
                    "CI Upper": upper,
                    "Confidence Level": confidence_level,
                    "Bootstrap Share Above 0": float(
                        np.mean(distribution > 0.0)
                    ),
                    "CI Excludes Zero": bool(lower > 0.0 or upper < 0.0),
                    "Interpretation": _direction_label(lower, upper, 0.0),
                    "Requested Replicates": n_bootstrap,
                    "Valid Replicates": int(np.isfinite(distribution).sum()),
                    "Block Length": block_length,
                    "Fold Count": int(pd.Series(folds).nunique()),
                    "Random State": random_state,
                }
            )
    return pd.DataFrame(rows)
