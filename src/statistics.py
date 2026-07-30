"""Statistical inference utilities for the main and supplementary studies.

The functions in this module operate on chronologically out-of-fold
probabilities.  They deliberately avoid fitting models so that the same
inference code can be reused across model families, feature specifications,
assets, and prediction targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score


MetricName = Literal["Macro F1", "PR-AUC", "Inverse Brier"]
Alternative = Literal["greater", "less", "two-sided"]


_METRIC_ALIASES: dict[str, MetricName] = {
    "macro f1": "Macro F1",
    "macro-f1": "Macro F1",
    "macro_f1": "Macro F1",
    "f1 macro": "Macro F1",
    "pr auc": "PR-AUC",
    "pr-auc": "PR-AUC",
    "pr_auc": "PR-AUC",
    "average precision": "PR-AUC",
    "average_precision": "PR-AUC",
    "ap": "PR-AUC",
    "inverse brier": "Inverse Brier",
    "inverse-brier": "Inverse Brier",
    "inverse_brier": "Inverse Brier",
    "1-brier": "Inverse Brier",
}


@dataclass(frozen=True)
class BootstrapDifferenceResult:
    """Summary of a paired moving-block bootstrap comparison.

    ``difference`` and every element of ``bootstrap_differences`` are defined
    as comparison minus baseline.  For Inverse Brier this is equivalent to
    baseline Brier minus comparison Brier, so positive values consistently
    favour the comparison specification for every supported metric.
    """

    metric: MetricName
    baseline_score: float
    comparison_score: float
    difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    probability_positive: float
    requested_replicates: int
    valid_replicates: int
    block_length: int
    fold_count: int
    random_state: int
    bootstrap_differences: np.ndarray = field(repr=False, compare=False)

    def to_record(self) -> dict[str, float | int | str]:
        """Return a compact row suitable for an examiner-facing CSV table."""

        return {
            "Metric": self.metric,
            "Baseline Score": self.baseline_score,
            "Comparison Score": self.comparison_score,
            "Difference": self.difference,
            "CI Lower": self.ci_lower,
            "CI Upper": self.ci_upper,
            "Confidence Level": self.confidence_level,
            "Bootstrap P(Delta > 0)": self.probability_positive,
            "Requested Replicates": self.requested_replicates,
            "Valid Replicates": self.valid_replicates,
            "Block Length": self.block_length,
            "Fold Count": self.fold_count,
            "Random State": self.random_state,
        }


@dataclass(frozen=True)
class PlaceboShiftResult:
    """A jointly shifted copy of the input data and its shift audit."""

    data: pd.DataFrame
    audit: pd.DataFrame


def _canonical_metric_name(metric: str) -> MetricName:
    key = str(metric).strip().lower()
    try:
        return _METRIC_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(["Macro F1", "PR-AUC", "Inverse Brier"])
        raise ValueError(f"Unsupported metric {metric!r}. Expected one of: {supported}.") from exc


def _as_binary_array(values: Iterable[int | float], *, name: str) -> np.ndarray:
    result = np.asarray(list(values))
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if pd.isna(result).any():
        raise ValueError(f"{name} contains missing values.")
    try:
        numeric = result.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary 0/1 values.") from exc
    if not np.array_equal(result.astype(float), numeric.astype(float)):
        raise ValueError(f"{name} must contain binary 0/1 values.")
    if not set(np.unique(numeric)).issubset({0, 1}):
        raise ValueError(f"{name} must contain binary 0/1 values.")
    return numeric


def _as_probability_array(
    values: Iterable[int | float],
    *,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probabilities.") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite probabilities.")
    if ((result < 0.0) | (result > 1.0)).any():
        raise ValueError(f"{name} probabilities must lie in [0, 1].")
    return result


def probability_metric(
    y_true: Iterable[int | float],
    probability: Iterable[int | float],
    *,
    metric: str,
    threshold: float = 0.50,
) -> float:
    """Evaluate one supported probability-based classification metric.

    ``Inverse Brier`` is defined as ``1 - Brier score``.  This preserves a
    common "higher is better" direction across all metrics while retaining the
    same pairwise difference as baseline-Brier minus comparison-Brier.
    """

    actual = _as_binary_array(y_true, name="y_true")
    predicted_probability = _as_probability_array(probability, name="probability")
    if len(actual) != len(predicted_probability):
        raise ValueError("y_true and probability must have the same length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")

    metric_name = _canonical_metric_name(metric)
    if metric_name == "Macro F1":
        prediction = (predicted_probability >= threshold).astype(int)
        return float(f1_score(actual, prediction, average="macro", zero_division=0))
    if metric_name == "PR-AUC":
        if np.unique(actual).size < 2:
            raise ValueError("PR-AUC requires both target classes.")
        return float(average_precision_score(actual, predicted_probability))
    return float(1.0 - brier_score_loss(actual, predicted_probability))


def _normalise_folds(
    folds: Iterable[object] | None,
    *,
    row_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if folds is None:
        labels = np.zeros(row_count, dtype=int)
    else:
        raw = np.asarray(list(folds), dtype=object)
        if raw.ndim != 1 or len(raw) != row_count:
            raise ValueError("folds must be one-dimensional and match y_true length.")
        if pd.isna(raw).any():
            raise ValueError("folds contains missing values.")
        labels = raw

    # Preserve first-appearance order rather than sorting arbitrary mixed-type
    # labels.  Within-fold row order remains the caller's chronological order.
    unique_labels = list(dict.fromkeys(labels.tolist()))
    positions = [np.flatnonzero(labels == label) for label in unique_labels]
    if not positions or any(len(group) == 0 for group in positions):
        raise ValueError("At least one non-empty fold is required.")
    return labels, positions


def _sample_fold_moving_blocks(
    positions: np.ndarray,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample overlapping chronological blocks without crossing a fold."""

    fold_size = len(positions)
    local_block_length = min(block_length, fold_size)
    possible_starts = np.arange(0, fold_size - local_block_length + 1)
    draw_count = ceil(fold_size / local_block_length)
    starts = rng.choice(possible_starts, size=draw_count, replace=True)
    local_indices = np.concatenate(
        [
            np.arange(start, start + local_block_length, dtype=int)
            for start in starts
        ]
    )[:fold_size]
    return positions[local_indices]


def fold_aware_moving_block_indices(
    folds: Iterable[object],
    *,
    n_bootstrap: int = 2_000,
    block_length: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """Return shared moving-block resamples without crossing outer folds.

    The returned array has shape ``(n_bootstrap, n_observations)``.  A caller
    can reuse the same rows for every model, metric and benchmark, preserving
    the pairing required by the main-study comparisons.
    """

    fold_values = np.asarray(list(folds), dtype=object)
    if fold_values.ndim != 1 or fold_values.size == 0:
        raise ValueError("folds must be a non-empty one-dimensional sequence.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    _, fold_positions = _normalise_folds(
        fold_values,
        row_count=len(fold_values),
    )
    rng = np.random.default_rng(random_state)
    return np.vstack(
        [
            np.concatenate(
                [
                    _sample_fold_moving_blocks(
                        positions,
                        block_length=block_length,
                        rng=rng,
                    )
                    for positions in fold_positions
                ]
            )
            for _ in range(n_bootstrap)
        ]
    )


def paired_moving_block_bootstrap(
    y_true: Iterable[int | float],
    baseline_probability: Iterable[int | float],
    comparison_probability: Iterable[int | float],
    *,
    folds: Iterable[object] | None = None,
    metric: str = "Macro F1",
    threshold: float = 0.50,
    n_bootstrap: int = 2_000,
    block_length: int = 10,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapDifferenceResult:
    """Estimate a paired metric-difference CI using fold-aware moving blocks.

    Every replicate resamples overlapping blocks separately inside each outer
    test fold.  The same sampled row indices are applied to both probability
    vectors, preserving the pairing needed for an incremental-value test.
    Blocks never cross fold boundaries.
    """

    actual = _as_binary_array(y_true, name="y_true")
    baseline = _as_probability_array(
        baseline_probability,
        name="baseline_probability",
    )
    comparison = _as_probability_array(
        comparison_probability,
        name="comparison_probability",
    )
    if not (len(actual) == len(baseline) == len(comparison)):
        raise ValueError(
            "y_true, baseline_probability, and comparison_probability "
            "must have the same length."
        )
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    metric_name = _canonical_metric_name(metric)
    _, fold_positions = _normalise_folds(folds, row_count=len(actual))
    baseline_score = probability_metric(
        actual,
        baseline,
        metric=metric_name,
        threshold=threshold,
    )
    comparison_score = probability_metric(
        actual,
        comparison,
        metric=metric_name,
        threshold=threshold,
    )

    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 20, n_bootstrap + 100)
    while len(differences) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled_indices = np.concatenate(
            [
                _sample_fold_moving_blocks(
                    positions,
                    block_length=block_length,
                    rng=rng,
                )
                for positions in fold_positions
            ]
        )
        sampled_actual = actual[sampled_indices]
        # A PR-AUC replicate with one class has no valid ranking interpretation.
        if metric_name == "PR-AUC" and np.unique(sampled_actual).size < 2:
            continue
        sampled_baseline = probability_metric(
            sampled_actual,
            baseline[sampled_indices],
            metric=metric_name,
            threshold=threshold,
        )
        sampled_comparison = probability_metric(
            sampled_actual,
            comparison[sampled_indices],
            metric=metric_name,
            threshold=threshold,
        )
        differences.append(sampled_comparison - sampled_baseline)

    if len(differences) < n_bootstrap:
        raise ValueError(
            "Could not obtain the requested number of valid bootstrap "
            "replicates. Check class coverage within folds."
        )

    distribution = np.asarray(differences, dtype=float)
    tail = (1.0 - confidence_level) / 2.0
    ci_lower, ci_upper = np.quantile(distribution, [tail, 1.0 - tail])
    return BootstrapDifferenceResult(
        metric=metric_name,
        baseline_score=baseline_score,
        comparison_score=comparison_score,
        difference=comparison_score - baseline_score,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=float(confidence_level),
        probability_positive=float(np.mean(distribution > 0.0)),
        requested_replicates=n_bootstrap,
        valid_replicates=len(distribution),
        block_length=block_length,
        fold_count=len(fold_positions),
        random_state=int(random_state),
        bootstrap_differences=distribution,
    )


def joint_circular_block_shift(
    frame: pd.DataFrame,
    news_columns: Sequence[str],
    *,
    block_length: int = 5,
    group_columns: Sequence[str] | None = None,
    min_shift_blocks: int = 1,
    random_state: int = 42,
) -> PlaceboShiftResult:
    """Circularly shift all news columns together by a block-aligned offset.

    A separate non-zero offset is sampled for each group (normally ``Asset``).
    All news columns, including attention, sentiment, and ``no_news``, receive
    the same row permutation, preserving their contemporaneous relationships
    while breaking their original alignment with returns.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if isinstance(news_columns, str):
        raise TypeError("news_columns must be a sequence of column names, not a string.")
    columns = list(dict.fromkeys(news_columns))
    if not columns:
        raise ValueError("news_columns must contain at least one column.")
    missing_news = [column for column in columns if column not in frame.columns]
    if missing_news:
        raise ValueError(f"Missing news columns: {', '.join(missing_news)}")
    groups = list(dict.fromkeys(group_columns or []))
    missing_groups = [column for column in groups if column not in frame.columns]
    if missing_groups:
        raise ValueError(f"Missing group columns: {', '.join(missing_groups)}")
    overlap = sorted(set(columns) & set(groups))
    if overlap:
        raise ValueError(
            "News and group columns must be distinct. Overlap: "
            + ", ".join(overlap)
        )
    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")
    if not isinstance(min_shift_blocks, int) or min_shift_blocks < 1:
        raise ValueError("min_shift_blocks must be a positive integer.")

    if groups:
        grouping = frame.groupby(groups, sort=False, dropna=False).indices
        grouped_positions = [
            (
                label if isinstance(label, tuple) else (label,),
                np.sort(np.asarray(positions, dtype=int)),
            )
            for label, positions in grouping.items()
        ]
    else:
        grouped_positions = [(("__all__",), np.arange(len(frame), dtype=int))]

    result = frame.copy()
    news_column_positions = [result.columns.get_loc(column) for column in columns]
    rng = np.random.default_rng(random_state)
    audit_rows: list[dict[str, object]] = []
    for label, positions in grouped_positions:
        row_count = len(positions)
        minimum_rows = min_shift_blocks * block_length
        candidates = np.arange(minimum_rows, row_count, block_length, dtype=int)
        if candidates.size == 0:
            raise ValueError(
                "Each placebo group must contain more rows than "
                "min_shift_blocks * block_length."
            )
        shift_rows = int(rng.choice(candidates))
        original_values = frame.iloc[positions][columns].to_numpy(copy=True)
        shifted_values = np.roll(original_values, shift=shift_rows, axis=0)
        result.iloc[positions, news_column_positions] = shifted_values
        audit_row: dict[str, object] = {}
        if groups:
            audit_row.update(dict(zip(groups, label)))
        else:
            audit_row["Group"] = "__all__"
        audit_row.update(
            {
                "Rows": row_count,
                "Shift Rows": shift_rows,
                "Shift Blocks": shift_rows // block_length,
                "Block Length": block_length,
                "Random State": random_state,
            }
        )
        audit_rows.append(audit_row)

    return PlaceboShiftResult(
        data=result,
        audit=pd.DataFrame(audit_rows),
    )


def plus_one_empirical_p_value(
    observed: float,
    null_distribution: Iterable[int | float],
    *,
    alternative: Alternative = "greater",
) -> float:
    """Return a finite-sample placebo p-value with the plus-one correction."""

    observed_value = float(observed)
    null = np.asarray(list(null_distribution), dtype=float)
    if not np.isfinite(observed_value):
        raise ValueError("observed must be finite.")
    if null.ndim != 1 or null.size == 0:
        raise ValueError("null_distribution must be a non-empty one-dimensional sequence.")
    if not np.isfinite(null).all():
        raise ValueError("null_distribution contains non-finite values.")
    if alternative == "greater":
        exceedances = int(np.count_nonzero(null >= observed_value))
    elif alternative == "less":
        exceedances = int(np.count_nonzero(null <= observed_value))
    elif alternative == "two-sided":
        exceedances = int(np.count_nonzero(np.abs(null) >= abs(observed_value)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")
    return float((exceedances + 1) / (len(null) + 1))


def benjamini_hochberg_qvalues(
    p_values: Iterable[int | float],
) -> np.ndarray:
    """Adjust p-values with Benjamini-Hochberg while preserving NaN positions."""

    try:
        values = np.asarray(list(p_values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_values must contain numeric values.") from exc
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.isinf(values).any():
        raise ValueError("p_values must not contain infinite values.")
    finite_mask = ~np.isnan(values)
    finite = values[finite_mask]
    if ((finite < 0.0) | (finite > 1.0)).any():
        raise ValueError("Finite p-values must lie in [0, 1].")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if finite.size == 0:
        return adjusted

    order = np.argsort(finite, kind="mergesort")
    ranked = finite[order]
    raw_adjusted = ranked * finite.size / np.arange(1, finite.size + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted[finite_mask] = restored
    return adjusted
