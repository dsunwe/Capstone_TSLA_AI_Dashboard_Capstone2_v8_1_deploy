from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from .config import FEATURE_GROUPS, TARGET_DIRECTION_COLUMN
from .temporal import TemporalSplitPlan


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    table: pd.DataFrame
    train_rows: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    gap_start_date: pd.Timestamp | None = None
    gap_end_date: pd.Timestamp | None = None
    test_start_date: pd.Timestamp | None = None
    test_end_date: pd.Timestamp | None = None
    plan_id: str | None = None
    plan_hash: str | None = None
    imputation_strategy: str = "Training-fitted median"

    @property
    def boundary_metadata(self) -> dict[str, object]:
        return {
            "Plan ID": self.plan_id,
            "Plan Hash": self.plan_hash,
            "Train Start": self.train_start_date,
            "Train End": self.train_end_date,
            "Purge Start": self.gap_start_date,
            "Purge End": self.gap_end_date,
            "Test Start": self.test_start_date,
            "Test End": self.test_end_date,
            "Train Rows": self.train_rows,
            "Imputation": self.imputation_strategy,
        }


def _feature_group_lookup() -> dict[str, str]:
    return {
        feature: group
        for group, details in FEATURE_GROUPS.items()
        if group != "Target"
        for feature in details["features"]
    }


def select_features_from_rows(
    training_rows: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    top_k: int = 13,
    correlation_threshold: float = 0.95,
    random_state: int = 42,
    plan: TemporalSplitPlan | None = None,
    minimum_rows: int = 30,
) -> FeatureSelectionResult:
    """Select features using only the explicitly supplied training rows.

    Missing feature values are imputed with medians fitted on these rows. The
    function has no access to future rows and is suitable for outer-fold OOF
    selection as well as the fixed chronological holdout.
    """

    required = {"Date", TARGET_DIRECTION_COLUMN}
    missing_required = sorted(required.difference(training_rows.columns))
    if missing_required:
        raise KeyError(f"Training rows are missing required columns: {missing_required}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1].")

    candidates = list(dict.fromkeys(candidate_features))
    available = [feature for feature in candidates if feature in training_rows.columns]
    if not available:
        raise ValueError("No candidate features are available in the training rows.")

    train = training_rows.copy()
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    train = train.dropna(subset=["Date", TARGET_DIRECTION_COLUMN]).sort_values("Date")
    if train["Date"].duplicated().any():
        raise ValueError("Training rows must have unique dates.")
    if len(train) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} labelled training rows are required for feature selection."
        )
    labels = train[TARGET_DIRECTION_COLUMN].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Feature selection requires both target classes in the training rows.")

    x_train = train[available]
    usable = [feature for feature in available if x_train[feature].notna().any()]
    if not usable:
        raise ValueError("All candidate features are missing throughout the training rows.")

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(x_train[usable]),
        columns=usable,
        index=x_train.index,
    )
    variances = imputed.var(ddof=0)
    non_constant = variances[variances > 1e-12].index.tolist()
    if not non_constant:
        raise ValueError(
            "No non-constant candidate feature remains after training-only imputation."
        )

    mi_values = mutual_info_classif(
        imputed[non_constant],
        labels,
        random_state=random_state,
    )
    mutual_information = pd.Series(mi_values, index=non_constant).sort_values(
        ascending=False,
        kind="mergesort",
    )
    correlation = imputed[non_constant].corr().abs()
    kept: list[str] = []
    removed_for_correlation: set[str] = set()
    for feature in mutual_information.index:
        if any(correlation.loc[feature, existing] >= correlation_threshold for existing in kept):
            removed_for_correlation.add(feature)
        else:
            kept.append(feature)
    selected = kept[: min(top_k, len(kept))]

    group_lookup = _feature_group_lookup()
    table = pd.DataFrame(
        {
            "Feature": available,
            "Feature Group": [group_lookup.get(feature, "Other") for feature in available],
            "Training Missing %": [
                float(x_train[feature].isna().mean() * 100) for feature in available
            ],
            "Training Variance": [
                float(variances.get(feature, 0.0)) for feature in available
            ],
            "Mutual Information": [
                float(mutual_information.get(feature, 0.0)) for feature in available
            ],
            "Correlation Filter": [
                (
                    "All missing"
                    if feature not in usable
                    else (
                        "Constant"
                        if feature not in non_constant
                        else ("Removed" if feature in removed_for_correlation else "Kept")
                    )
                )
                for feature in available
            ],
            "Selected": [feature in selected for feature in available],
        }
    )
    table["MI Rank"] = (
        table["Mutual Information"].rank(method="min", ascending=False).astype(int)
    )

    train_start = pd.Timestamp(train["Date"].iloc[0])
    train_end = pd.Timestamp(train["Date"].iloc[-1])
    metadata = {
        "Plan ID": plan.plan_id if plan is not None else None,
        "Plan Hash": plan.plan_hash if plan is not None else None,
        "Train Start": train_start,
        "Train End": train_end,
        "Purge Start": plan.gap_start_date if plan is not None else None,
        "Purge End": plan.gap_end_date if plan is not None else None,
        "Test Start": plan.test_start_date if plan is not None else None,
        "Test End": plan.test_end_date if plan is not None else None,
        "Training Imputation": "Median fitted on supplied training rows",
    }
    for column, value in metadata.items():
        table[column] = value
    table = table.sort_values(
        ["Selected", "Mutual Information", "Feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return FeatureSelectionResult(
        selected_features=selected,
        table=table,
        train_rows=len(train),
        train_start_date=train_start,
        train_end_date=train_end,
        gap_start_date=plan.gap_start_date if plan is not None else None,
        gap_end_date=plan.gap_end_date if plan is not None else None,
        test_start_date=plan.test_start_date if plan is not None else None,
        test_end_date=plan.test_end_date if plan is not None else None,
        plan_id=plan.plan_id if plan is not None else None,
        plan_hash=plan.plan_hash if plan is not None else None,
    )
