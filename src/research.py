from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CANDIDATE_FEATURES,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_LOOKBACK,
    DEFAULT_LSTM_SEEDS,
    DEFAULT_LSTM_UNITS,
    FEATURE_GROUPS,
    LSTM_MODEL_NAME,
    PRESPECIFIED_OOF_FEATURES,
    TARGET_DIRECTION_COLUMN,
)
from .context import build_context_frame
from .data import align_datasets_to_common_window, load_builtin_datasets
from .features import add_features
from .ml import _classification_metrics, run_classification
from .selection import FeatureSelectionResult, select_features_from_rows
from .temporal import (
    DEFAULT_MAX_LOOKBACK,
    TemporalSplitPlan,
    build_temporal_split_plan,
    rows_for_dates,
)


@dataclass
class MultiAssetResult:
    benchmark: pd.DataFrame
    summary: pd.DataFrame
    feature_stability: pd.DataFrame
    feature_selection: pd.DataFrame
    tuning_results: pd.DataFrame
    dataset_summary: pd.DataFrame


def select_features_from_training(
    engineered: pd.DataFrame,
    *,
    candidate_features: list[str] | None = None,
    train_size: float = 0.80,
    top_k: int = 13,
    correlation_threshold: float = 0.95,
    random_state: int = 42,
    split_plan: TemporalSplitPlan | None = None,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
) -> FeatureSelectionResult:
    """Select inputs only from the canonical plan's chronological training rows."""

    candidates = list(candidate_features or CANDIDATE_FEATURES)
    available = [feature for feature in candidates if feature in engineered.columns]
    if not available:
        raise ValueError("No candidate features are available in the engineered data.")
    plan = split_plan or build_temporal_split_plan(
        engineered,
        train_size=train_size,
        max_lookback=max_lookback,
    )
    training_rows = rows_for_dates(
        engineered,
        plan.train_dates,
        required_columns=[*available, TARGET_DIRECTION_COLUMN],
    )
    return select_features_from_rows(
        training_rows,
        candidates,
        top_k=top_k,
        correlation_threshold=correlation_threshold,
        random_state=random_state,
        plan=plan,
    )


def prespecified_feature_evidence_from_training(
    engineered: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    train_size: float = 0.80,
    split_plan: TemporalSplitPlan | None = None,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
) -> FeatureSelectionResult:
    """Audit the frozen RQ2 technical inputs without re-selecting them.

    The revised RQ2 compares pre-specified standalone families under one common
    Logistic Regression.  Its downstream Technical specification therefore
    must not be reconstructed with the legacy MI/correlation/Top-k selector.
    This function keeps the six protocol-ordered channels fixed and uses the
    training period only to report availability and variance evidence.
    """

    features = list(dict.fromkeys(feature_names or PRESPECIFIED_OOF_FEATURES))
    missing = [feature for feature in features if feature not in engineered.columns]
    if missing:
        raise ValueError(
            "The frozen RQ2 technical inputs are missing from the engineered data: "
            + ", ".join(missing)
        )
    plan = split_plan or build_temporal_split_plan(
        engineered,
        train_size=train_size,
        max_lookback=max_lookback,
    )
    training_rows = rows_for_dates(
        engineered,
        plan.train_dates,
        required_columns=[*features, TARGET_DIRECTION_COLUMN],
    ).copy()
    training_rows["Date"] = pd.to_datetime(training_rows["Date"], errors="coerce")
    training_rows = (
        training_rows.dropna(subset=["Date", TARGET_DIRECTION_COLUMN])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if training_rows.empty:
        raise ValueError("No labelled training rows are available for RQ2 input auditing.")

    x_train = training_rows[features]
    medians = x_train.median(axis=0, skipna=True)
    imputed = x_train.fillna(medians)
    if imputed.isna().all(axis=0).any():
        unavailable = imputed.columns[imputed.isna().all(axis=0)].tolist()
        raise ValueError(
            "Frozen RQ2 inputs are entirely missing in the training period: "
            + ", ".join(unavailable)
        )

    table = pd.DataFrame(
        {
            "Feature": features,
            "Feature Group": "Technical — EMA-aware compact",
            "Training Missing %": [
                float(x_train[feature].isna().mean() * 100) for feature in features
            ],
            "Training Variance": [
                float(imputed[feature].var(ddof=0)) for feature in features
            ],
            "Selection Basis": "Pre-specified in revised RQ2; no MI or Top-k filtering",
            "Selected": True,
        }
    )
    metadata = {
        "Plan ID": plan.plan_id,
        "Plan Hash": plan.plan_hash,
        "Train Start": pd.Timestamp(training_rows["Date"].iloc[0]),
        "Train End": pd.Timestamp(training_rows["Date"].iloc[-1]),
        "Purge Start": plan.gap_start_date,
        "Purge End": plan.gap_end_date,
        "Test Start": plan.test_start_date,
        "Test End": plan.test_end_date,
        "Training Imputation": "Median fitted on supplied training rows",
    }
    for column, value in metadata.items():
        table[column] = value

    return FeatureSelectionResult(
        selected_features=features,
        table=table,
        train_rows=len(training_rows),
        train_start_date=pd.Timestamp(training_rows["Date"].iloc[0]),
        train_end_date=pd.Timestamp(training_rows["Date"].iloc[-1]),
        gap_start_date=plan.gap_start_date,
        gap_end_date=plan.gap_end_date,
        test_start_date=plan.test_start_date,
        test_end_date=plan.test_end_date,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        imputation_strategy="Training-fitted median",
    )


def _resolve_plan(
    engineered: pd.DataFrame,
    split_plan: TemporalSplitPlan | None,
) -> TemporalSplitPlan:
    return split_plan or build_temporal_split_plan(engineered)


def _result_with_plan_metadata(
    result: Any,
    plan: TemporalSplitPlan,
    *,
    experiment: str,
    feature_count: int,
    features: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Experiment": experiment,
        "Feature Count": feature_count,
    }
    if features is not None:
        row["Features"] = ", ".join(features)
    row.update(plan.boundary_metadata())
    row.update(result.metrics)
    return row


def run_feature_group_ablation(
    engineered: pd.DataFrame,
    selected_features: list[str],
    *,
    model_name: str = "Logistic Regression (Baseline)",
    probability_threshold: float = 0.50,
    split_plan: TemporalSplitPlan | None = None,
) -> pd.DataFrame:
    """Compare raw, single-group, all-candidate, and selected-feature pipelines."""

    plan = _resolve_plan(engineered, split_plan)
    experiments: list[tuple[str, list[str]]] = [
        ("Raw OHLCV only", ["Open", "High", "Low", "Close", "Volume"]),
    ]
    for group, details in FEATURE_GROUPS.items():
        if group == "Target":
            continue
        available = [feature for feature in details["features"] if feature in engineered.columns]
        if available:
            experiments.append((f"{group} only", available))
    experiments.extend(
        [
            ("All candidate features", [f for f in CANDIDATE_FEATURES if f in engineered.columns]),
            ("Training-selected features", selected_features),
        ]
    )
    rows: list[dict[str, Any]] = []
    for experiment_name, features in experiments:
        result = run_classification(
            engineered,
            model_name=model_name,
            probability_threshold=probability_threshold,
            feature_names=features,
            include_benchmark=False,
            split_plan=plan,
        )
        if result is None:
            continue
        rows.append(
            _result_with_plan_metadata(
                result,
                plan,
                experiment=experiment_name,
                feature_count=len(features),
                features=features,
            )
        )
    return pd.DataFrame(rows)


def run_feature_family_ablation(
    engineered: pd.DataFrame,
    *,
    model_name: str = "Logistic Regression (Baseline)",
    probability_threshold: float = 0.50,
    split_plan: TemporalSplitPlan | None = None,
) -> pd.DataFrame:
    """Test whether each examiner-requested external feature family adds value."""

    plan = _resolve_plan(engineered, split_plan)
    proxy_group = "Market-implied risk and survey-based sentiment proxies"
    context_groups = {"Market context", "Macroeconomic context", proxy_group}
    technical = [
        feature
        for group, details in FEATURE_GROUPS.items()
        if group not in context_groups | {"Target"}
        for feature in details["features"]
        if feature in engineered.columns
    ]
    market = [
        feature
        for feature in FEATURE_GROUPS["Market context"]["features"]
        if feature in engineered.columns
    ]
    macro = [
        feature
        for feature in FEATURE_GROUPS["Macroeconomic context"]["features"]
        if feature in engineered.columns
    ]
    sentiment = [
        feature
        for feature in FEATURE_GROUPS[proxy_group]["features"]
        if feature in engineered.columns
    ]
    experiments = [
        ("A. Technical only", technical),
        ("B. Technical + market", [*technical, *market]),
        ("C. Technical + macro", [*technical, *macro]),
        ("D. Technical + sentiment", [*technical, *sentiment]),
        ("E. All feature families", [*technical, *market, *macro, *sentiment]),
    ]
    rows: list[dict[str, Any]] = []
    for experiment_name, features in experiments:
        deduplicated_features = list(dict.fromkeys(features))
        result = run_classification(
            engineered,
            model_name=model_name,
            probability_threshold=probability_threshold,
            feature_names=deduplicated_features,
            include_benchmark=False,
            split_plan=plan,
        )
        if result is None:
            continue
        rows.append(
            _result_with_plan_metadata(
                result,
                plan,
                experiment=experiment_name,
                feature_count=len(deduplicated_features),
            )
        )
    return pd.DataFrame(rows)


def prepare_multi_asset_engineered() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cleaned, original_summary = load_builtin_datasets()
    aligned, common_start, common_end = align_datasets_to_common_window(cleaned)
    common_dates = next(iter(aligned.values()))["Date"]
    context = build_context_frame(common_dates).frame
    engineered = {
        asset: add_features(frame, context_data=context)
        for asset, frame in aligned.items()
    }
    summary = original_summary.copy()
    aligned_rows = {asset: len(frame) for asset, frame in aligned.items()}
    summary["Common Start"] = common_start
    summary["Common End"] = common_end
    summary["Aligned Rows"] = summary["Asset"].map(aligned_rows)
    summary["Data Source"] = "Nasdaq historical snapshot (private academic evaluation)"
    summary["Context Sources"] = "QQQ, S&P 500, Nasdaq Composite, FRED macro/sentiment"
    return engineered, summary


def _majority_row(
    engineered: pd.DataFrame,
    *,
    split_plan: TemporalSplitPlan,
) -> dict[str, float]:
    train = rows_for_dates(
        engineered,
        split_plan.train_dates,
        required_columns=[TARGET_DIRECTION_COLUMN],
    )
    test = rows_for_dates(
        engineered,
        split_plan.test_dates,
        required_columns=[TARGET_DIRECTION_COLUMN],
    )
    y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
    y_test = test[TARGET_DIRECTION_COLUMN].astype(int)
    majority_class = int(y_train.mode().iloc[0])
    prediction = np.full(len(y_test), majority_class, dtype=int)
    probability = np.full(len(y_test), float(y_train.mean()))
    return _classification_metrics(y_test, prediction, probability)


def run_multi_asset_benchmark(
    engineered_by_asset: dict[str, pd.DataFrame] | None = None,
    *,
    train_size: float = 0.80,
    probability_threshold: float = 0.50,
    top_k_features: int = 13,
    tune: bool = True,
    tuning_iterations: int = 4,
    include_lstm: bool = False,
    lstm_lookback: int = DEFAULT_LSTM_LOOKBACK,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    lstm_epochs: int = DEFAULT_LSTM_EPOCHS,
    lstm_seeds: tuple[int, ...] = tuple(DEFAULT_LSTM_SEEDS),
    split_plan: TemporalSplitPlan | None = None,
    prespecified_features: list[str] | None = None,
) -> MultiAssetResult:
    if engineered_by_asset is None:
        engineered_by_asset, dataset_summary = prepare_multi_asset_engineered()
    else:
        dataset_summary = pd.DataFrame(
            [
                {
                    "Asset": asset,
                    "Rows": len(frame),
                    "Start Date": frame["Date"].min(),
                    "End Date": frame["Date"].max(),
                }
                for asset, frame in engineered_by_asset.items()
            ]
        )

    plan = split_plan or build_temporal_split_plan(engineered_by_asset)
    for asset, frame in engineered_by_asset.items():
        # Fail closed if any asset cannot cover the exact common plan.
        rows_for_dates(
            frame,
            plan.eligible_dates,
            required_columns=[TARGET_DIRECTION_COLUMN],
        )

    benchmark_rows: list[dict[str, Any]] = []
    selection_tables: list[pd.DataFrame] = []
    tuning_tables: list[pd.DataFrame] = []
    for asset, engineered in engineered_by_asset.items():
        if prespecified_features is None:
            selection = select_features_from_training(
                engineered,
                train_size=train_size,
                top_k=top_k_features,
                split_plan=plan,
            )
        else:
            selection = prespecified_feature_evidence_from_training(
                engineered,
                feature_names=prespecified_features,
                train_size=train_size,
                split_plan=plan,
            )
        selection_table = selection.table.copy()
        selection_table.insert(0, "Asset", asset)
        selection_tables.append(selection_table)
        features = selection.selected_features
        baseline_metrics = _majority_row(engineered, split_plan=plan)
        benchmark_rows.append(
            {
                "Asset": asset,
                "Model": "Majority Baseline",
                "Tuned": "No",
                "Selected Feature Count": len(features),
                "Selected Parameters": "Training-set majority class",
                **plan.boundary_metadata(),
                **baseline_metrics,
                "Training Time (s)": 0.0,
                "Inference Time (s)": 0.0,
            }
        )
        model_specs = [
            ("Logistic Regression (Baseline)", "Logistic Regression"),
            ("Random Forest Classifier", "Random Forest"),
            ("XGBoost Classifier", "XGBoost"),
        ]
        if include_lstm:
            model_specs.append((LSTM_MODEL_NAME, "LSTM"))
        for model_name, display_name in model_specs:
            result = run_classification(
                engineered,
                model_name=model_name,
                train_size=train_size,
                probability_threshold=probability_threshold,
                feature_names=features,
                tune=tune,
                tuning_iterations=tuning_iterations,
                include_benchmark=False,
                lstm_lookback=lstm_lookback,
                lstm_units=lstm_units,
                lstm_epochs=lstm_epochs,
                lstm_seeds=lstm_seeds,
                split_plan=plan,
            )
            if result is None:
                continue
            benchmark_rows.append(
                {
                    "Asset": asset,
                    "Model": display_name,
                    "Tuned": (
                        "Chronological early stopping"
                        if model_name == LSTM_MODEL_NAME
                        else ("Yes" if tune else "No")
                    ),
                    "Selected Feature Count": len(features),
                    "Selected Parameters": str(result.selected_parameters or "Default"),
                    **plan.boundary_metadata(),
                    **result.metrics,
                    "Training Time (s)": result.training_time_seconds,
                    "Inference Time (s)": result.inference_time_seconds,
                }
            )
            if not result.tuning_results.empty:
                tuning_table = result.tuning_results.copy()
                tuning_table.insert(0, "Model", display_name)
                tuning_table.insert(0, "Asset", asset)
                tuning_tables.append(tuning_table)

    benchmark = pd.DataFrame(benchmark_rows)
    summary = (
        benchmark.groupby("Model", as_index=False)
        .agg(
            Assets=("Asset", "nunique"),
            Mean_Accuracy=("Accuracy", "mean"),
            Std_Accuracy=("Accuracy", "std"),
            Mean_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            Mean_Macro_F1=("Macro F1", "mean"),
            Std_Macro_F1=("Macro F1", "std"),
            Mean_ROC_AUC=("ROC-AUC", "mean"),
            Total_Training_Time=("Training Time (s)", "sum"),
        )
        .sort_values("Mean_Macro_F1", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(
        0,
        "Rank by Mean Macro-F1",
        np.arange(1, len(summary) + 1),
    )
    all_selection = pd.concat(selection_tables, ignore_index=True)
    selected_only = all_selection[all_selection["Selected"]].copy()
    stability = (
        selected_only.groupby(["Feature", "Feature Group"], as_index=False)
        .agg(
            Selected_In_Assets=("Asset", "nunique"),
            Assets=("Asset", lambda x: ", ".join(sorted(x))),
        )
        .sort_values(["Selected_In_Assets", "Feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    stability["Selection Rate"] = stability["Selected_In_Assets"] / max(
        len(engineered_by_asset), 1
    )
    tuning_results = (
        pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame()
    )
    return MultiAssetResult(
        benchmark=benchmark,
        summary=summary,
        feature_stability=stability,
        feature_selection=all_selection,
        tuning_results=tuning_results,
        dataset_summary=dataset_summary,
    )
