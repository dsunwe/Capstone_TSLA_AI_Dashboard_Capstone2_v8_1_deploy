import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CANDIDATE_FEATURES,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)
from src.ml import (
    run_classification,
    walk_forward_evaluate,
    walk_forward_oof_predictions,
    walk_forward_validate,
)


def test_nested_oof_tuning_is_outer_fold_local_and_exportable(engineered_data):
    metrics, predictions, tuning, _, _ = walk_forward_evaluate(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        n_splits=3,
        nested_tuning=True,
        inner_splits=3,
        tuning_iterations=2,
        return_diagnostics=True,
    )
    assert metrics["Fold"].nunique() == 3
    assert predictions["Fold"].nunique() == 3
    assert len(tuning) == 3 * 2
    assert tuning.groupby("Outer Fold").size().eq(2).all()
    assert tuning.groupby("Outer Fold")["Selected"].sum().eq(1).all()
    assert tuning["Inner Splits"].eq(3).all()
    assert tuning["Purge Gap"].eq(TARGET_PURGE_ROWS).all()
    assert tuning["Selection Metric"].eq("Macro F1").all()
    assert metrics["Nested Tuning Protocol"].str.contains("gap=5").all()
    assert predictions["Selected Parameters"].notna().all()
from src.temporal import build_temporal_split_plan, rows_for_dates


def test_ml_uses_chronological_split(engineered_data):
    result = run_classification(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        train_size=0.8,
    )
    assert result is not None
    assert result.train_end_date < result.test_start_date
    assert result.purge_start_date <= result.purge_end_date
    assert result.train_end_date < result.purge_start_date
    assert result.purge_end_date < result.test_start_date
    assert len(result.results) == result.test_rows
    assert TARGET_RETURN_COLUMN in result.results
    assert set(result.results["Signal"].unique()).issubset({0, 1})


def test_ml_result_includes_majority_and_model_comparison(engineered_data):
    train_size = 0.8
    result = run_classification(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        train_size=train_size,
    )
    assert result is not None

    comparison = result.baseline_comparison
    assert comparison["Model"].tolist() == [
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]
    assert list(comparison.columns) == [
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Training Time (s)",
        "Inference Time (s)",
    ]

    plan = build_temporal_split_plan(engineered_data, train_size=train_size)
    assert plan.gap == TARGET_PURGE_ROWS == 5
    assert len(plan.gap_dates) == TARGET_PURGE_ROWS
    y_train = rows_for_dates(
        engineered_data,
        plan.train_dates,
        required_columns=[TARGET_DIRECTION_COLUMN],
    )[TARGET_DIRECTION_COLUMN].astype(int)
    y_test = rows_for_dates(
        engineered_data,
        plan.test_dates,
        required_columns=[TARGET_DIRECTION_COLUMN],
    )[TARGET_DIRECTION_COLUMN].astype(int)
    majority_class = int(y_train.mode().iloc[0])
    prediction = np.full(len(y_test), majority_class, dtype=int)
    probability = np.full(len(y_test), float(y_train.mean()))
    baseline = comparison.loc[comparison["Model"] == "Majority Baseline"].iloc[0]

    assert baseline["Accuracy"] == pytest.approx(accuracy_score(y_test, prediction))
    assert baseline["Precision"] == pytest.approx(
        precision_score(y_test, prediction, zero_division=0)
    )
    assert baseline["Recall"] == pytest.approx(recall_score(y_test, prediction, zero_division=0))
    assert baseline["F1-score"] == pytest.approx(f1_score(y_test, prediction, zero_division=0))
    if y_test.nunique() == 2:
        assert baseline["ROC-AUC"] == pytest.approx(roc_auc_score(y_test, probability))
    else:
        assert np.isnan(baseline["ROC-AUC"])


def test_walk_forward_returns_folds(engineered_data):
    result = walk_forward_validate(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        n_splits=3,
    )
    assert not result.empty
    assert result["Fold"].nunique() == 3


def test_walk_forward_compares_each_fold_with_majority_baseline(engineered_data):
    result = walk_forward_validate(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        n_splits=3,
    )

    assert {
        "Majority Baseline Accuracy",
        "Accuracy Improvement",
        "Majority Baseline F1-score",
        "F1-score Improvement",
    }.issubset(result.columns)
    assert np.allclose(
        result["Accuracy Improvement"],
        result["Accuracy"] - result["Majority Baseline Accuracy"],
    )
    train_end = pd.to_datetime(result["Train End"])
    purge_start = pd.to_datetime(result["Purge Start"])
    purge_end = pd.to_datetime(result["Purge End"])
    test_start = pd.to_datetime(result["Test Start"])
    assert bool((train_end < purge_start).all())
    assert bool((purge_start < purge_end).all())
    assert bool((purge_end < test_start).all())


def test_xgboost_classifier_produces_probabilities(engineered_data):
    result = run_classification(
        engineered_data,
        model_name="XGBoost Classifier",
        train_size=0.8,
        include_benchmark=False,
    )
    assert result is not None
    assert result.results["Predicted Probability"].between(0, 1).all()


def test_training_only_tuning_records_selected_parameters(engineered_data):
    result = run_classification(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        train_size=0.8,
        tune=True,
        tuning_iterations=3,
        include_benchmark=False,
    )
    assert result is not None
    assert result.selected_parameters
    assert not result.tuning_results.empty
    assert result.train_end_date < result.test_start_date


def test_fixed_model_uses_the_explicit_canonical_plan(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    assert plan.gap == TARGET_PURGE_ROWS == 5
    result = run_classification(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        include_benchmark=False,
        split_plan=plan,
    )

    assert result is not None
    assert result.split_plan_id == plan.plan_id
    assert result.split_plan_hash == plan.plan_hash
    assert result.train_end_date == plan.train_end_date
    assert result.purge_start_date == plan.gap_start_date
    assert result.purge_end_date == plan.gap_end_date
    assert result.test_start_date == plan.test_start_date
    assert tuple(result.results["Date"]) == plan.test_dates
    assert TARGET_RETURN_COLUMN in result.results


def test_foldwise_oof_records_training_only_feature_provenance(engineered_data):
    result = walk_forward_oof_predictions(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        n_splits=3,
        foldwise_feature_selection=True,
        candidate_features=CANDIDATE_FEATURES,
        top_k_features=8,
    )

    assert not result.empty
    assert result["Feature Protocol"].eq(
        "Fold-wise training-only MI + correlation selection"
    ).all()
    assert result["Selected Feature Count"].eq(8).all()
    assert result["Feature Set Hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    train_end = pd.to_datetime(result["Train End"])
    purge_start = pd.to_datetime(result["Purge Start"])
    purge_end = pd.to_datetime(result["Purge End"])
    test_date = pd.to_datetime(result["Date"])
    assert (train_end < purge_start).all()
    assert (purge_start < purge_end).all()
    assert (purge_end < test_date).all()
    assert TARGET_RETURN_COLUMN in result
    assert np.array_equal(
        result["Actual Direction"].to_numpy(),
        (result[TARGET_RETURN_COLUMN].to_numpy() > 0).astype(int),
    )
    assert np.array_equal(
        result["Signal"].to_numpy(),
        (
            result["Predicted Probability"].to_numpy()
            >= result["Probability Threshold"].to_numpy()
        ).astype(int),
    )
    assert result.groupby("Fold")["Selected Features"].nunique().eq(1).all()
