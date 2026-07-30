from types import SimpleNamespace

import pandas as pd

from src.config import (
    CANDIDATE_FEATURES,
    PRESPECIFIED_OOF_FEATURES,
    TARGET_DIRECTION_COLUMN,
)
from src.ml import walk_forward_oof_predictions
from src.research import (
    run_feature_family_ablation,
    run_feature_group_ablation,
    run_multi_asset_benchmark,
    prespecified_feature_evidence_from_training,
    select_features_from_training,
)
from src.selection import select_features_from_rows
from src.temporal import build_temporal_split_plan, rows_for_dates


def test_feature_selection_uses_training_period_and_returns_requested_count(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    result = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    assert len(result.selected_features) == 10
    assert result.train_end_date == plan.train_end_date
    assert result.train_end_date < result.test_start_date
    assert result.plan_id == plan.plan_id
    assert result.plan_hash == plan.plan_hash
    assert result.train_rows == len(plan.train_dates)
    assert result.table["Selected"].sum() == 10
    assert {"Mutual Information", "Correlation Filter", "MI Rank"}.issubset(
        result.table.columns
    )
    assert result.table["Plan ID"].nunique() == 1
    assert result.table["Test Start"].nunique() == 1


def test_selection_is_invariant_to_holdout_feature_and_label_changes(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    baseline = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    changed = engineered_data.copy()
    test_mask = changed["Date"].isin(plan.test_dates)
    changed.loc[test_mask, CANDIDATE_FEATURES] = 1_000_000
    changed.loc[test_mask, TARGET_DIRECTION_COLUMN] = (
        1 - changed.loc[test_mask, TARGET_DIRECTION_COLUMN]
    )

    repeated = select_features_from_training(
        changed,
        top_k=10,
        split_plan=plan,
    )

    assert repeated.selected_features == baseline.selected_features
    pd.testing.assert_series_equal(
        repeated.table.set_index("Feature")["Mutual Information"],
        baseline.table.set_index("Feature")["Mutual Information"],
    )


def test_selection_keeps_exact_training_dates_and_fits_missing_value_imputation_there(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    changed = engineered_data.copy()
    missing_dates = plan.train_dates[10:30]
    changed.loc[changed["Date"].isin(missing_dates), "Momentum_5"] = pd.NA

    result = select_features_from_training(
        changed,
        candidate_features=[
            "Lag_Return_1",
            "Lag_Return_3",
            "Momentum_5",
            "Volatility_10",
            "Volume_Ratio",
        ],
        top_k=3,
        split_plan=plan,
    )
    momentum = result.table.set_index("Feature").loc["Momentum_5"]

    assert result.train_rows == len(plan.train_dates)
    assert momentum["Training Missing %"] > 0
    assert result.imputation_strategy == "Training-fitted median"


def test_prespecified_rq2_evidence_keeps_exact_six_inputs_without_top_k_selection(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    result = prespecified_feature_evidence_from_training(
        engineered_data,
        split_plan=plan,
    )

    assert result.selected_features == list(PRESPECIFIED_OOF_FEATURES)
    assert result.table["Feature"].tolist() == list(PRESPECIFIED_OOF_FEATURES)
    assert len(result.table) == 6
    assert result.table["Selected"].all()
    assert result.table["Feature Group"].eq(
        "Technical — EMA-aware compact"
    ).all()
    assert "Mutual Information" not in result.table
    assert "MI Rank" not in result.table
    assert "Correlation Filter" not in result.table
    assert result.plan_id == plan.plan_id
    assert result.train_end_date < result.test_start_date


def test_select_features_from_rows_is_pure_training_row_interface(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    training_rows = rows_for_dates(engineered_data, plan.train_dates)

    result = select_features_from_rows(
        training_rows,
        CANDIDATE_FEATURES,
        top_k=7,
        plan=plan,
    )

    assert len(result.selected_features) == 7
    assert result.train_start_date == plan.train_start_date
    assert result.train_end_date == plan.train_end_date
    assert result.test_start_date == plan.test_start_date


def _dummy_classification_result():
    return SimpleNamespace(
        metrics={
            "Accuracy": 0.50,
            "Balanced Accuracy": 0.50,
            "Precision": 0.50,
            "Recall": 0.50,
            "F1-score": 0.50,
            "Macro F1": 0.50,
            "MCC": 0.0,
            "ROC-AUC": 0.50,
            "PR-AUC": 0.50,
            "Brier Score": 0.25,
        },
        selected_parameters={},
        tuning_results=pd.DataFrame(),
        training_time_seconds=0.0,
        inference_time_seconds=0.0,
    )


def test_ablation_includes_raw_all_and_training_selected_on_one_plan(
    engineered_data,
    monkeypatch,
):
    plan = build_temporal_split_plan(engineered_data)
    selection = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    received_plans = []

    def fake_run_classification(*args, **kwargs):
        received_plans.append(kwargs["split_plan"])
        return _dummy_classification_result()

    monkeypatch.setattr("src.research.run_classification", fake_run_classification)
    result = run_feature_group_ablation(
        engineered_data,
        selection.selected_features,
        model_name="Logistic Regression (Baseline)",
        split_plan=plan,
    )
    assert {
        "Raw OHLCV only",
        "All candidate features",
        "Training-selected features",
    }.issubset(set(result["Experiment"]))
    assert {"Balanced Accuracy", "Macro F1", "ROC-AUC"}.issubset(result.columns)
    assert all(received is plan for received in received_plans)
    assert result["Plan ID"].nunique() == 1
    assert result["Test Start"].nunique() == 1
    assert result["Test Rows"].nunique() == 1
    assert result.iloc[0]["Test Start"] == plan.test_start_date


def test_family_ablation_passes_one_plan_to_every_feature_subset(
    engineered_data,
    monkeypatch,
):
    plan = build_temporal_split_plan(engineered_data)
    received = []

    def fake_run_classification(*args, **kwargs):
        received.append(kwargs["split_plan"])
        return _dummy_classification_result()

    monkeypatch.setattr("src.research.run_classification", fake_run_classification)
    result = run_feature_family_ablation(engineered_data, split_plan=plan)

    assert len(result) == 5
    assert all(item is plan for item in received)
    assert result["Plan Hash"].eq(plan.plan_hash).all()
    assert result["Train End"].eq(plan.train_end_date).all()
    assert result["Test Start"].eq(plan.test_start_date).all()


def test_multi_asset_benchmark_shares_one_common_plan(
    engineered_data,
    monkeypatch,
):
    assets = {
        "TSLA": engineered_data.copy(),
        "AAPL": engineered_data.copy(),
        "NVDA": engineered_data.copy(),
    }
    plan = build_temporal_split_plan(assets)
    received = []

    def fake_run_classification(*args, **kwargs):
        received.append(kwargs["split_plan"])
        return _dummy_classification_result()

    monkeypatch.setattr("src.research.run_classification", fake_run_classification)
    result = run_multi_asset_benchmark(
        assets,
        top_k_features=5,
        tune=False,
        split_plan=plan,
    )

    assert all(item is plan for item in received)
    assert result.benchmark["Plan ID"].nunique() == 1
    assert result.benchmark["Test Start"].nunique() == 1
    assert result.feature_selection["Plan ID"].nunique() == 1
    assert result.feature_selection["Asset"].nunique() == 3


def test_multi_asset_benchmark_can_hold_the_rq2_six_feature_scope_fixed(
    engineered_data,
    monkeypatch,
):
    assets = {
        "TSLA": engineered_data.copy(),
        "AAPL": engineered_data.copy(),
        "NVDA": engineered_data.copy(),
    }
    plan = build_temporal_split_plan(assets)

    monkeypatch.setattr(
        "src.research.run_classification",
        lambda *args, **kwargs: _dummy_classification_result(),
    )
    result = run_multi_asset_benchmark(
        assets,
        tune=False,
        split_plan=plan,
        prespecified_features=list(PRESPECIFIED_OOF_FEATURES),
    )

    for asset in assets:
        rows = result.feature_selection.loc[
            result.feature_selection["Asset"].eq(asset)
        ]
        assert rows["Feature"].tolist() == list(PRESPECIFIED_OOF_FEATURES)
        assert rows["Selected"].all()


def test_walk_forward_oof_predictions_are_unique_and_fold_labeled(engineered_data):
    result = walk_forward_oof_predictions(
        engineered_data,
        model_name="Logistic Regression (Baseline)",
        n_splits=3,
    )
    assert not result.empty
    assert result["Date"].is_unique
    assert result["Fold"].nunique() == 3
    assert result["Predicted Probability"].between(0, 1).all()
