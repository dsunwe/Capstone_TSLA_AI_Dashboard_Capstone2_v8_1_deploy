from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd

import src.export as export_module
from src.config import PRESPECIFIED_OOF_FEATURES
from src.experiment import build_experiment_configuration
from src.research import select_features_from_training
from src.temporal import build_temporal_split_plan
from src.validation import validate_temporal_plan_alignment
from src.v8_evidence import (
    ASSETS,
    BASELINES,
    MODELS,
    PERIODS,
    RQ3_ROOT,
    RQ4_ROOT,
    load_rq3_selection,
    load_rq4_headline_metrics,
    load_rq4_model_results,
    read_csv,
    validation_summary,
)


def _model_result_for_plan(plan, **overrides):
    values = {
        "split_plan_id": plan.plan_id,
        "split_plan_hash": plan.plan_hash,
        "train_start_date": plan.train_start_date,
        "train_end_date": plan.train_end_date,
        "purge_start_date": plan.gap_start_date,
        "purge_end_date": plan.gap_end_date,
        "test_start_date": plan.test_start_date,
        "test_end_date": plan.test_end_date,
        "train_rows": len(plan.train_dates),
        "test_rows": len(plan.test_dates),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_temporal_validation_accepts_one_exact_selection_and_model_plan(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    selection = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    model_result = _model_result_for_plan(plan)

    report = validate_temporal_plan_alignment(selection, model_result, plan)

    assert report["Test ID"].tolist() == ["T01", "T02", "T03", "T04"]
    assert report["Status"].eq("Pass").all()


def test_temporal_validation_rejects_plan_identifier_mismatch(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    selection = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    mismatched_model = _model_result_for_plan(
        plan,
        split_plan_id="temporal-v1-not-the-common-plan",
    )

    report = validate_temporal_plan_alignment(selection, mismatched_model, plan)
    status = report.set_index("Test ID")["Status"]

    assert status["T01"] == "Fail"
    assert status["T02"] == "Pass"
    assert status["T03"] == "Pass"


def test_temporal_validation_rejects_selection_or_model_crossing_boundaries(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    selection = select_features_from_training(
        engineered_data,
        top_k=10,
        split_plan=plan,
    )
    crossing_selection = replace(
        selection,
        train_end_date=plan.test_start_date,
    )
    crossing_model = _model_result_for_plan(
        plan,
        train_end_date=plan.gap_start_date,
        test_start_date=plan.gap_end_date,
    )

    report = validate_temporal_plan_alignment(
        crossing_selection,
        crossing_model,
        plan,
    )
    status = report.set_index("Test ID")["Status"]

    assert status["T01"] == "Pass"
    assert status["T02"] == "Fail"
    assert status["T03"] == "Fail"


def test_experiment_configuration_records_v8_temporal_and_decision_contract(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    settings = {
        "threshold": 0.50,
        "n_estimators": 200,
        "max_depth": 6,
        "tune_model": True,
        "lstm_lookback": 20,
        "lstm_units": 16,
        "lstm_dropout": 0.2,
        "lstm_epochs": 40,
        "lstm_batch_size": 32,
        "cv_splits": 5,
        "transaction_cost_bps": 10,
        "slippage_bps": 5,
    }

    configuration = build_experiment_configuration(
        engineered_data,
        source_name="synthetic_test.csv",
        model_name="LSTM Classifier",
        settings=settings,
        selected_features=list(PRESPECIFIED_OOF_FEATURES),
        split_plan=plan,
    )
    values = configuration.set_index("Setting")["Value"].to_dict()

    assert values["Release"] == "8.1"
    assert (
        values["Release name"]
        == "Examiner-Ready Literature and Lean Dashboard"
    )
    assert values["Common temporal plan ID"] == plan.plan_id
    assert values["Common temporal plan SHA-256"] == plan.plan_hash
    assert (
        values["Expanding-OOF feature protocol"]
        == "Same six RQ2 inputs in every outer fold"
    )
    assert values["Expanding-OOF candidate features"] == ", ".join(
        PRESPECIFIED_OOF_FEATURES
    )
    assert (
        values["Prediction target"]
        == "1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])] using information "
        "available through Close[t]"
    )
    assert values["Boundary purge gap"] == "5 trading-session anchors"
    assert (
        values["Main-study uncertainty protocol"]
        == "Descriptive downstream evaluation; no statistical-significance claim"
    )
    assert values["OOF LSTM ensemble rule"].startswith("Arithmetic mean")
    assert values["Backtest signal time"] == "After day-t close"
    assert values["Backtest execution time"] == "Next trading-session open"
    assert (
        values["Backtest valuation return"]
        == "Five staggered 20% sleeves; each sleeve holds from Open[t+1] "
        "to its corresponding Open[t+6]"
    )


def test_save_project_outputs_writes_temporal_execution_and_ledger_artifacts(
    tmp_path,
    monkeypatch,
):
    destinations = {
        "FEATURES_DATA_PATH": tmp_path / "features.csv",
        "TEMPORAL_SPLIT_MANIFEST_PATH": tmp_path / "temporal_manifest.csv",
        "HELDOUT_BACKTEST_EXECUTION_PATH": tmp_path / "fixed_execution.csv",
        "OOF_BACKTEST_EXECUTION_PATH": tmp_path / "oof_execution.csv",
        "HELDOUT_TRADE_LEDGER_PATH": tmp_path / "fixed_ledger.csv",
        "OOF_TRADE_LEDGER_PATH": tmp_path / "oof_ledger.csv",
    }
    for attribute, path in destinations.items():
        monkeypatch.setattr(export_module, attribute, path)

    engineered = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05"],
            "Price": [100.0, 101.0],
        }
    )
    temporal_manifest = pd.DataFrame(
        {
            "Plan ID": ["temporal-v1-test"] * 3,
            "Partition": ["Train", "Purge", "Test"],
            "Date": ["2025-12-31", "2026-01-01", "2026-01-02"],
        }
    )
    fixed_execution = pd.DataFrame(
        {
            "Signal Date": ["2026-01-02"],
            "Execution Date": ["2026-01-05"],
            "Valuation Date": ["2026-01-12"],
        }
    )
    oof_execution = fixed_execution.assign(Fold=1)
    fixed_ledger = pd.DataFrame(
        {
            "Trade": [1],
            "Entry Execution Date": ["2026-01-05"],
            "Status": ["Closed"],
        }
    )
    oof_ledger = fixed_ledger.assign(Fold=1)

    saved = export_module.save_project_outputs(
        engineered=engineered,
        ml_result=None,
        backtest_metrics=pd.DataFrame(),
        backtest_equity=pd.DataFrame(),
        sensitivity=pd.DataFrame(),
        validation_report=pd.DataFrame(),
        temporal_split_manifest=temporal_manifest,
        heldout_backtest_execution=fixed_execution,
        oof_backtest_execution=oof_execution,
        heldout_trade_ledger=fixed_ledger,
        oof_trade_ledger=oof_ledger,
    )

    assert set(saved) == set(destinations.values())
    assert all(path.exists() for path in destinations.values())
    pd.testing.assert_frame_equal(
        pd.read_csv(destinations["TEMPORAL_SPLIT_MANIFEST_PATH"]),
        temporal_manifest,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(destinations["HELDOUT_BACKTEST_EXECUTION_PATH"]),
        fixed_execution,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(destinations["OOF_BACKTEST_EXECUTION_PATH"]),
        oof_execution,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(destinations["HELDOUT_TRADE_LEDGER_PATH"]),
        fixed_ledger,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(destinations["OOF_TRADE_LEDGER_PATH"]),
        oof_ledger,
    )


def test_v8_formal_validation_ledgers_have_84_saved_passes():
    summary = validation_summary()
    table = summary["table"].set_index("Module")

    assert summary["all_passed"] is True
    assert summary["passed"] == summary["total"] == 84
    assert table.loc["RQ2", ["Passed", "Total"]].tolist() == [22, 22]
    assert table.loc["RQ3", ["Passed", "Total"]].tolist() == [24, 24]
    assert table.loc["RQ4", ["Passed", "Total"]].tolist() == [38, 38]
    assert table["Status"].eq("PASS").all()


def test_rq4_links_three_assets_four_models_and_three_transparent_baselines():
    model_results = load_rq4_model_results()
    headline = load_rq4_headline_metrics()

    assert len(model_results) == len(ASSETS) * len(PERIODS) * len(MODELS)
    assert set(model_results["Asset"]) == set(ASSETS)
    assert set(model_results["Period"]) == set(PERIODS)
    assert set(model_results["Model"]) == set(MODELS)
    assert not model_results.duplicated(["Asset", "Period", "Model"]).any()
    assert (
        model_results.groupby(["Asset", "Period"])["Model"]
        .nunique()
        .eq(len(MODELS))
        .all()
    )

    expected_strategies = set(MODELS) | set(BASELINES)
    assert len(headline) == len(ASSETS) * len(PERIODS) * len(expected_strategies)
    assert set(headline["Strategy"]) == expected_strategies
    assert headline.groupby(["Asset", "Period"])["Strategy"].nunique().eq(7).all()


def test_rq4_signal_ledger_records_t_plus_1_execution_and_t_plus_6_end():
    ledger = read_csv(RQ4_ROOT / "outputs" / "rq4_signal_ledger.csv")
    executable = ledger.loc[ledger["Five-Session Executable"].astype(bool)].copy()

    assert not executable.empty
    assert (
        executable["Execution Price Row"] - executable["Signal Price Row"]
    ).eq(1).all()
    assert (
        executable["End Price Row"] - executable["Signal Price Row"]
    ).eq(6).all()
    assert pd.to_datetime(executable["Date"]).lt(
        pd.to_datetime(executable["Target_Date"])
    ).all()
    assert pd.to_datetime(executable["Execution Date"]).lt(
        pd.to_datetime(executable["Scheduled End Date"])
    ).all()


def test_rq3_selection_excludes_audit_and_rq4_returns():
    selection = load_rq3_selection()
    selected = selection.loc[selection["Selected for RQ4"].astype(bool)]

    assert len(selection) == len(MODELS) == 4
    assert len(selected) == 1
    assert selected.iloc[0]["Model"] == "Lagged Logistic Regression"
    assert selection["Selection Data"].eq(
        "Development outer-fold OOF only"
    ).all()
    assert not selection["Audit Used"].astype(bool).any()
    assert not selection["RQ4 Returns Used"].astype(bool).any()


def test_rq4_is_descriptive_without_bootstrap_or_rolling_retraining():
    protocol = json.loads((RQ4_ROOT / "protocol.json").read_text(encoding="utf-8"))
    positioning = protocol["research_positioning"]

    assert positioning["statistical_inference"].startswith(
        "No RQ4 bootstrap, p-value"
    )
    assert positioning["excluded_extension"].startswith(
        "No static-model versus quarterly rolling-retraining"
    )
    assert protocol["upstream_prediction_protocol"][
        "return_metrics_used_for_training_or_selection"
    ] is False
    assert not any(
        "bootstrap" in path.name.casefold()
        for path in RQ4_ROOT.rglob("*")
        if path.is_file()
    )
    assert (RQ3_ROOT / "outputs" / "rq3_paired_block_bootstrap.csv").is_file()
