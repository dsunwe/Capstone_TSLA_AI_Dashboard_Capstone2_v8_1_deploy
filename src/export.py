from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import dump

from .config import (
    BACKTEST_EQUITY_PATH,
    BACKTEST_METRICS_PATH,
    CROSS_ASSET_BENCHMARK_PATH,
    CROSS_ASSET_SUMMARY_PATH,
    CONTEXT_AUDIT_PATH,
    DATASET_SUMMARY_PATH,
    EXPERIMENT_CONFIG_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_ABLATION_PATH,
    FEATURE_FAMILY_ABLATION_PATH,
    FEATURE_SELECTION_PATH,
    FEATURE_STABILITY_PATH,
    FEATURES_DATA_PATH,
    FIXED_MODEL_PREDICTIONS_PATH,
    HELDOUT_BACKTEST_EXECUTION_PATH,
    HELDOUT_TRADE_LEDGER_PATH,
    ML_BASELINE_COMPARISON_PATH,
    ML_RESULTS_PATH,
    MODEL_PATH,
    LSTM_METADATA_PATH,
    LSTM_MODEL_PATH,
    LSTM_PREPROCESSOR_PATH,
    LSTM_RESULTS_PATH,
    LSTM_TRAINING_PATH,
    MAIN_OOF_CLASSIFICATION_UNCERTAINTY_PATH,
    MAIN_OOF_RETURN_SEPARATION_UNCERTAINTY_PATH,
    MAIN_OOF_STRATEGY_BENCHMARK_UNCERTAINTY_PATH,
    OOF_BACKTEST_EQUITY_PATH,
    OOF_BACKTEST_EXECUTION_PATH,
    OOF_BACKTEST_METRICS_PATH,
    OOF_LSTM_SEED_PREDICTIONS_PATH,
    OOF_LSTM_SEED_RESULTS_PATH,
    OOF_NESTED_TUNING_PATH,
    OOF_PREDICTIONS_PATH,
    OOF_SIGNAL_DIAGNOSTICS_PATH,
    OOF_TRADE_LEDGER_PATH,
    PREPROCESSING_AUDIT_PATH,
    SENSITIVITY_PATH,
    TEMPORAL_SPLIT_MANIFEST_PATH,
    TUNING_RESULTS_PATH,
    VALIDATION_REPORT_PATH,
    WALK_FORWARD_RESULTS_PATH,
)
from .ml import MLResult


def save_project_outputs(
    *,
    engineered: pd.DataFrame,
    ml_result: MLResult | None,
    backtest_metrics: pd.DataFrame,
    backtest_equity: pd.DataFrame,
    sensitivity: pd.DataFrame,
    validation_report: pd.DataFrame,
    walk_forward: pd.DataFrame | None = None,
    experiment_config: pd.DataFrame | None = None,
    preprocessing_audit: pd.DataFrame | None = None,
    feature_selection: pd.DataFrame | None = None,
    feature_ablation: pd.DataFrame | None = None,
    tuning_results: pd.DataFrame | None = None,
    dataset_summary: pd.DataFrame | None = None,
    cross_asset_benchmark: pd.DataFrame | None = None,
    cross_asset_summary: pd.DataFrame | None = None,
    feature_stability: pd.DataFrame | None = None,
    oof_predictions: pd.DataFrame | None = None,
    oof_backtest_metrics: pd.DataFrame | None = None,
    oof_backtest_equity: pd.DataFrame | None = None,
    context_audit: pd.DataFrame | None = None,
    feature_family_ablation: pd.DataFrame | None = None,
    temporal_split_manifest: pd.DataFrame | None = None,
    heldout_backtest_execution: pd.DataFrame | None = None,
    oof_backtest_execution: pd.DataFrame | None = None,
    heldout_trade_ledger: pd.DataFrame | None = None,
    oof_trade_ledger: pd.DataFrame | None = None,
    fixed_model_predictions: pd.DataFrame | None = None,
    oof_signal_diagnostics: pd.DataFrame | None = None,
    oof_nested_tuning: pd.DataFrame | None = None,
    oof_lstm_seed_results: pd.DataFrame | None = None,
    oof_lstm_seed_predictions: pd.DataFrame | None = None,
    main_oof_classification_uncertainty: pd.DataFrame | None = None,
    main_oof_return_separation_uncertainty: pd.DataFrame | None = None,
    main_oof_strategy_benchmark_uncertainty: pd.DataFrame | None = None,
) -> list[Path]:
    saved: list[Path] = []
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_DATA_PATH.write_text(engineered.to_csv(index=False), encoding="utf-8")
    saved.append(FEATURES_DATA_PATH)

    if ml_result is not None:
        ML_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ml_result.results.to_csv(ML_RESULTS_PATH, index=False)
        ml_result.baseline_comparison.to_csv(ML_BASELINE_COMPARISON_PATH, index=False)
        ml_result.feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        saved.extend([ML_RESULTS_PATH, ML_BASELINE_COMPARISON_PATH, FEATURE_IMPORTANCE_PATH])
        if ml_result.model_family == "sequence":
            ml_result.results.to_csv(LSTM_RESULTS_PATH, index=False)
            ml_result.training_history.to_csv(LSTM_TRAINING_PATH, index=False)
            dump(ml_result.preprocessor, LSTM_PREPROCESSOR_PATH)
            model_paths: list[str] = []
            models = ml_result.model if isinstance(ml_result.model, list) else [ml_result.model]
            for index, model in enumerate(models):
                seed = (
                    int(ml_result.seed_results.iloc[index]["Seed"])
                    if index < len(ml_result.seed_results)
                    else index
                )
                path = (
                    LSTM_MODEL_PATH
                    if index == 0
                    else LSTM_MODEL_PATH.with_name(
                        f"{LSTM_MODEL_PATH.stem}_seed_{seed}{LSTM_MODEL_PATH.suffix}"
                    )
                )
                model.save(path)
                model_paths.append(path.name)
                saved.append(path)
            metadata = {
                "model_name": ml_result.model_name,
                "model_files": model_paths,
                "feature_names": ml_result.feature_names,
                "lookback": ml_result.lookback,
                "parameter_count": ml_result.parameter_count,
                "selected_parameters": ml_result.selected_parameters,
                "train_end": str(ml_result.train_end_date),
                "test_start": str(ml_result.test_start_date),
                "split_plan_id": ml_result.split_plan_id,
                "split_plan_sha256": ml_result.split_plan_hash,
                "purge_start": str(ml_result.purge_start_date),
                "purge_end": str(ml_result.purge_end_date),
                "fixed_holdout_feature_protocol": (
                    "training-only selection on the canonical common temporal plan"
                ),
                "expanding_oof_feature_protocol": (
                    "outer-fold training-only selection; see walk-forward tables"
                ),
            }
            LSTM_METADATA_PATH.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            saved.extend(
                [
                    LSTM_RESULTS_PATH,
                    LSTM_TRAINING_PATH,
                    LSTM_PREPROCESSOR_PATH,
                    LSTM_METADATA_PATH,
                ]
            )
        else:
            dump(ml_result.model, MODEL_PATH)
            saved.append(MODEL_PATH)

    if not backtest_metrics.empty:
        backtest_metrics.to_csv(BACKTEST_METRICS_PATH, index=False)
        saved.append(BACKTEST_METRICS_PATH)
    if not backtest_equity.empty:
        backtest_equity.to_csv(BACKTEST_EQUITY_PATH, index=False)
        saved.append(BACKTEST_EQUITY_PATH)
    if not sensitivity.empty:
        sensitivity.to_csv(SENSITIVITY_PATH, index=False)
        saved.append(SENSITIVITY_PATH)
    if not validation_report.empty:
        validation_report.to_csv(VALIDATION_REPORT_PATH, index=False)
        saved.append(VALIDATION_REPORT_PATH)
    if walk_forward is not None:
        # Overwrite an earlier result even when the active configuration has
        # not been validated yet; this prevents a stale CSV from appearing to
        # belong to the current settings.
        walk_forward.to_csv(WALK_FORWARD_RESULTS_PATH, index=False)
        saved.append(WALK_FORWARD_RESULTS_PATH)
    if experiment_config is not None and not experiment_config.empty:
        experiment_config.to_csv(EXPERIMENT_CONFIG_PATH, index=False)
        saved.append(EXPERIMENT_CONFIG_PATH)
    optional_tables = [
        (preprocessing_audit, PREPROCESSING_AUDIT_PATH),
        (feature_selection, FEATURE_SELECTION_PATH),
        (feature_ablation, FEATURE_ABLATION_PATH),
        (tuning_results, TUNING_RESULTS_PATH),
        (dataset_summary, DATASET_SUMMARY_PATH),
        (cross_asset_benchmark, CROSS_ASSET_BENCHMARK_PATH),
        (cross_asset_summary, CROSS_ASSET_SUMMARY_PATH),
        (feature_stability, FEATURE_STABILITY_PATH),
        (oof_predictions, OOF_PREDICTIONS_PATH),
        (oof_backtest_metrics, OOF_BACKTEST_METRICS_PATH),
        (oof_backtest_equity, OOF_BACKTEST_EQUITY_PATH),
        (context_audit, CONTEXT_AUDIT_PATH),
        (feature_family_ablation, FEATURE_FAMILY_ABLATION_PATH),
        (temporal_split_manifest, TEMPORAL_SPLIT_MANIFEST_PATH),
        (heldout_backtest_execution, HELDOUT_BACKTEST_EXECUTION_PATH),
        (oof_backtest_execution, OOF_BACKTEST_EXECUTION_PATH),
        (heldout_trade_ledger, HELDOUT_TRADE_LEDGER_PATH),
        (oof_trade_ledger, OOF_TRADE_LEDGER_PATH),
        (fixed_model_predictions, FIXED_MODEL_PREDICTIONS_PATH),
        (oof_signal_diagnostics, OOF_SIGNAL_DIAGNOSTICS_PATH),
        (oof_nested_tuning, OOF_NESTED_TUNING_PATH),
        (oof_lstm_seed_results, OOF_LSTM_SEED_RESULTS_PATH),
        (oof_lstm_seed_predictions, OOF_LSTM_SEED_PREDICTIONS_PATH),
        (
            main_oof_classification_uncertainty,
            MAIN_OOF_CLASSIFICATION_UNCERTAINTY_PATH,
        ),
        (
            main_oof_return_separation_uncertainty,
            MAIN_OOF_RETURN_SEPARATION_UNCERTAINTY_PATH,
        ),
        (
            main_oof_strategy_benchmark_uncertainty,
            MAIN_OOF_STRATEGY_BENCHMARK_UNCERTAINTY_PATH,
        ),
    ]
    for table, path in optional_tables:
        if table is not None:
            table.to_csv(path, index=False)
            saved.append(path)
    return saved
