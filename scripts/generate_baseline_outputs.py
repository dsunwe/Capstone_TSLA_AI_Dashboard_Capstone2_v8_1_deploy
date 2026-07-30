"""Legacy v7.5 output generator retained only for provenance.

This script does not implement the final v8 five-session RQ2-RQ4 protocol and
must not be used to regenerate submission evidence. Use
``scripts/reproduce_all.py`` instead. An explicit legacy flag is required to
avoid accidental overwriting of retained v7.5 continuity artifacts.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.backtesting import (
    ML_STRATEGY_SPEC,
    RULE_STRATEGY_SPECS,
    available_strategy_specs,
    build_multi_model_backtest_comparison,
    build_oof_signal_diagnostics,
    build_trade_ledger,
    run_single_window_sensitivity,
)
from src.config import (
    CANDIDATE_FEATURES,
    DATA_PROCESSED_DIR,
    DATASET_PATHS,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_LOOKBACK,
    DEFAULT_LSTM_SEEDS,
    DEFAULT_LSTM_UNITS,
    FEATURE_IMPORTANCE_PATH,
    LSTM_MODEL_NAME,
    ML_BASELINE_COMPARISON_PATH,
    MODEL_LEVEL_NAMES,
    ML_RESULTS_PATH,
    OUTPUT_FIGURES_DIR,
    WALK_FORWARD_RESULTS_PATH,
    ensure_project_directories,
)
from src.context import build_context_frame
from src.data import load_and_preprocess_csv, preprocessing_audit_table
from src.experiment import build_experiment_configuration
from src.export import save_project_outputs
from src.ml import (
    fixed_holdout_model_suite,
    run_classification,
    walk_forward_model_suite,
)
from src.main_study_inference import (
    build_classification_uncertainty,
    build_return_separation_uncertainty,
    build_strategy_benchmark_uncertainty,
)
from src.research import (
    prepare_multi_asset_engineered,
    run_feature_family_ablation,
    run_feature_group_ablation,
    run_multi_asset_benchmark,
    select_features_from_training,
)
from src.temporal import build_temporal_split_plan
from src.validation import (
    combine_validation_reports,
    validate_backtest,
    validate_context,
    validate_dataset,
    validate_features,
    validate_ml_result,
    validate_oof_robustness_evidence,
    validate_temporal_plan_alignment,
)
from src.visualization import (
    ablation_metric_chart,
    chronological_split_chart,
    confusion_matrix_heatmap,
    cross_asset_model_chart,
    feature_importance_chart,
    feature_stability_chart,
    main_oof_uncertainty_chart,
    model_line_chart,
    normalised_asset_price_chart,
    walk_forward_metric_chart,
)


def save_figure(fig, filename: str) -> Path | None:
    if fig is None:
        return None
    path = OUTPUT_FIGURES_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def combined_trade_ledgers(
    strategy_frame: pd.DataFrame,
    *,
    validation_label: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    strategy_suffix: str,
) -> pd.DataFrame:
    """Build an auditable all-strategy ledger from one execution-aligned frame."""

    ledgers: list[pd.DataFrame] = []
    if "Signal Model" in strategy_frame.columns:
        model_order = strategy_frame["Signal Model"].drop_duplicates().tolist()
        reference = strategy_frame.loc[
            strategy_frame["Signal Model"].eq(model_order[0])
        ].reset_index(drop=True)
        specifications = list(RULE_STRATEGY_SPECS)
    else:
        model_order = []
        reference = strategy_frame
        specifications = available_strategy_specs(reference)

    for strategy, _, position_column in specifications:
        ledger = build_trade_ledger(
            reference["Execution Date"],
            reference[position_column],
            reference["Open-to-Open Return"],
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            prices=reference["Execution Open"],
            signal_dates=reference["Signal Date"],
            valuation_dates=reference["Valuation Date"],
            valuation_prices=reference["Valuation Open"],
        )
        ledger.insert(0, "Strategy", strategy)
        ledger.insert(0, "Validation", validation_label)
        ledgers.append(ledger)

    for model_name in model_order:
        model_frame = strategy_frame.loc[
            strategy_frame["Signal Model"].eq(model_name)
        ].reset_index(drop=True)
        ledger = build_trade_ledger(
            model_frame["Execution Date"],
            model_frame[ML_STRATEGY_SPEC[2]],
            model_frame["Open-to-Open Return"],
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            prices=model_frame["Execution Open"],
            signal_dates=model_frame["Signal Date"],
            valuation_dates=model_frame["Valuation Date"],
            valuation_prices=model_frame["Valuation Open"],
        )
        ledger.insert(0, "Strategy", f"{model_name} {strategy_suffix}")
        ledger.insert(0, "Validation", validation_label)
        ledgers.append(ledger)
    return pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def main() -> None:
    ensure_project_directories()
    formal_lstm_epochs = 40
    engineered_by_asset, dataset_summary = prepare_multi_asset_engineered()
    split_plan = build_temporal_split_plan(engineered_by_asset)
    for asset, frame in engineered_by_asset.items():
        frame.to_csv(DATA_PROCESSED_DIR / f"{asset}_features_common_window.csv", index=False)

    cross_asset = run_multi_asset_benchmark(
        engineered_by_asset,
        top_k_features=15,
        tune=True,
        tuning_iterations=4,
        include_lstm=True,
        lstm_lookback=DEFAULT_LSTM_LOOKBACK,
        lstm_units=DEFAULT_LSTM_UNITS,
        lstm_epochs=formal_lstm_epochs,
        lstm_seeds=tuple(DEFAULT_LSTM_SEEDS),
        split_plan=split_plan,
    )

    tsla = engineered_by_asset["TSLA"]
    context_result = build_context_frame(tsla["Date"])
    selection = select_features_from_training(
        tsla,
        top_k=15,
        split_plan=split_plan,
    )
    ablation = run_feature_group_ablation(
        tsla,
        selection.selected_features,
        model_name="Logistic Regression (Baseline)",
        split_plan=split_plan,
    )
    family_ablation = run_feature_family_ablation(
        tsla,
        model_name="Logistic Regression (Baseline)",
        split_plan=split_plan,
    )
    ml_result = run_classification(
        tsla,
        model_name=LSTM_MODEL_NAME,
        feature_names=selection.selected_features,
        train_size=0.80,
        probability_threshold=0.50,
        include_benchmark=True,
        lstm_lookback=DEFAULT_LSTM_LOOKBACK,
        lstm_units=DEFAULT_LSTM_UNITS,
        lstm_dropout=DEFAULT_LSTM_DROPOUT,
        lstm_epochs=formal_lstm_epochs,
        lstm_batch_size=DEFAULT_LSTM_BATCH_SIZE,
        lstm_seeds=tuple(DEFAULT_LSTM_SEEDS),
        split_plan=split_plan,
    )
    if ml_result is None:
        raise RuntimeError("TSLA LSTM could not be trained.")

    fixed_model_predictions = fixed_holdout_model_suite(
        tsla,
        feature_names=selection.selected_features,
        split_plan=split_plan,
        probability_threshold=0.50,
        lstm_result=ml_result,
        lstm_lookback=DEFAULT_LSTM_LOOKBACK,
        lstm_units=DEFAULT_LSTM_UNITS,
        lstm_dropout=DEFAULT_LSTM_DROPOUT,
        lstm_epochs=formal_lstm_epochs,
        lstm_batch_size=DEFAULT_LSTM_BATCH_SIZE,
        lstm_seeds=tuple(DEFAULT_LSTM_SEEDS),
        tune_tabular=True,
        tuning_splits=3,
        tuning_iterations=4,
    )
    (
        walk_forward,
        oof_predictions,
        oof_nested_tuning,
        oof_lstm_seed_results,
        oof_lstm_seed_predictions,
    ) = walk_forward_model_suite(
        tsla,
        model_names=tuple(MODEL_LEVEL_NAMES),
        n_splits=5,
        probability_threshold=0.50,
        feature_names=CANDIDATE_FEATURES,
        lstm_lookback=DEFAULT_LSTM_LOOKBACK,
        lstm_units=DEFAULT_LSTM_UNITS,
        lstm_dropout=DEFAULT_LSTM_DROPOUT,
        lstm_epochs=30,
        lstm_batch_size=DEFAULT_LSTM_BATCH_SIZE,
        lstm_seeds=tuple(DEFAULT_LSTM_SEEDS),
        foldwise_feature_selection=True,
        candidate_features=CANDIDATE_FEATURES,
        top_k_features=15,
        nested_tuning=True,
        inner_splits=3,
        tuning_iterations=4,
        return_diagnostics=True,
    )
    heldout_metrics, heldout_equity, strategy_frame = (
        build_multi_model_backtest_comparison(
            tsla,
            fixed_model_predictions,
            transaction_cost_bps=10,
            slippage_bps=5,
            validation_label="Fixed holdout",
            strategy_suffix="Fixed Signal",
        )
    )
    oof_metrics, oof_equity, oof_strategy_frame = (
        build_multi_model_backtest_comparison(
            tsla,
            oof_predictions,
            transaction_cost_bps=10,
            slippage_bps=5,
            validation_label="Walk-forward out-of-fold",
            strategy_suffix="OOF Signal",
        )
    )
    oof_signal_diagnostics = build_oof_signal_diagnostics(oof_predictions)
    main_oof_classification_uncertainty = build_classification_uncertainty(
        oof_predictions,
        n_bootstrap=2_000,
        block_length=20,
        random_state=42,
    )
    main_oof_return_separation_uncertainty = (
        build_return_separation_uncertainty(
            oof_predictions,
            n_bootstrap=2_000,
            block_length=20,
            random_state=42,
        )
    )
    main_oof_strategy_benchmark_uncertainty = (
        build_strategy_benchmark_uncertainty(
            oof_predictions,
            oof_strategy_frame,
            n_bootstrap=2_000,
            block_length=20,
            random_state=42,
        )
    )
    sensitivity = run_single_window_sensitivity(
        tsla,
        transaction_cost_bps=10,
        slippage_bps=5,
    )
    _, tsla_profile = load_and_preprocess_csv(DATASET_PATHS["TSLA"])
    preprocessing_audit = preprocessing_audit_table(tsla_profile)
    settings = {
        "threshold": 0.50,
        "n_estimators": 200,
        "max_depth": 6,
        "cv_splits": 5,
        "transaction_cost_bps": 10,
        "slippage_bps": 5,
        "tune_model": True,
        "oof_nested_tuning": (
            "Three tabular estimators: 3-split inner TimeSeriesSplit(gap=2), "
            "4 candidates, Macro-F1 selection"
        ),
        "oof_lstm_seeds": ", ".join(str(seed) for seed in DEFAULT_LSTM_SEEDS),
        "lstm_lookback": DEFAULT_LSTM_LOOKBACK,
        "lstm_units": DEFAULT_LSTM_UNITS,
        "lstm_dropout": DEFAULT_LSTM_DROPOUT,
        "lstm_epochs": 30,
        "fixed_holdout_lstm_epochs": formal_lstm_epochs,
        "lstm_batch_size": DEFAULT_LSTM_BATCH_SIZE,
    }
    experiment_config = build_experiment_configuration(
        tsla,
        source_name=DATASET_PATHS["TSLA"].name,
        model_name="Five model-level OOF suite",
        settings=settings,
        selected_features=selection.selected_features,
        split_plan=split_plan,
    )
    validation = combine_validation_reports(
        validate_dataset(tsla),
        validate_context(context_result.audit, tsla),
        validate_features(tsla, selection.selected_features),
        validate_ml_result(ml_result),
        validate_temporal_plan_alignment(selection, ml_result, split_plan),
        validate_backtest(heldout_metrics, heldout_equity, strategy_frame),
        validate_oof_robustness_evidence(
            oof_predictions,
            oof_nested_tuning,
            oof_lstm_seed_results,
            oof_lstm_seed_predictions,
            main_oof_classification_uncertainty,
            main_oof_return_separation_uncertainty,
            main_oof_strategy_benchmark_uncertainty,
        ),
    )
    heldout_trade_ledger = combined_trade_ledgers(
        strategy_frame,
        validation_label="Fixed holdout",
        transaction_cost_bps=10,
        slippage_bps=5,
        strategy_suffix="Fixed Signal",
    )
    oof_trade_ledger = combined_trade_ledgers(
        oof_strategy_frame,
        validation_label="Walk-forward out-of-fold",
        transaction_cost_bps=10,
        slippage_bps=5,
        strategy_suffix="OOF Signal",
    )
    save_project_outputs(
        engineered=tsla,
        ml_result=ml_result,
        backtest_metrics=heldout_metrics,
        backtest_equity=heldout_equity,
        sensitivity=sensitivity,
        validation_report=validation,
        walk_forward=walk_forward,
        experiment_config=experiment_config,
        preprocessing_audit=preprocessing_audit,
        feature_selection=selection.table,
        feature_ablation=ablation,
        tuning_results=cross_asset.tuning_results,
        dataset_summary=dataset_summary,
        cross_asset_benchmark=cross_asset.benchmark,
        cross_asset_summary=cross_asset.summary,
        feature_stability=cross_asset.feature_stability,
        oof_predictions=oof_predictions,
        oof_backtest_metrics=oof_metrics,
        oof_backtest_equity=oof_equity,
        context_audit=context_result.audit,
        feature_family_ablation=family_ablation,
        temporal_split_manifest=split_plan.manifest(),
        heldout_backtest_execution=strategy_frame,
        oof_backtest_execution=oof_strategy_frame,
        heldout_trade_ledger=heldout_trade_ledger,
        oof_trade_ledger=oof_trade_ledger,
        fixed_model_predictions=fixed_model_predictions,
        oof_signal_diagnostics=oof_signal_diagnostics,
        oof_nested_tuning=oof_nested_tuning,
        oof_lstm_seed_results=oof_lstm_seed_results,
        oof_lstm_seed_predictions=oof_lstm_seed_predictions,
        main_oof_classification_uncertainty=(
            main_oof_classification_uncertainty
        ),
        main_oof_return_separation_uncertainty=(
            main_oof_return_separation_uncertainty
        ),
        main_oof_strategy_benchmark_uncertainty=(
            main_oof_strategy_benchmark_uncertainty
        ),
    )

    figure_paths = [
        save_figure(normalised_asset_price_chart(engineered_by_asset), "01_multi_asset_prices.png"),
        save_figure(
            chronological_split_chart(
                ml_result.train_start_date,
                ml_result.train_end_date,
                ml_result.test_start_date,
                ml_result.test_end_date,
            ),
            "02_chronological_split.png",
        ),
        save_figure(ablation_metric_chart(family_ablation), "03_feature_family_ablation.png"),
        save_figure(ablation_metric_chart(ablation), "04_feature_group_ablation.png"),
        save_figure(
            cross_asset_model_chart(cross_asset.benchmark),
            "05_cross_asset_five_model_benchmark.png",
        ),
        save_figure(
            feature_stability_chart(cross_asset.feature_stability),
            "06_feature_stability.png",
        ),
        save_figure(
            confusion_matrix_heatmap(ml_result.confusion_matrix),
            "07_tsla_lstm_confusion_matrix.png",
        ),
        save_figure(
            feature_importance_chart(ml_result.feature_importance),
            "08_tsla_lstm_occlusion_importance.png",
        ),
        save_figure(
            walk_forward_metric_chart(walk_forward),
            "09_all_model_oof_accuracy.png",
        ),
        save_figure(
            model_line_chart(
                oof_equity,
                "Equity",
                "All-Model Walk-Forward OOF Equity",
            ),
            "10_all_model_oof_backtesting_equity.png",
        ),
        save_figure(
            model_line_chart(
                oof_equity,
                "Drawdown",
                "All-Model Walk-Forward OOF Drawdown",
            ),
            "11_all_model_oof_backtesting_drawdown.png",
        ),
        save_figure(
            main_oof_uncertainty_chart(
                main_oof_classification_uncertainty,
                main_oof_return_separation_uncertainty,
            ),
            "16_main_oof_uncertainty_intervals.png",
        ),
    ]
    figure_paths = [path for path in figure_paths if path is not None]

    print("Capstone II evidence outputs generated.")
    print("\nCross-asset summary:")
    print(cross_asset.summary.to_string(index=False))
    print("\nTSLA held-out comparison:")
    print(ml_result.baseline_comparison.to_string(index=False))
    print("\nTSLA walk-forward mean:")
    print(walk_forward.mean(numeric_only=True).to_string())
    print("\nOOF backtest metrics:")
    print(oof_metrics.to_string(index=False))
    print(f"\nInternal validation: {(validation['Status'] == 'Pass').sum()}/{len(validation)} passed")
    print(f"Presentation figures: {len(figure_paths)}")
    print(f"ML results: {ML_RESULTS_PATH}")
    print(f"ML comparison: {ML_BASELINE_COMPARISON_PATH}")
    print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"Walk-forward results: {WALK_FORWARD_RESULTS_PATH}")


if __name__ == "__main__":
    if "--allow-legacy-v75" not in sys.argv:
        raise SystemExit(
            "Legacy v7.5 generator blocked. Use scripts/reproduce_all.py for "
            "the v8 RQ2-RQ4 evidence. To inspect the historical generator "
            "deliberately, rerun with --allow-legacy-v75."
        )
    main()
