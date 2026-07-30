"""Generate all frozen v7 sentiment-sub-study evidence and figures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_FIGURES_DIR,
    OUTPUT_TABLES_DIR,
    SENTIMENT_ABLATION_PATH,
    SENTIMENT_BOOTSTRAP_PATH,
    SENTIMENT_OOF_PATH,
    SENTIMENT_PLACEBO_DISTRIBUTION_PATH,
    SENTIMENT_PLACEBO_PATH,
    SENTIMENT_SUMMARY_PATH,
    SENTIMENT_THRESHOLDS_PATH,
)
from src.sentiment_study import (
    build_bootstrap_evidence,
    feature_set_table,
    paired_feature_set_deltas,
    prepare_sentiment_substudy,
    run_dual_target_benchmark,
    run_primary_placebo_test,
)
from src.validation import validate_sentiment_study


def save_coverage_figure(coverage: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(coverage))
    width = 0.36
    axis.bar(
        x - width / 2,
        coverage["Raw Articles"],
        width,
        label="Raw source rows",
        color="#94a3b8",
    )
    axis.bar(
        x + width / 2,
        coverage["Unique Stories"],
        width,
        label="Usable mapped stories",
        color="#0f766e",
    )
    axis.set_xticks(x, coverage["Asset"])
    axis.set_ylabel("Headlines")
    axis.set_title("Frozen Firm-News Coverage and Deduplication")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_primary_ablation_figure(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[
        summary["Target"].eq("large_move")
        & summary["Model"].eq("Random Forest Classifier")
    ].copy()
    pivot = selected.pivot(index="Feature Set", columns="Asset", values="PR-AUC")
    pivot = pivot.reindex(["F0", "F1", "F2", "F3", "F4", "F5"])
    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    pivot.plot(kind="bar", ax=axis, color=["#64748b", "#f59e0b", "#0f766e"])
    axis.set_ylabel("Average Precision (PR-AUC)")
    axis.set_title("Large-Move Random Forest: F0–F5 Out-of-Fold Results")
    axis.set_xlabel("Feature set")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Asset")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_bootstrap_figure(bootstrap: pd.DataFrame, path: Path) -> None:
    selected = bootstrap.loc[
        bootstrap["Block Length"].eq(10)
        & bootstrap["Model"].eq("Random Forest Classifier")
    ].copy()
    selected["Label"] = selected["Asset"] + " · " + selected["Target"]
    selected = selected.sort_values(["Target", "Asset"]).reset_index(drop=True)
    positions = np.arange(len(selected))
    difference = selected["Difference"].to_numpy()
    lower = np.maximum(difference - selected["CI Lower"].to_numpy(), 0.0)
    upper = np.maximum(selected["CI Upper"].to_numpy() - difference, 0.0)
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    colors = np.where(selected["CI Lower"].gt(0), "#0f766e", "#64748b")
    for index, color in enumerate(colors):
        axis.errorbar(
            difference[index],
            positions[index],
            xerr=np.asarray([[lower[index]], [upper[index]]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            linewidth=2,
            markersize=6,
        )
    axis.axvline(0, color="#b91c1c", linestyle="--", linewidth=1)
    axis.set_yticks(positions, selected["Label"])
    axis.set_xlabel("F5 − F1 metric difference")
    axis.set_title("Paired Moving-Block Bootstrap 95% Intervals (L=10)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_placebo_figure(
    placebo_summary: pd.DataFrame,
    null_distribution: pd.DataFrame,
    path: Path,
) -> None:
    observed = float(placebo_summary["Observed Delta"].iloc[0])
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.hist(
        null_distribution["Placebo Delta PR-AUC"],
        bins=24,
        color="#94a3b8",
        edgecolor="white",
    )
    axis.axvline(
        observed,
        color="#b91c1c",
        linewidth=2,
        label=f"Observed F5−F1 = {observed:.4f}",
    )
    axis.set_xlabel("Placebo F5 − observed F1 PR-AUC")
    axis.set_ylabel("Replications")
    axis.set_title("TSLA Large-Move Random-Forest Placebo Distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--lstm-units", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--placebo-iterations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-lstm", action="store_true")
    args = parser.parse_args()

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing common TSLA/AAPL/NVDA short-window frames...", flush=True)
    frames = prepare_sentiment_substudy()
    print("Running dual targets × F0-F5 × five models × outer folds...", flush=True)
    result = run_dual_target_benchmark(
        frames,
        n_splits=args.folds,
        random_state=args.random_state,
        include_lstm=not args.skip_lstm,
        lstm_units=args.lstm_units,
        lstm_epochs=args.lstm_epochs,
    )
    result.oof_predictions.to_csv(SENTIMENT_OOF_PATH, index=False)
    result.summary.to_csv(SENTIMENT_SUMMARY_PATH, index=False)
    result.summary.to_csv(SENTIMENT_ABLATION_PATH, index=False)
    result.fold_thresholds.to_csv(SENTIMENT_THRESHOLDS_PATH, index=False)
    model_summary = (
        result.summary.groupby(["Target", "Model"], as_index=False)
        .agg(
            Mean_Macro_F1=("Macro F1", "mean"),
            Mean_PR_AUC=("PR-AUC", "mean"),
            Mean_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            Mean_Brier_Score=("Brier Score", "mean"),
        )
        .sort_values(["Target", "Mean_PR_AUC"], ascending=[True, False])
    )
    model_summary.to_csv(
        OUTPUT_TABLES_DIR / "v7_cross_asset_model_summary.csv",
        index=False,
    )
    paired_feature_set_deltas(result.summary).to_csv(
        OUTPUT_TABLES_DIR / "v7_f1_f5_paired_deltas.csv",
        index=False,
    )
    feature_set_table().to_csv(
        OUTPUT_TABLES_DIR / "v7_feature_set_definitions.csv",
        index=False,
    )
    print("Running paired moving-block bootstrap intervals...", flush=True)
    bootstrap = build_bootstrap_evidence(
        result.oof_predictions,
        n_bootstrap=args.bootstrap_replicates,
        random_state=args.random_state,
    )
    bootstrap.to_csv(SENTIMENT_BOOTSTRAP_PATH, index=False)
    print(
        "Running the pre-specified primary TSLA large-move comparison "
        "placebo...",
        flush=True,
    )
    placebo, distribution = run_primary_placebo_test(
        result.study_frames["TSLA"],
        result.oof_predictions,
        iterations=args.placebo_iterations,
        n_splits=args.folds,
        random_state=args.random_state,
    )
    placebo.to_csv(SENTIMENT_PLACEBO_PATH, index=False)
    distribution.to_csv(SENTIMENT_PLACEBO_DISTRIBUTION_PATH, index=False)
    primary_bootstrap = bootstrap.loc[
        bootstrap["Asset"].eq("TSLA")
        & bootstrap["Target"].eq("large_move")
        & bootstrap["Block Length"].eq(10)
    ].iloc[0]
    primary_placebo = placebo.iloc[0]
    pd.DataFrame(
        [
            {
                "Finding": (
                    "Pre-specified primary comparison in the project workflow: "
                    "TSLA large-move incremental news value"
                ),
                "Reference": "F1",
                "Comparison": "F5",
                "Metric": "PR-AUC",
                "Reference Score": primary_bootstrap["Baseline Score"],
                "Comparison Score": primary_bootstrap["Comparison Score"],
                "Difference": primary_bootstrap["Difference"],
                "CI Lower": primary_bootstrap["CI Lower"],
                "CI Upper": primary_bootstrap["CI Upper"],
                "Placebo p-value": primary_placebo["Empirical p-value"],
                "Interpretation": "No robust positive incremental news value",
            }
        ]
    ).to_csv(OUTPUT_TABLES_DIR / "v7_key_findings.csv", index=False)

    daily = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "news" / "daily_news_features.csv"
    )
    validation = validate_sentiment_study(
        daily,
        result.oof_predictions,
        result.fold_thresholds,
    )
    validation.to_csv(
        OUTPUT_TABLES_DIR / "v7_sentiment_validation_report.csv",
        index=False,
    )
    configuration = {
        "outer_folds": args.folds,
        "purge_gap": 1,
        "large_move_training_quantile": 0.80,
        "probability_threshold": 0.50,
        "lstm_units": args.lstm_units,
        "lstm_epochs": args.lstm_epochs,
        "lstm_lookback": 20,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_lengths": [5, 10, 20],
        "placebo_iterations": args.placebo_iterations,
        "placebo_block_length": 5,
        "random_state": args.random_state,
        "primary_comparison_in_project_workflow": (
            "TSLA / large_move / Random Forest / F5 - F1 / PR-AUC"
        ),
    }
    (OUTPUT_TABLES_DIR / "v7_sentiment_experiment_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    coverage = pd.read_csv(OUTPUT_TABLES_DIR / "news_coverage_audit.csv")
    save_coverage_figure(
        coverage, OUTPUT_FIGURES_DIR / "12_news_coverage_and_deduplication.png"
    )
    save_primary_ablation_figure(
        result.summary, OUTPUT_FIGURES_DIR / "13_v7_f0_f5_large_move_ablation.png"
    )
    save_bootstrap_figure(
        bootstrap, OUTPUT_FIGURES_DIR / "14_v7_block_bootstrap_intervals.png"
    )
    save_placebo_figure(
        placebo,
        distribution,
        OUTPUT_FIGURES_DIR / "15_v7_primary_placebo_distribution.png",
    )
    passed = int(validation["Status"].eq("Pass").sum())
    print(
        f"Completed v7 outputs: {len(result.summary)} score rows, "
        f"{len(result.oof_predictions):,} OOF rows, "
        f"{passed}/{len(validation)} v7 internal checks passed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
