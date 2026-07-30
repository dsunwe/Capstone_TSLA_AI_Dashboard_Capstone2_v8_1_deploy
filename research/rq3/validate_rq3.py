"""Validate saved v8.3 RQ3 evidence without retraining the four models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

from run_rq3_experiment import (
    AUDIT_START,
    FEATURES,
    HORIZON,
    LOOKBACK,
    MAIN_MODELS,
    OUTER_SPLITS,
    OUTER_TEST_SIZE,
    PURGE_ROWS,
    THRESHOLD,
    prepare_data,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FREEZE = ROOT / "freeze"
DATA = ROOT / "data" / "TSLA_features_common_window.csv"
PROTOCOL = ROOT / "protocol.json"
UPSTREAM_PROTOCOL = ROOT / "upstream" / "rq2_protocol.json"
UPSTREAM_OOF = ROOT / "upstream" / "rq2_feature_family_oof_predictions.csv"
UPSTREAM_AUDIT = (
    ROOT / "upstream" / "rq2_feature_family_audit_predictions.csv"
)

EXPECTED_COMPARATORS = {
    *MAIN_MODELS,
    "Training-majority",
    "Weekly-trend persistence",
    "EMA8/13 state",
    "RQ2 current-day Logistic Regression",
}
STOCHASTIC_MODELS = {"Random Forest", "XGBoost", "Compact LSTM"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "Check": name,
        "Status": "Pass" if bool(passed) else "Fail",
        "Evidence": evidence,
    }


def sets_match_by_group(
    frame: pd.DataFrame,
    group_column: str,
    date_column: str = "Date",
) -> tuple[bool, int]:
    date_sets = [
        set(pd.to_datetime(group[date_column]))
        for _, group in frame.groupby(group_column, sort=False)
    ]
    if not date_sets:
        return False, 0
    return all(values == date_sets[0] for values in date_sets), len(date_sets[0])


def metric_triplet(frame: pd.DataFrame) -> tuple[float, float, float]:
    actual = frame["Actual"].astype(int)
    prediction = frame["Prediction"].astype(int)
    return (
        accuracy_score(actual, prediction),
        balanced_accuracy_score(actual, prediction),
        f1_score(actual, prediction, average="macro", zero_division=0),
    )


def main(*, write_artifacts: bool = False) -> None:
    rows: list[dict[str, object]] = []
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    metadata = json.loads(
        (OUTPUTS / "rq3_experiment_metadata.json").read_text(encoding="utf-8")
    )
    expected_hashes = {
        DATA: protocol["input_snapshot"]["sha256"],
        UPSTREAM_PROTOCOL: protocol["upstream_rq2"]["protocol_sha256"],
        UPSTREAM_OOF: protocol["upstream_rq2"]["oof_predictions_sha256"],
        UPSTREAM_AUDIT: protocol["upstream_rq2"][
            "audit_predictions_sha256"
        ],
    }
    actual_hashes = {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in expected_hashes
    }
    rows.append(
        check(
            "Frozen input and upstream RQ2 hashes match the RQ3 protocol",
            all(
                actual_hashes[path.relative_to(ROOT).as_posix()] == expected
                for path, expected in expected_hashes.items()
            ),
            json.dumps(actual_hashes, sort_keys=True),
        )
    )
    rows.append(
        check(
            "Saved metadata identifies the executed protocol and script",
            metadata["protocol_sha256"] == file_sha256(PROTOCOL)
            and metadata["run_script_sha256"]
            == file_sha256(ROOT / "run_rq3_experiment.py")
            and metadata["input_sha256"] == file_sha256(DATA),
            (
                f"protocol={metadata['protocol_sha256']}; "
                f"script={metadata['run_script_sha256']}; "
                f"input={metadata['input_sha256']}"
            ),
        )
    )
    rows.append(
        check(
            "Protocol freezes the six RQ2 features and common 20-session input",
            protocol["frozen_features"] == FEATURES
            and protocol["input_representation"][
                "lookback_trading_sessions"
            ]
            == LOOKBACK
            and protocol["input_representation"]["sequence_shape"]
            == [20, 6],
            f"features={FEATURES}; shape=[20, 6]",
        )
    )

    prepared = prepare_data()
    origins = prepared.origins
    independent_current = prepared.full["Close"].rolling(HORIZON).mean()
    independent_future = sum(
        prepared.full["Close"].shift(-offset)
        for offset in range(1, HORIZON + 1)
    ) / HORIZON
    independent_target = (independent_future > independent_current).astype(int)
    origin_positions = origins["Full_Row_Position"].astype(int).to_numpy()
    target_matches = np.array_equal(
        origins["Weekly_Trend_Direction"].astype(int).to_numpy(),
        independent_target.iloc[origin_positions].astype(int).to_numpy(),
    )
    rows.append(
        check(
            "Target independently recomputes from forward and trailing mean closes",
            target_matches,
            f"origins={len(origins)}; mismatches={int(not target_matches)}",
        )
    )
    sequence_last_matches = np.allclose(
        prepared.sequences[:, -1, :],
        origins[FEATURES].to_numpy(dtype=float),
        equal_nan=True,
    )
    sequence_dates_match = (
        pd.to_datetime(origins["Sequence_End"]).equals(origins["Date"])
        and (
            pd.to_datetime(origins["Sequence_Start"])
            < pd.to_datetime(origins["Sequence_End"])
        ).all()
    )
    rows.append(
        check(
            "Every model origin has one causal 20 by 6 sequence ending at t",
            prepared.sequences.shape == (2_458, 20, 6)
            and sequence_last_matches
            and sequence_dates_match,
            (
                f"shape={prepared.sequences.shape}; "
                f"last_step_match={sequence_last_matches}; "
                f"date_match={sequence_dates_match}"
            ),
        )
    )
    development = origins.iloc[prepared.development_indices]
    audit = origins.iloc[prepared.audit_indices]
    rows.append(
        check(
            "Development and retrospective-audit populations retain all RQ2 origins",
            len(development) == 2_046
            and development["Date"].min() == pd.Timestamp("2016-10-03")
            and development["Date"].max() == pd.Timestamp("2024-11-18")
            and development["Target_Date"].max()
            == pd.Timestamp("2024-11-25")
            and len(audit) == 407
            and audit["Date"].min() == AUDIT_START
            and audit["Date"].max() == pd.Timestamp("2026-07-15")
            and audit.iloc[0]["Sequence_End"] == audit.iloc[0]["Date"],
            (
                f"development={len(development)}, "
                f"{development['Date'].min().date()}.."
                f"{development['Date'].max().date()}; "
                f"audit={len(audit)}, "
                f"{audit['Date'].min().date()}..{audit['Date'].max().date()}"
            ),
        )
    )

    outer_manifest = pd.read_csv(
        OUTPUTS / "rq3_outer_folds.csv",
        parse_dates=[
            "Train Start",
            "Train End",
            "Train Target End",
            "Test Start",
            "Test End",
        ],
    )
    splitter = TimeSeriesSplit(
        n_splits=OUTER_SPLITS,
        test_size=OUTER_TEST_SIZE,
        gap=PURGE_ROWS,
    )
    independent_outer = []
    for fold, (train_local, test_local) in enumerate(
        splitter.split(prepared.development_indices),
        start=1,
    ):
        train = development.iloc[train_local]
        test = development.iloc[test_local]
        independent_outer.append(
            (
                fold,
                len(train),
                train["Date"].max(),
                train["Target_Date"].max(),
                test["Date"].min(),
                test["Date"].max(),
                len(test),
            )
        )
    saved_outer = [
        (
            int(row["Fold"]),
            int(row["Train Rows"]),
            row["Train End"],
            row["Train Target End"],
            row["Test Start"],
            row["Test End"],
            int(row["Test Rows"]),
        )
        for _, row in outer_manifest.iterrows()
    ]
    rows.append(
        check(
            "Five outer folds independently reproduce with gap five",
            independent_outer == saved_outer
            and (
                outer_manifest["Train Target End"]
                < outer_manifest["Test Start"]
            ).all(),
            repr(
                [
                    (
                        fold,
                        train_rows,
                        str(train_end.date()),
                        str(target_end.date()),
                        str(test_start.date()),
                        str(test_end.date()),
                        test_rows,
                    )
                    for (
                        fold,
                        train_rows,
                        train_end,
                        target_end,
                        test_start,
                        test_end,
                        test_rows,
                    ) in independent_outer
                ]
            ),
        )
    )

    inner_manifest = pd.read_csv(
        OUTPUTS / "rq3_inner_folds.csv",
        parse_dates=[
            "Train Start",
            "Train End",
            "Train Target End",
            "Validation Start",
            "Validation End",
        ],
    )
    rows.append(
        check(
            "All six training scopes use three common purged inner folds",
            len(inner_manifest) == 18
            and inner_manifest.groupby("Scope")["Inner Fold"].nunique().eq(3).all()
            and inner_manifest["Validation Rows"].eq(125).all()
            and (
                inner_manifest["Train Target End"]
                < inner_manifest["Validation Start"]
            ).all()
            and inner_manifest["Validation End"].lt(AUDIT_START).all(),
            (
                f"rows={len(inner_manifest)}; "
                f"scopes={inner_manifest['Scope'].nunique()}; "
                f"max_validation={inner_manifest['Validation End'].max().date()}"
            ),
        )
    )

    tuning = pd.read_csv(OUTPUTS / "rq3_inner_tuning_results.csv")
    candidate_counts = {
        "Lagged Logistic Regression": 3,
        "Random Forest": 4,
        "XGBoost": 4,
        "Compact LSTM": 4,
    }
    tuning_complete = len(tuning) == 270
    for (scope, model), group in tuning.groupby(["Scope", "Model"]):
        tuning_complete &= (
            group["Candidate"].nunique() == candidate_counts[model]
            and group.groupby("Candidate")["Inner Fold"].nunique().eq(3).all()
            and group.loc[group["Selected Candidate"], "Candidate"].nunique() == 1
        )
    rows.append(
        check(
            "Bounded inner tuning preserves every candidate and validation result",
            tuning_complete
            and pd.to_datetime(tuning["Validation End"]).lt(AUDIT_START).all(),
            (
                f"rows={len(tuning)}; scopes={tuning['Scope'].nunique()}; "
                f"models={tuning['Model'].nunique()}"
            ),
        )
    )
    lstm_rows = tuning.loc[tuning["Model"].eq("Compact LSTM")]
    rows.append(
        check(
            "LSTM early stopping is confined to inner validation",
            lstm_rows["Best Epoch"].between(1, 100).all()
            and lstm_rows["Epochs Run"].between(1, 100).all()
            and pd.to_datetime(lstm_rows["Preprocessor Fit End"]).le(
                pd.to_datetime(lstm_rows["Train End"])
            ).all(),
            (
                f"best_epoch_range="
                f"{int(lstm_rows['Best Epoch'].min())}.."
                f"{int(lstm_rows['Best Epoch'].max())}"
            ),
        )
    )

    predictions = pd.read_csv(
        OUTPUTS / "rq3_all_predictions.csv",
        parse_dates=["Date", "Target_Date"],
    )
    oof = predictions.loc[predictions["Period"].eq("Development OOF")]
    audit_predictions = predictions.loc[
        predictions["Period"].eq("Retrospective audit")
    ]
    oof_counts = oof.groupby("Comparator")["Date"].nunique()
    audit_counts = audit_predictions.groupby("Comparator")["Date"].nunique()
    oof_dates_match, oof_date_count = sets_match_by_group(oof, "Comparator")
    audit_dates_match, audit_date_count = sets_match_by_group(
        audit_predictions, "Comparator"
    )
    rows.append(
        check(
            "Every model, bridge, and baseline uses identical OOF and audit origins",
            set(oof_counts.index) == EXPECTED_COMPARATORS
            and set(audit_counts.index) == EXPECTED_COMPARATORS
            and oof_counts.eq(1_250).all()
            and audit_counts.eq(407).all()
            and oof_dates_match
            and audit_dates_match,
            (
                f"comparators={len(oof_counts)}; OOF={oof_date_count}; "
                f"audit={audit_date_count}"
            ),
        )
    )
    duplicate_count = int(
        predictions.duplicated(["Period", "Comparator", "Date"]).sum()
    )
    probability_ok = (
        np.isfinite(predictions["Probability"]).all()
        and predictions["Probability"].between(0, 1).all()
        and (
            predictions["Prediction"]
            == (predictions["Probability"] >= THRESHOLD).astype(int)
        ).all()
    )
    rows.append(
        check(
            "Prediction ledger is unique, finite, bounded, and threshold-consistent",
            duplicate_count == 0 and probability_ok,
            (
                f"duplicates={duplicate_count}; "
                f"probability_range="
                f"{predictions['Probability'].min():.9f}.."
                f"{predictions['Probability'].max():.9f}"
            ),
        )
    )

    fold_metrics = pd.read_csv(OUTPUTS / "rq3_fold_metrics.csv")
    metric_mismatches: list[str] = []
    for (period, comparator, fold), frame in predictions.groupby(
        ["Period", "Comparator", "Fold"],
        sort=False,
    ):
        row = fold_metrics.loc[
            fold_metrics["Period"].eq(period)
            & fold_metrics["Comparator"].eq(comparator)
            & fold_metrics["Fold"].astype(str).eq(str(fold))
        ]
        if len(row) != 1:
            metric_mismatches.append(
                f"{period}/{comparator}/{fold}: rows={len(row)}"
            )
            continue
        accuracy, balanced, macro_f1 = metric_triplet(frame)
        saved = row.iloc[0]
        if not (
            np.isclose(saved["Accuracy"], accuracy)
            and np.isclose(saved["Balanced Accuracy"], balanced)
            and np.isclose(saved["Macro F1"], macro_f1)
        ):
            metric_mismatches.append(f"{period}/{comparator}/{fold}")
    rows.append(
        check(
            "Fold and audit metrics independently recompute from predictions",
            not metric_mismatches and len(fold_metrics) == 48,
            (
                "No mismatches."
                if not metric_mismatches
                else " | ".join(metric_mismatches)
            ),
        )
    )

    summary = pd.read_csv(OUTPUTS / "rq3_performance_summary.csv")
    summary_mismatches: list[str] = []
    for comparator in EXPECTED_COMPARATORS:
        row = summary.loc[summary["Comparator"].eq(comparator)]
        if len(row) != 1:
            summary_mismatches.append(f"{comparator}: rows={len(row)}")
            continue
        saved = row.iloc[0]
        for period, prefix in [
            ("Development OOF", "Development Pooled"),
            ("Retrospective audit", "Audit"),
        ]:
            frame = predictions.loc[
                predictions["Comparator"].eq(comparator)
                & predictions["Period"].eq(period)
            ]
            accuracy, balanced, macro_f1 = metric_triplet(frame)
            if not (
                np.isclose(saved[f"{prefix} Accuracy"], accuracy)
                and np.isclose(
                    saved[f"{prefix} Balanced Accuracy"], balanced
                )
                and np.isclose(saved[f"{prefix} Macro F1"], macro_f1)
            ):
                summary_mismatches.append(f"{comparator}/{period}")
    rows.append(
        check(
            "Pooled performance summary independently recomputes",
            not summary_mismatches,
            (
                "No mismatches."
                if not summary_mismatches
                else " | ".join(summary_mismatches)
            ),
        )
    )

    upstream_oof = pd.read_csv(UPSTREAM_OOF, parse_dates=["Date"])
    upstream_audit = pd.read_csv(UPSTREAM_AUDIT, parse_dates=["Date"])
    family = "Technical — EMA-aware compact"
    upstream_oof = upstream_oof.loc[
        upstream_oof["Feature Family"].eq(family)
    ].sort_values("Date")
    upstream_audit = upstream_audit.loc[
        upstream_audit["Feature Family"].eq(family)
    ].sort_values("Date")
    bridge_oof = oof.loc[
        oof["Comparator"].eq("RQ2 current-day Logistic Regression")
    ].sort_values("Date")
    bridge_audit = audit_predictions.loc[
        audit_predictions["Comparator"].eq(
            "RQ2 current-day Logistic Regression"
        )
    ].sort_values("Date")
    bridge_matches = (
        upstream_oof["Date"].reset_index(drop=True).equals(
            bridge_oof["Date"].reset_index(drop=True)
        )
        and upstream_audit["Date"].reset_index(drop=True).equals(
            bridge_audit["Date"].reset_index(drop=True)
        )
        and np.array_equal(
            upstream_oof["Actual"].astype(int).to_numpy(),
            bridge_oof["Actual"].astype(int).to_numpy(),
        )
        and np.array_equal(
            upstream_audit["Actual"].astype(int).to_numpy(),
            bridge_audit["Actual"].astype(int).to_numpy(),
        )
        and np.allclose(
            upstream_oof["Probability"].to_numpy(),
            bridge_oof["Probability"].to_numpy(),
            atol=1e-12,
            rtol=0,
        )
        and np.allclose(
            upstream_audit["Probability"].to_numpy(),
            bridge_audit["Probability"].to_numpy(),
            atol=1e-12,
            rtol=0,
        )
    )
    rows.append(
        check(
            "RQ2 current-day bridge exactly reproduces the frozen upstream predictions",
            bridge_matches,
            (
                f"OOF rows={len(bridge_oof)}; audit rows={len(bridge_audit)}; "
                f"exact_probability_tolerance=1e-12"
            ),
        )
    )

    seed_predictions = pd.read_csv(
        OUTPUTS / "rq3_seed_level_predictions.csv",
        parse_dates=["Date"],
    )
    ensemble_mismatches: list[str] = []
    for model in STOCHASTIC_MODELS:
        model_seed = seed_predictions.loc[
            seed_predictions["Model"].eq(model)
        ]
        counts = model_seed.groupby(["Period", "Date"])["Seed"].nunique()
        if not counts.eq(3).all():
            ensemble_mismatches.append(f"{model}: non-three-seed rows")
            continue
        mean_probability = (
            model_seed.groupby(["Period", "Date"], as_index=False)[
                "Probability"
            ]
            .mean()
            .rename(columns={"Probability": "Seed Mean"})
        )
        saved = predictions.loc[
            predictions["Comparator"].eq(model),
            ["Period", "Date", "Probability"],
        ]
        merged = saved.merge(
            mean_probability,
            on=["Period", "Date"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != 1_657 or not np.allclose(
            merged["Probability"],
            merged["Seed Mean"],
            atol=1e-12,
            rtol=0,
        ):
            ensemble_mismatches.append(model)
    rows.append(
        check(
            "RF, XGBoost, and LSTM probabilities equal the saved three-seed means",
            not ensemble_mismatches,
            (
                "No mismatches."
                if not ensemble_mismatches
                else " | ".join(ensemble_mismatches)
            ),
        )
    )

    selection = pd.read_csv(OUTPUTS / "rq3_one_se_selection.csv")
    best = selection.sort_values(
        ["Mean_Balanced_Accuracy", "Complexity Rank"],
        ascending=[False, True],
    ).iloc[0]
    independent_se = float(
        best["SD_Balanced_Accuracy"] / np.sqrt(OUTER_SPLITS)
    )
    independent_cutoff = float(
        best["Mean_Balanced_Accuracy"] - independent_se
    )
    independent_eligible = selection["Mean_Balanced_Accuracy"].ge(
        independent_cutoff
    )
    independent_selected = selection.loc[independent_eligible].sort_values(
        "Complexity Rank"
    ).iloc[0]["Model"]
    saved_selected = selection.loc[
        selection["Selected for RQ4"], "Model"
    ].iloc[0]
    rows.append(
        check(
            "Development-only one-SE selection independently recomputes",
            independent_selected == saved_selected
            and np.isclose(
                selection["Best Model Standard Error"].iloc[0],
                independent_se,
            )
            and np.isclose(
                selection["One-SE Cutoff"].iloc[0],
                independent_cutoff,
            )
            and selection["Audit Used"].eq(False).all()
            and selection["RQ4 Returns Used"].eq(False).all(),
            (
                f"best_mean_model={best['Model']}; SE={independent_se:.9f}; "
                f"cutoff={independent_cutoff:.9f}; "
                f"selected={independent_selected}"
            ),
        )
    )

    configurations = pd.read_csv(
        OUTPUTS / "rq3_selected_configurations.csv"
    )
    final_configurations = configurations.loc[
        configurations["Scope"].eq("Full development")
    ]
    rows.append(
        check(
            "Every model has one development-only final audit configuration",
            set(final_configurations["Model"]) == set(MAIN_MODELS)
            and len(final_configurations) == 4
            and final_configurations["Audit Used"].eq(False).all()
            and final_configurations.loc[
                final_configurations["Model"].eq("Compact LSTM"),
                "Selected Epoch",
            ].between(1, 100).all(),
            final_configurations[
                ["Model", "Selected Configuration", "Selected Epoch"]
            ].to_json(orient="records"),
        )
    )

    phases = pd.read_csv(OUTPUTS / "rq3_phase_offset_metrics.csv")
    phase_ok = True
    for (period, comparator), group in phases.groupby(
        ["Period", "Comparator"]
    ):
        phase_ok &= group["Phase Offset"].nunique() == 5
        if period == "Development OOF":
            phase_ok &= group["Rows"].eq(250).all()
            phase_ok &= int(group["Rows"].sum()) == 1_250
        else:
            phase_ok &= sorted(group["Rows"].tolist()) == [81, 81, 81, 82, 82]
            phase_ok &= int(group["Rows"].sum()) == 407
    rows.append(
        check(
            "Five immutable phase-offset evaluations cover every saved prediction",
            phase_ok
            and set(phases["Comparator"]) == EXPECTED_COMPARATORS,
            (
                f"rows={len(phases)}; comparators="
                f"{phases['Comparator'].nunique()}"
            ),
        )
    )

    bootstrap = pd.read_csv(
        OUTPUTS / "rq3_paired_block_bootstrap.csv"
    )
    primary = bootstrap.loc[
        bootstrap["Comparison Family"].eq(
            "Primary complex-versus-linear"
        )
    ]
    secondary = bootstrap.loc[
        bootstrap["Comparison Family"].eq(
            "Secondary model-versus-transparent-baseline"
        )
    ]
    point_mismatches: list[str] = []
    for _, values in bootstrap.iterrows():
        frame = predictions.loc[
            predictions["Period"].eq(values["Period"])
        ]
        actual_a = frame.loc[
            frame["Comparator"].eq(values["Comparator A"])
        ].sort_values("Date")
        actual_b = frame.loc[
            frame["Comparator"].eq(values["Comparator B"])
        ].sort_values("Date")
        if values["Metric"] == "Accuracy":
            point = accuracy_score(
                actual_a["Actual"], actual_a["Prediction"]
            ) - accuracy_score(actual_b["Actual"], actual_b["Prediction"])
        else:
            point = balanced_accuracy_score(
                actual_a["Actual"], actual_a["Prediction"]
            ) - balanced_accuracy_score(
                actual_b["Actual"], actual_b["Prediction"]
            )
        if not np.isclose(point, values["Point Difference"]):
            point_mismatches.append(
                f"{values['Period']}/{values['Metric']}/"
                f"{values['Comparator A']}/{values['Comparator B']}"
            )
    rows.append(
        check(
            "All pre-specified paired block-bootstrap contrasts and point estimates exist",
            len(primary) == 12
            and len(secondary) == 32
            and bootstrap["Bootstrap Replicates"].eq(5_000).all()
            and bootstrap["Block Length"].eq(20).all()
            and bootstrap["Holm-adjusted p"].between(0, 1).all()
            and not point_mismatches,
            (
                f"primary={len(primary)}; secondary={len(secondary)}; "
                f"point_mismatches={len(point_mismatches)}"
            ),
        )
    )

    freeze_manifest_path = FREEZE / "rq3_freeze_manifest.json"
    freeze_manifest = json.loads(
        freeze_manifest_path.read_text(encoding="utf-8")
    )
    freeze_hashes_match = all(
        file_sha256(ROOT / relative_path) == expected_hash
        for relative_path, expected_hash in freeze_manifest["files"].items()
    )
    selected_model = freeze_manifest["selected_model"]
    selected_oof = pd.read_csv(
        FREEZE / "rq3_selected_oof_predictions.csv",
        parse_dates=["Date"],
    )
    selected_audit = pd.read_csv(
        FREEZE / "rq3_selected_audit_predictions.csv",
        parse_dates=["Date"],
    )
    source_selected_oof = oof.loc[
        oof["Comparator"].eq(selected_model)
    ].sort_values("Date")
    source_selected_audit = audit_predictions.loc[
        audit_predictions["Comparator"].eq(selected_model)
    ].sort_values("Date")
    selected_prediction_match = (
        selected_oof["Date"].reset_index(drop=True).equals(
            source_selected_oof["Date"].reset_index(drop=True)
        )
        and selected_audit["Date"].reset_index(drop=True).equals(
            source_selected_audit["Date"].reset_index(drop=True)
        )
        and np.allclose(
            selected_oof["Probability"],
            source_selected_oof["Probability"],
            atol=0,
            rtol=0,
        )
        and np.allclose(
            selected_audit["Probability"],
            source_selected_audit["Probability"],
            atol=0,
            rtol=0,
        )
    )
    rows.append(
        check(
            "Frozen selected predictions and manifest hashes match the reviewed ledger",
            freeze_hashes_match
            and selected_model == saved_selected
            and selected_prediction_match
            and freeze_manifest["protocol_sha256"] == file_sha256(PROTOCOL),
            (
                f"selected={selected_model}; files="
                f"{len(freeze_manifest['files'])}; "
                f"hashes_match={freeze_hashes_match}"
            ),
        )
    )

    rq4_signal = pd.read_csv(
        FREEZE / "rq3_frozen_rq4_signal.csv",
        parse_dates=["Date"],
    )
    rows.append(
        check(
            "Frozen RQ4 signal contains exactly the selected OOF and audit decisions",
            len(rq4_signal) == 1_657
            and rq4_signal["Date"].nunique() == 1_657
            and rq4_signal["Selected Model"].eq(selected_model).all()
            and rq4_signal["Threshold"].eq(THRESHOLD).all()
            and rq4_signal["Protocol SHA256"].eq(file_sha256(PROTOCOL)).all()
            and (
                rq4_signal["Binary Signal"]
                == (rq4_signal["Probability"] >= THRESHOLD).astype(int)
            ).all(),
            (
                f"rows={len(rq4_signal)}; periods="
                f"{rq4_signal['Period'].value_counts().to_dict()}"
            ),
        )
    )

    preprocessor = json.loads(
        (FREEZE / "rq3_selected_preprocessor.json").read_text(
            encoding="utf-8"
        )
    )
    selected_metadata = json.loads(
        (FREEZE / "rq3_selected_model.json").read_text(encoding="utf-8")
    )
    audit_tensor = prepared.sequences[prepared.audit_indices].copy()
    flat = audit_tensor.reshape(-1, len(FEATURES))
    medians = np.asarray(preprocessor["imputer_statistics"], dtype=float)
    missing = np.isnan(flat)
    if missing.any():
        flat[missing] = np.take(medians, np.where(missing)[1])
    means = np.asarray(preprocessor["scaler_mean"], dtype=float)
    scales = np.asarray(preprocessor["scaler_scale"], dtype=float)
    scaled = (
        ((flat - means) / scales)
        .reshape(audit_tensor.shape)
        .astype(np.float32)
    )
    model_paths = sorted(
        FREEZE.glob("rq3_selected_model_seed_*.joblib")
    )
    reconstructed_probability: np.ndarray | None = None
    if model_paths:
        probabilities = []
        x_model = (
            scaled
            if selected_model == "Compact LSTM"
            else scaled.reshape(len(scaled), -1)
        )
        for path in model_paths:
            model = joblib.load(path)
            probabilities.append(model.predict_proba(x_model)[:, 1])
        reconstructed_probability = np.mean(
            np.vstack(probabilities),
            axis=0,
        )
    model_reproduces = (
        reconstructed_probability is not None
        and np.allclose(
            reconstructed_probability,
            selected_audit.sort_values("Date")["Probability"].to_numpy(),
            atol=1e-12,
            rtol=0,
        )
    )
    rows.append(
        check(
            "Frozen selected estimator and preprocessing reproduce audit probabilities",
            selected_metadata["selected_model"] == selected_model
            and preprocessor["fit_end"] == "2024-11-18"
            and model_reproduces,
            (
                f"selected={selected_model}; preprocessor_fit_end="
                f"{preprocessor['fit_end']}; reproduced={model_reproduces}"
            ),
        )
    )

    notebook_status_path = OUTPUTS / "rq3_notebook_validation.json"
    notebook_status = (
        json.loads(notebook_status_path.read_text(encoding="utf-8"))
        if notebook_status_path.exists()
        else {}
    )
    rows.append(
        check(
            "Reader-facing notebook executed top to bottom",
            notebook_status.get("status") == "executed"
            and notebook_status.get("code_cells_executed", 0) > 0
            and (ROOT / "notebooks" / "v8_3_rq3_model_comparison.ipynb").exists(),
            json.dumps(notebook_status, sort_keys=True),
        )
    )

    report = pd.DataFrame(rows)
    summary_report = {
        "status": "passed" if report["Status"].eq("Pass").all() else "failed",
        "checks": len(report),
        "passed": int(report["Status"].eq("Pass").sum()),
        "failed": int(report["Status"].eq("Fail").sum()),
        "artifacts_written": bool(write_artifacts),
    }
    if write_artifacts:
        report.to_csv(OUTPUTS / "rq3_validation_checks.csv", index=False)
        (OUTPUTS / "rq3_validation_report.json").write_text(
            json.dumps(summary_report, indent=2),
            encoding="utf-8",
        )
    print(report.to_string(index=False))
    print(json.dumps(summary_report, indent=2))
    if summary_report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate saved RQ3 evidence without retraining. The default mode "
            "is read-only."
        )
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Replace the saved validation CSV and JSON after recomputation.",
    )
    arguments = parser.parse_args()
    main(write_artifacts=arguments.write_artifacts)
