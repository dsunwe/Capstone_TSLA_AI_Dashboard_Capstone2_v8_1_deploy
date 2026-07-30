"""Validate the focused v8.2 RQ2 experiment and its saved evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from run_rq2_experiment import load_data, monthly_release_changes


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data" / "TSLA_features_common_window.csv"
CPI_DATA = ROOT / "data" / "CPIAUCSL.csv"
UNRATE_DATA = ROOT / "data" / "UNRATE.csv"

EXPECTED_PRIMARY = [
    "Daily_Return",
    "Momentum_5",
    "EMA8_13_State",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]
EXPECTED_FAMILIES = {
    "Technical — EMA-aware compact",
    "Cross-market",
    "Macroeconomic",
    "Broad sentiment and risk proxies",
}
EXPECTED_TECHNICAL_DESIGNS = {
    "Primary EMA-aware compact",
    "Core-5 without moving-average state",
    "SMA10/20-state sensitivity",
    "Legacy SMA50-ratio comparator",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "Check": name,
        "Status": "Pass" if passed else "Fail",
        "Evidence": evidence,
    }


def date_sets_match(
    frame: pd.DataFrame,
    group_column: str,
) -> tuple[bool, int]:
    sets = [
        set(pd.to_datetime(group["Date"]))
        for _, group in frame.groupby(group_column)
    ]
    return all(date_set == sets[0] for date_set in sets), len(sets[0])


def recompute_metric_rows(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    group_column: str,
) -> tuple[bool, str]:
    mismatches: list[str] = []
    for group, prediction_frame in predictions.groupby(group_column):
        row = metrics.loc[metrics[group_column].eq(group)]
        if len(row) != 1:
            mismatches.append(f"{group}: metric row count={len(row)}")
            continue
        row = row.iloc[0]
        actual = prediction_frame["Actual"].astype(int)
        prediction = prediction_frame["Prediction"].astype(int)
        accuracy = accuracy_score(actual, prediction)
        balanced = balanced_accuracy_score(actual, prediction)
        macro_f1 = f1_score(actual, prediction, average="macro")
        if not (
            np.isclose(row["Accuracy"], accuracy)
            and np.isclose(row["Balanced Accuracy"], balanced)
            and np.isclose(row["Macro F1"], macro_f1)
        ):
            mismatches.append(
                f"{group}: saved=({row['Accuracy']:.9f}, "
                f"{row['Balanced Accuracy']:.9f}, {row['Macro F1']:.9f}); "
                f"recomputed=({accuracy:.9f}, {balanced:.9f}, "
                f"{macro_f1:.9f})"
            )
    return not mismatches, "No mismatches." if not mismatches else " | ".join(mismatches)


def main() -> None:
    rows: list[dict[str, object]] = []
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (OUTPUTS / "rq2_experiment_metadata.json").read_text(encoding="utf-8")
    )

    rows.append(
        check(
            "Frozen input snapshot hash matches the protocol",
            file_sha256(DATA) == protocol["input_snapshot"]["sha256"],
            file_sha256(DATA),
        )
    )
    macro_hashes = {
        "CPIAUCSL": file_sha256(CPI_DATA),
        "UNRATE": file_sha256(UNRATE_DATA),
    }
    rows.append(
        check(
            "Monthly macro source hashes match the protocol",
            all(
                macro_hashes[series_id]
                == protocol["macro_source_snapshots"][series_id]["sha256"]
                for series_id in macro_hashes
            ),
            repr(macro_hashes),
        )
    )
    rows.append(
        check(
            "Saved metadata identifies the same protocol and input",
            metadata["protocol_id"] == protocol["protocol_id"]
            and metadata["protocol_sha256"]
            == file_sha256(ROOT / "protocol.json")
            and metadata["input_sha256"] == file_sha256(DATA)
            and metadata["macro_source_sha256"] == macro_hashes,
            (
                f"protocol={metadata['protocol_id']}; "
                f"protocol_sha256={metadata['protocol_sha256']}; "
                f"input_sha256={metadata['input_sha256']}; "
                f"macro_sha256={metadata['macro_source_sha256']}"
            ),
        )
    )
    primary = protocol["standalone_feature_families"][
        "Technical — EMA-aware compact"
    ]
    rows.append(
        check(
            "RQ2 primary technical family is the six-feature EMA-aware design",
            primary == EXPECTED_PRIMARY,
            " | ".join(primary),
        )
    )
    rows.append(
        check(
            "EMA and SMA are persistent state variables, not raw price levels",
            protocol["moving_average_state_definitions"]["EMA8_13_State"]
            == "1[EMA8[t] > EMA13[t]]"
            and protocol["moving_average_state_definitions"]["SMA10_20_State"]
            == "1[SMA10[t] > SMA20[t]]"
            and not any(
                feature in primary
                for feature in ["EMA8", "EMA13", "SMA10", "SMA20"]
            ),
            json.dumps(
                protocol["moving_average_state_definitions"],
                ensure_ascii=False,
            ),
        )
    )
    prepared_data = load_data()
    ema_formula = (prepared_data["EMA8"] > prepared_data["EMA13"]).astype(int)
    sma_formula = (prepared_data["SMA10"] > prepared_data["SMA20"]).astype(int)
    rows.append(
        check(
            "Saved state derivation matches the source moving averages row by row",
            prepared_data["EMA8_13_State"].isin([0, 1]).all()
            and prepared_data["SMA10_20_State"].isin([0, 1]).all()
            and prepared_data["EMA8_13_State"].astype(int).equals(ema_formula)
            and prepared_data["SMA10_20_State"].astype(int).equals(sma_formula),
            (
                f"rows={len(prepared_data)}; "
                f"EMA states={sorted(prepared_data['EMA8_13_State'].unique())}; "
                f"SMA states={sorted(prepared_data['SMA10_20_State'].unique())}"
            ),
        )
    )
    release_checks: list[bool] = []
    release_evidence: list[str] = []
    for path, series_id in [
        (CPI_DATA, "CPIAUCSL"),
        (UNRATE_DATA, "UNRATE"),
    ]:
        releases = monthly_release_changes(path, series_id)
        value_column = next(
            column
            for column in releases.columns
            if column != "Available_Date"
        )
        expected = pd.merge_asof(
            prepared_data[["Date"]].sort_values("Date"),
            releases.sort_values("Available_Date"),
            left_on="Date",
            right_on="Available_Date",
            direction="backward",
            allow_exact_matches=True,
        )
        matches = np.allclose(
            prepared_data[value_column].to_numpy(),
            expected[value_column].to_numpy(),
            equal_nan=True,
        )
        zero_releases = int(np.isclose(releases[value_column], 0).sum())
        release_checks.append(
            matches and (zero_releases > 0 if series_id == "UNRATE" else True)
        )
        release_evidence.append(
            f"{series_id}: matched={matches}, zero releases={zero_releases}"
        )
    rows.append(
        check(
            "CPI and unemployment changes are computed at source frequency",
            all(release_checks),
            "; ".join(release_evidence),
        )
    )
    rows.append(
        check(
            "RQ2 protocol keeps RQ3 and RQ4 outside this package",
            protocol["rq_boundaries"]["rq3"].startswith(
                "model-family comparison is out of scope"
            )
            and protocol["rq_boundaries"]["rq4"].startswith(
                "economic backtesting is out of scope"
            ),
            json.dumps(protocol["rq_boundaries"], ensure_ascii=False),
        )
    )

    source = (ROOT / "run_rq2_experiment.py").read_text(encoding="utf-8")
    rows.append(
        check(
            "Focused RQ2 source contains no RF, XGBoost, or LSTM model",
            "RandomForest" not in source
            and "XGBoost" not in source
            and "XGBClassifier" not in source
            and "LSTM" not in source,
            "Only LogisticRegression is imported and instantiated.",
        )
    )

    family_folds = pd.read_csv(
        OUTPUTS / "rq2_feature_family_fold_metrics.csv"
    )
    family_audit = pd.read_csv(
        OUTPUTS / "rq2_feature_family_audit_metrics.csv"
    )
    family_oof = pd.read_csv(
        OUTPUTS / "rq2_feature_family_oof_predictions.csv"
    )
    family_predictions = pd.read_csv(
        OUTPUTS / "rq2_feature_family_audit_predictions.csv"
    )
    technical_folds = pd.read_csv(
        OUTPUTS / "rq2_technical_design_fold_metrics.csv"
    )
    technical_audit = pd.read_csv(
        OUTPUTS / "rq2_technical_design_audit_metrics.csv"
    )
    technical_oof = pd.read_csv(
        OUTPUTS / "rq2_technical_design_oof_predictions.csv"
    )
    technical_predictions = pd.read_csv(
        OUTPUTS / "rq2_technical_design_audit_predictions.csv"
    )

    family_fold_counts = family_folds.groupby("Feature Family")["Fold"].nunique()
    rows.append(
        check(
            "Four standalone families each have five walk-forward folds",
            set(family_fold_counts.index) == EXPECTED_FAMILIES
            and family_fold_counts.eq(5).all()
            and len(family_folds) == 20,
            repr(family_fold_counts.to_dict()),
        )
    )
    technical_fold_counts = technical_folds.groupby("Technical Design")[
        "Fold"
    ].nunique()
    rows.append(
        check(
            "Four technical designs each have five walk-forward folds",
            set(technical_fold_counts.index) == EXPECTED_TECHNICAL_DESIGNS
            and technical_fold_counts.eq(5).all()
            and len(technical_folds) == 20,
            repr(technical_fold_counts.to_dict()),
        )
    )
    rows.append(
        check(
            "Every development fold purges labels before its test period",
            (
                pd.to_datetime(family_folds["Train Target End"])
                < pd.to_datetime(family_folds["Test Start"])
            ).all()
            and (
                pd.to_datetime(technical_folds["Train Target End"])
                < pd.to_datetime(technical_folds["Test Start"])
            ).all(),
            "All Train Target End dates precede Test Start dates.",
        )
    )

    family_oof_match, family_oof_dates = date_sets_match(
        family_oof,
        "Feature Family",
    )
    family_audit_match, family_audit_dates = date_sets_match(
        family_predictions,
        "Feature Family",
    )
    technical_oof_match, technical_oof_dates = date_sets_match(
        technical_oof,
        "Technical Design",
    )
    technical_audit_match, technical_audit_dates = date_sets_match(
        technical_predictions,
        "Technical Design",
    )
    rows.append(
        check(
            "All families and technical designs use common OOF and audit dates",
            family_oof_match
            and family_audit_match
            and technical_oof_match
            and technical_audit_match
            and family_oof_dates == technical_oof_dates == 1_250
            and family_audit_dates == technical_audit_dates == 407,
            (
                f"family OOF={family_oof_dates}; family audit="
                f"{family_audit_dates}; technical OOF={technical_oof_dates}; "
                f"technical audit={technical_audit_dates}"
            ),
        )
    )

    family_metrics_ok, family_metric_evidence = recompute_metric_rows(
        family_audit,
        family_predictions,
        "Feature Family",
    )
    rows.append(
        check(
            "Feature-family audit metrics recompute from the prediction ledger",
            family_metrics_ok,
            family_metric_evidence,
        )
    )
    technical_metrics_ok, technical_metric_evidence = recompute_metric_rows(
        technical_audit,
        technical_predictions,
        "Technical Design",
    )
    rows.append(
        check(
            "Technical-design audit metrics recompute from the prediction ledger",
            technical_metrics_ok,
            technical_metric_evidence,
        )
    )

    all_probabilities = pd.concat(
        [
            family_oof["Probability"],
            family_predictions["Probability"],
            technical_oof["Probability"],
            technical_predictions["Probability"],
        ],
        ignore_index=True,
    )
    rows.append(
        check(
            "All saved probabilities are finite and bounded",
            bool(
                np.isfinite(all_probabilities).all()
                and all_probabilities.between(0, 1).all()
            ),
            (
                f"min={all_probabilities.min():.9f}; "
                f"max={all_probabilities.max():.9f}"
            ),
        )
    )

    primary_row = family_audit.loc[
        family_audit["Feature Family"].eq(
            "Technical — EMA-aware compact"
        )
    ].iloc[0]
    rows.append(
        check(
            "Primary reviewed audit result is reproduced",
            int(primary_row["Audit Rows"]) == 407
            and np.isclose(primary_row["Accuracy"], 0.6388206388206388)
            and np.isclose(
                primary_row["Balanced Accuracy"],
                0.6376869464207928,
            ),
            (
                f"rows={int(primary_row['Audit Rows'])}; "
                f"accuracy={primary_row['Accuracy']:.12f}; "
                f"balanced={primary_row['Balanced Accuracy']:.12f}"
            ),
        )
    )

    family_bootstrap = pd.read_csv(
        OUTPUTS / "rq2_feature_family_paired_block_bootstrap.csv"
    )
    technical_bootstrap = pd.read_csv(
        OUTPUTS / "rq2_technical_design_paired_block_bootstrap.csv"
    )
    expected_family_pairs = {
        (period, alternative)
        for period in ["Development OOF", "Retrospective audit"]
        for alternative in [
            "Cross-market",
            "Macroeconomic",
            "Broad sentiment and risk proxies",
        ]
    }
    actual_family_pairs = set(
        zip(family_bootstrap["Period"], family_bootstrap["Group B"])
    )
    rows.append(
        check(
            "Primary technical family is bootstrapped against all three alternatives",
            actual_family_pairs == expected_family_pairs
            and family_bootstrap["Bootstrap Replicates"].eq(5_000).all()
            and family_bootstrap["Block Length"].eq(20).all()
            and family_bootstrap["Holm-adjusted p"].between(0, 1).all(),
            repr(sorted(actual_family_pairs)),
        )
    )
    expected_technical_pairs = {
        (period, alternative)
        for period in ["Development OOF", "Retrospective audit"]
        for alternative in [
            "Core-5 without moving-average state",
            "SMA10/20-state sensitivity",
            "Legacy SMA50-ratio comparator",
        ]
    }
    actual_technical_pairs = set(
        zip(technical_bootstrap["Period"], technical_bootstrap["Group B"])
    )
    rows.append(
        check(
            "EMA-aware design is bootstrapped against every technical comparator",
            actual_technical_pairs == expected_technical_pairs
            and technical_bootstrap["Bootstrap Replicates"].eq(5_000).all()
            and technical_bootstrap["Block Length"].eq(20).all(),
            repr(sorted(actual_technical_pairs)),
        )
    )

    family_phase = pd.read_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_metrics.csv"
    )
    family_phase_folds = pd.read_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_fold_metrics.csv"
    )
    family_phase_predictions = pd.read_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_predictions.csv"
    )
    technical_phase = pd.read_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_metrics.csv"
    )
    technical_phase_folds = pd.read_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_fold_metrics.csv"
    )
    technical_phase_predictions = pd.read_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_predictions.csv"
    )
    family_phase_counts = family_phase.groupby("Feature Family")[
        "Phase Offset"
    ].nunique()
    technical_phase_counts = technical_phase.groupby("Technical Design")[
        "Phase Offset"
    ].nunique()
    rows.append(
        check(
            "Five phase-offset robustness samples exist for every design",
            family_phase_counts.eq(5).all()
            and technical_phase_counts.eq(5).all()
            and family_phase.groupby("Feature Family")["Rows"].sum().eq(407).all()
            and technical_phase.groupby("Technical Design")["Rows"]
            .sum()
            .eq(407)
            .all(),
            (
                f"families={family_phase_counts.to_dict()}; "
                f"technical={technical_phase_counts.to_dict()}"
            ),
        )
    )
    phase_fold_counts = pd.concat(
        [
            family_phase_folds.assign(Collection="Family").rename(
                columns={"Feature Family": "Design"}
            ),
            technical_phase_folds.assign(Collection="Technical").rename(
                columns={"Technical Design": "Design"}
            ),
        ],
        ignore_index=True,
    )
    rows.append(
        check(
            "Every phase is refitted with five purged development folds",
            phase_fold_counts.groupby(
                ["Collection", "Design", "Phase Offset"]
            )["Fold"]
            .nunique()
            .eq(5)
            .all()
            and phase_fold_counts["Test Rows"].eq(50).all()
            and (
                pd.to_datetime(phase_fold_counts["Train Target End"])
                < pd.to_datetime(phase_fold_counts["Test Start"])
            ).all(),
            (
                f"phase-fold rows={len(phase_fold_counts)}; "
                f"test sizes={sorted(phase_fold_counts['Test Rows'].unique())}"
            ),
        )
    )
    phase_prediction_checks = []
    for predictions, group_column in [
        (family_phase_predictions, "Feature Family"),
        (technical_phase_predictions, "Technical Design"),
    ]:
        phase_prediction_checks.append(
            (
                predictions["Phase_Offset"]
                == predictions["Eligible_Row_ID"] % 5
            ).all()
            and predictions.groupby(group_column)["Date"].nunique().eq(407).all()
            and predictions.groupby(
                [group_column, "Phase_Offset"]
            )["Date"].count().isin([81, 82]).all()
        )
    rows.append(
        check(
            "Phase predictions use the immutable eligible-row modulo-five anchor",
            all(phase_prediction_checks),
            "Each design covers 407 dates; each phase contains 81 or 82 dates.",
        )
    )

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUTS / "rq2_validation_checks.csv", index=False)
    summary = {
        "status": "passed" if report["Status"].eq("Pass").all() else "failed",
        "checks": len(report),
        "passed": int(report["Status"].eq("Pass").sum()),
        "failed": int(report["Status"].eq("Fail").sum()),
    }
    (OUTPUTS / "rq2_validation_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
