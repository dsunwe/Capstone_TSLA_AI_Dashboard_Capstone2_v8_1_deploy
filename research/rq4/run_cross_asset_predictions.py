"""Generate AAPL and NVDA RQ3-compatible predictions for the RQ4 replication.

TSLA predictions are reused from the frozen v8.3 RQ3 experiment.  AAPL and
NVDA are fitted independently with the same target, six feature channels,
20-session representation, nested walk-forward tuning, threshold, seeds, and
development/audit split implemented by the frozen RQ3 engine.

This script deliberately produces predictions only.  It does not inspect or
optimise any RQ4 return metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
ENGINE_PATH = UPSTREAM / "rq3_engine.py"
RQ3_PROTOCOL_PATH = UPSTREAM / "rq3_protocol.json"
OUTPUTS.mkdir(parents=True, exist_ok=True)
UPSTREAM.mkdir(parents=True, exist_ok=True)

ASSETS_TO_FIT = ["AAPL", "NVDA"]
ALL_ASSETS = ["TSLA", *ASSETS_TO_FIT]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_engine():
    spec = importlib.util.spec_from_file_location("frozen_rq3_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen RQ3 engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_asset(asset: str) -> None:
    if asset not in ASSETS_TO_FIT:
        raise ValueError(f"Only {ASSETS_TO_FIT} require new prediction runs.")

    started = time.perf_counter()
    engine = load_engine()
    engine.DATA = DATA / f"{asset}_features_common_window.csv"
    protocol = json.loads(RQ3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    prepared = engine.prepare_data()
    origins = prepared.origins

    splitter = TimeSeriesSplit(
        n_splits=engine.OUTER_SPLITS,
        test_size=engine.OUTER_TEST_SIZE,
        gap=engine.PURGE_ROWS,
    )
    outer_splits = list(splitter.split(prepared.development_indices))
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    configuration_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []

    for fold, (train_local, test_local) in enumerate(outer_splits, start=1):
        train_indices = prepared.development_indices[train_local]
        test_indices = prepared.development_indices[test_local]
        engine.validate_temporal_boundary(
            prepared,
            train_indices,
            test_indices,
            f"{asset} outer fold {fold}",
        )
        train_rows = origins.iloc[train_indices]
        test_rows = origins.iloc[test_indices]
        outer_rows.append(
            {
                "Asset": asset,
                "Fold": fold,
                "Train Start": train_rows["Date"].min(),
                "Train End": train_rows["Date"].max(),
                "Train Target End": train_rows["Target_Date"].max(),
                "Test Start": test_rows["Date"].min(),
                "Test End": test_rows["Date"].max(),
                "Train Rows": len(train_rows),
                "Test Rows": len(test_rows),
                "Gap Rows": engine.PURGE_ROWS,
            }
        )

        for model_name in engine.MAIN_MODELS:
            print(f"{asset} fold {fold}: tuning {model_name}", flush=True)
            (
                selected_config,
                selected_epoch,
                model_tuning_rows,
                _,
            ) = engine.tune_model(
                prepared,
                protocol,
                train_indices,
                model_name,
                scope=f"{asset} outer fold {fold}",
                outer_fold=fold,
            )
            for row in model_tuning_rows:
                row["Asset"] = asset
            tuning_rows.extend(model_tuning_rows)
            configuration_rows.append(
                {
                    "Asset": asset,
                    "Scope": f"Outer fold {fold}",
                    "Outer Fold": fold,
                    "Model": model_name,
                    "Selected Configuration": engine.stable_json(selected_config),
                    "Selected Epoch": selected_epoch,
                    "Selection Metric": "Mean inner-fold Balanced Accuracy",
                    "Audit Used": False,
                    "RQ4 Return Used": False,
                }
            )
            (
                _,
                _,
                probability,
                _,
                elapsed,
            ) = engine.fit_predict_ensemble(
                prepared,
                train_indices,
                test_indices,
                model_name,
                selected_config,
                selected_epoch,
                retain_models=False,
            )
            frame = engine.append_prediction_frame(
                prediction_frames,
                prepared,
                test_indices,
                model_name,
                "Main model",
                "Development OOF",
                fold,
                probability,
            )
            fold_metric_rows.append(
                {
                    "Asset": asset,
                    **engine.fold_metric_row(
                        frame,
                        prepared,
                        train_indices,
                        test_indices,
                        elapsed,
                        selected_config,
                        selected_epoch,
                    ),
                }
            )

    full_development = prepared.development_indices
    audit_indices = prepared.audit_indices
    for model_name in engine.MAIN_MODELS:
        print(f"{asset} full development: tuning {model_name}", flush=True)
        (
            selected_config,
            selected_epoch,
            model_tuning_rows,
            _,
        ) = engine.tune_model(
            prepared,
            protocol,
            full_development,
            model_name,
            scope=f"{asset} full development",
            outer_fold=None,
        )
        for row in model_tuning_rows:
            row["Asset"] = asset
        tuning_rows.extend(model_tuning_rows)
        configuration_rows.append(
            {
                "Asset": asset,
                "Scope": "Full development",
                "Outer Fold": None,
                "Model": model_name,
                "Selected Configuration": engine.stable_json(selected_config),
                "Selected Epoch": selected_epoch,
                "Selection Metric": "Mean inner-fold Balanced Accuracy",
                "Audit Used": False,
                "RQ4 Return Used": False,
            }
        )
        (
            _,
            _,
            probability,
            _,
            elapsed,
        ) = engine.fit_predict_ensemble(
            prepared,
            full_development,
            audit_indices,
            model_name,
            selected_config,
            selected_epoch,
            retain_models=False,
        )
        frame = engine.append_prediction_frame(
            prediction_frames,
            prepared,
            audit_indices,
            model_name,
            "Main model",
            "Retrospective audit",
            "Audit",
            probability,
        )
        fold_metric_rows.append(
            {
                "Asset": asset,
                **engine.fold_metric_row(
                    frame,
                    prepared,
                    full_development,
                    audit_indices,
                    elapsed,
                    selected_config,
                    selected_epoch,
                ),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.insert(0, "Asset", asset)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    configurations = pd.DataFrame(configuration_rows)
    outer_manifest = pd.DataFrame(outer_rows)
    summary = engine.pooled_summary(
        predictions.drop(columns="Asset"),
        fold_metrics.drop(columns="Asset"),
    )
    summary.insert(0, "Asset", asset)

    predictions.to_csv(
        UPSTREAM / f"{asset}_rq3_model_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(OUTPUTS / f"{asset}_rq3_fold_metrics.csv", index=False)
    tuning.to_csv(
        OUTPUTS / f"{asset}_rq3_inner_tuning_results.csv",
        index=False,
    )
    configurations.to_csv(
        OUTPUTS / f"{asset}_rq3_selected_configurations.csv",
        index=False,
    )
    outer_manifest.to_csv(
        OUTPUTS / f"{asset}_rq3_outer_folds.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUTS / f"{asset}_rq3_classification_summary.csv",
        index=False,
    )

    metadata = {
        "asset": asset,
        "status": "retrospective cross-asset replication",
        "prediction_role": "asset-specific RQ3-compatible inputs for RQ4",
        "target": protocol["target"],
        "features": engine.FEATURES,
        "lookback": engine.LOOKBACK,
        "threshold": engine.THRESHOLD,
        "seeds": engine.SEEDS,
        "development_oof_rows_per_model": 1250,
        "retrospective_audit_rows_per_model": 407,
        "input_file": f"data/{asset}_features_common_window.csv",
        "input_sha256": sha256(engine.DATA),
        "rq3_engine_sha256": sha256(ENGINE_PATH),
        "rq3_protocol_sha256": sha256(RQ3_PROTOCOL_PATH),
        "return_metrics_used_for_training_or_selection": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "tensorflow": engine.tensorflow_module().__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUTS / f"{asset}_rq3_run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"{asset} complete in {metadata['runtime_seconds']:.1f} seconds",
        flush=True,
    )


def build_tsla_summary(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(UPSTREAM / "TSLA_rq3_all_predictions.csv")
    predictions = predictions.loc[
        predictions["Role"].eq("Main model")
        & predictions["Comparator"].isin(engine.MAIN_MODELS)
    ].copy()
    fold_metrics = pd.read_csv(UPSTREAM / "TSLA_rq3_fold_metrics.csv")
    fold_metrics = fold_metrics.loc[
        fold_metrics["Role"].eq("Main model")
        & fold_metrics["Comparator"].isin(engine.MAIN_MODELS)
    ].copy()
    summary = engine.pooled_summary(predictions, fold_metrics)
    summary.insert(0, "Asset", "TSLA")
    predictions.insert(0, "Asset", "TSLA")
    return predictions, summary


def combine_outputs() -> None:
    engine = load_engine()
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "xgboost": __import__("xgboost").__version__,
        "tensorflow": engine.tensorflow_module().__version__,
    }
    combined_predictions: list[pd.DataFrame] = []
    combined_summaries: list[pd.DataFrame] = []
    tsla_predictions, tsla_summary = build_tsla_summary(engine)
    combined_predictions.append(tsla_predictions)
    combined_summaries.append(tsla_summary)

    configuration_frames = [
        pd.read_csv(UPSTREAM / "TSLA_rq3_selected_configurations.csv").assign(
            Asset="TSLA"
        )
    ]
    for asset in ASSETS_TO_FIT:
        prediction_path = UPSTREAM / f"{asset}_rq3_model_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Run --asset {asset} before combining outputs."
            )
        combined_predictions.append(pd.read_csv(prediction_path))
        combined_summaries.append(
            pd.read_csv(OUTPUTS / f"{asset}_rq3_classification_summary.csv")
        )
        configuration_frames.append(
            pd.read_csv(
                OUTPUTS / f"{asset}_rq3_selected_configurations.csv"
            )
        )
        metadata_path = OUTPUTS / f"{asset}_rq3_run_metadata.json"
        asset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "input_sha256": sha256(
                DATA / f"{asset}_features_common_window.csv"
            ),
            "rq3_engine_sha256": sha256(ENGINE_PATH),
            "rq3_protocol_sha256": sha256(RQ3_PROTOCOL_PATH),
        }
        for field, expected_value in expected_metadata.items():
            if asset_metadata.get(field) != expected_value:
                raise AssertionError(
                    f"{asset} metadata {field} does not match the frozen input."
                )
        if asset_metadata.get(
            "return_metrics_used_for_training_or_selection", True
        ):
            raise AssertionError(
                f"{asset} metadata indicates return-informed fitting."
            )
        asset_metadata["software"] = software
        metadata_path.write_text(
            json.dumps(asset_metadata, indent=2),
            encoding="utf-8",
        )

    predictions = pd.concat(combined_predictions, ignore_index=True)
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["Target_Date"] = pd.to_datetime(predictions["Target_Date"])
    predictions = predictions.sort_values(
        ["Asset", "Period", "Comparator", "Date"]
    ).reset_index(drop=True)
    expected = {
        ("Development OOF", model): 1250
        for model in engine.MAIN_MODELS
    } | {
        ("Retrospective audit", model): 407
        for model in engine.MAIN_MODELS
    }
    observed = predictions.groupby(["Asset", "Period", "Comparator"]).size()
    for asset in ALL_ASSETS:
        for key, count in expected.items():
            if int(observed.loc[(asset, *key)]) != count:
                raise AssertionError(
                    f"Unexpected prediction count for {asset}, {key}: "
                    f"{observed.loc[(asset, *key)]}"
                )
    if predictions.duplicated(["Asset", "Period", "Comparator", "Date"]).any():
        raise AssertionError("Combined model predictions are not unique.")

    predictions.to_csv(UPSTREAM / "rq4_model_predictions.csv", index=False)
    pd.concat(combined_summaries, ignore_index=True).to_csv(
        OUTPUTS / "rq4_cross_asset_classification_summary.csv",
        index=False,
    )
    pd.concat(configuration_frames, ignore_index=True).to_csv(
        OUTPUTS / "rq4_cross_asset_selected_configurations.csv",
        index=False,
    )

    manifest = {
        "assets": ALL_ASSETS,
        "models": engine.MAIN_MODELS,
        "periods": {
            "Development OOF": 1250,
            "Retrospective audit": 407,
        },
        "probability_threshold": engine.THRESHOLD,
        "combined_prediction_file": "upstream/rq4_model_predictions.csv",
        "combined_prediction_sha256": sha256(
            UPSTREAM / "rq4_model_predictions.csv"
        ),
        "sources": {
            "TSLA": "Frozen v8.3 RQ3 predictions",
            "AAPL": "Asset-specific replication using frozen RQ3 engine",
            "NVDA": "Asset-specific replication using frozen RQ3 engine",
        },
        "source_prediction_sha256": {
            "TSLA": sha256(UPSTREAM / "TSLA_rq3_all_predictions.csv"),
            "AAPL": sha256(UPSTREAM / "AAPL_rq3_model_predictions.csv"),
            "NVDA": sha256(UPSTREAM / "NVDA_rq3_model_predictions.csv"),
        },
        "data_sha256": {
            asset: sha256(DATA / f"{asset}_features_common_window.csv")
            for asset in ALL_ASSETS
        },
        "rq3_engine_sha256": sha256(ENGINE_PATH),
        "rq3_protocol_sha256": sha256(RQ3_PROTOCOL_PATH),
        "return_metrics_used_for_training_or_selection": False,
        "software": software,
    }
    (UPSTREAM / "cross_asset_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("Combined three-asset model predictions written.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        choices=ASSETS_TO_FIT,
        help="Fit and predict one cross-asset replication.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine frozen TSLA and completed AAPL/NVDA outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.asset:
        run_asset(args.asset)
    elif args.combine_only:
        combine_outputs()
    else:
        for asset in ASSETS_TO_FIT:
            run_asset(asset)
        combine_outputs()


if __name__ == "__main__":
    main()
