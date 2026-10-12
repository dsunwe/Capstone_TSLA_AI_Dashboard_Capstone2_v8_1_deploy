"""Run the focused v8.3 RQ3 common-information model comparison.

The four main models receive the same six frozen RQ2 Technical feature streams
over the same 20 trading sessions. Logistic Regression, Random Forest, and
XGBoost receive the chronological tensor flattened to 120 ordered inputs;
the compact LSTM retains the 20 x 6 representation.

All model-family selection is based only on nested development walk-forward
results. The later 2024-11-26 to 2026-07-15 period is reported only as a
retrospective temporal audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq3-matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "TSLA_features_common_window.csv"
PROTOCOL = ROOT / "protocol.json"
UPSTREAM_RQ2_PROTOCOL = ROOT / "upstream" / "rq2_protocol.json"
UPSTREAM_RQ2_OOF = (
    ROOT / "upstream" / "rq2_feature_family_oof_predictions.csv"
)
UPSTREAM_RQ2_AUDIT = (
    ROOT / "upstream" / "rq2_feature_family_audit_predictions.csv"
)
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
FREEZE = ROOT / "freeze"
for directory in [OUTPUTS, FIGURES, FREEZE]:
    directory.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Daily_Return",
    "Momentum_5",
    "EMA8_13_State",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]
MAIN_MODELS = [
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
]
STOCHASTIC_MODELS = {"Random Forest", "XGBoost", "Compact LSTM"}
BASELINES = [
    "Training-majority",
    "Weekly-trend persistence",
    "EMA8/13 state",
]
BRIDGE_MODEL = "RQ2 current-day Logistic Regression"

WARMUP_START = pd.Timestamp("2016-10-03")
AUDIT_START = pd.Timestamp("2024-11-26")
HORIZON = 5
LOOKBACK = 20
OUTER_SPLITS = 5
OUTER_TEST_SIZE = 250
INNER_SPLITS = 3
INNER_TEST_SIZE = 125
PURGE_ROWS = 5
THRESHOLD = 0.50
SEEDS = [42, 43, 44]
TUNING_SEED = 42
BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 5_000


@dataclass
class PreparedData:
    full: pd.DataFrame
    origins: pd.DataFrame
    sequences: np.ndarray
    development_indices: np.ndarray
    audit_indices: np.ndarray


@dataclass
class ChannelPreprocessor:
    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    unique_fit_rows: int

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        shape = sequences.shape
        flat = sequences.reshape(-1, shape[-1])
        transformed = self.scaler.transform(self.imputer.transform(flat))
        return transformed.reshape(shape).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "imputer_statistics": self.imputer.statistics_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "fit_start": self.fit_start.strftime("%Y-%m-%d"),
            "fit_end": self.fit_end.strftime("%Y-%m-%d"),
            "unique_fit_rows": self.unique_fit_rows,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def tensorflow_module():
    import tensorflow as tf

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass
    return tf


def add_moving_average_state(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    available = data[["EMA8", "EMA13"]].notna().all(axis=1)
    data["EMA8_13_State"] = np.where(
        available,
        (data["EMA8"] > data["EMA13"]).astype(int),
        np.nan,
    )
    return data


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("Date").reset_index(drop=True)
    current_mean = data["Close"].rolling(HORIZON).mean()
    future_mean = sum(
        data["Close"].shift(-offset)
        for offset in range(1, HORIZON + 1)
    ) / HORIZON
    previous_mean = current_mean.shift(HORIZON)
    target_return = future_mean / current_mean - 1
    data["Current_Week_Mean_Close"] = current_mean
    data["Future_Week_Mean_Close"] = future_mean
    data["Weekly_Trend_Return"] = target_return
    data["Weekly_Trend_Direction"] = np.where(
        target_return.notna(),
        (target_return > 0).astype(int),
        np.nan,
    )
    data["Weekly_Persistence"] = np.where(
        previous_mean.notna(),
        (current_mean > previous_mean).astype(int),
        np.nan,
    )
    data["Target_Date"] = data["Date"].shift(-HORIZON)
    data["Full_Row_Position"] = np.arange(len(data), dtype=int)
    return data


def prepare_data() -> PreparedData:
    full = pd.read_csv(DATA, parse_dates=["Date"])
    full = full.sort_values("Date").drop_duplicates("Date", keep="last")
    full = add_moving_average_state(full)
    full = add_target(full)

    required = [
        "Date",
        "Target_Date",
        "Weekly_Trend_Direction",
        "Weekly_Persistence",
        *FEATURES,
    ]
    eligible_mask = (
        full["Date"].ge(WARMUP_START)
        & full[required].notna().all(axis=1)
    )
    origins = full.loc[eligible_mask].copy().reset_index(drop=True)
    origins["Eligible_Row_ID"] = np.arange(len(origins), dtype=int)
    origins["Phase_Offset"] = origins["Eligible_Row_ID"] % HORIZON

    sequences: list[np.ndarray] = []
    sequence_starts: list[pd.Timestamp] = []
    sequence_ends: list[pd.Timestamp] = []
    for row in origins.itertuples(index=False):
        anchor = int(row.Full_Row_Position)
        start = anchor - LOOKBACK + 1
        if start < 0:
            raise AssertionError("A target origin lacks a complete 20-session history.")
        window = full.iloc[start : anchor + 1][FEATURES].to_numpy(dtype=float)
        if window.shape != (LOOKBACK, len(FEATURES)):
            raise AssertionError(f"Unexpected sequence shape: {window.shape}")
        sequences.append(window)
        sequence_starts.append(full.iloc[start]["Date"])
        sequence_ends.append(full.iloc[anchor]["Date"])
    origin_sequences = np.asarray(sequences, dtype=np.float64)
    origins["Sequence_Start"] = sequence_starts
    origins["Sequence_End"] = sequence_ends

    if not origins["Sequence_End"].equals(origins["Date"]):
        raise AssertionError("At least one sequence extends beyond its origin.")
    last_steps = origin_sequences[:, -1, :]
    if not np.allclose(
        last_steps,
        origins[FEATURES].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AssertionError("The final sequence step differs from the origin features.")

    development_mask = origins["Target_Date"].lt(AUDIT_START)
    audit_mask = origins["Date"].ge(AUDIT_START)
    development_indices = np.flatnonzero(development_mask.to_numpy())
    audit_indices = np.flatnonzero(audit_mask.to_numpy())
    development = origins.iloc[development_indices]
    audit = origins.iloc[audit_indices]
    if development["Target_Date"].max() >= audit["Date"].min():
        raise AssertionError("Development labels overlap the audit feature period.")
    if len(development_indices) != 2_046 or len(audit_indices) != 407:
        raise AssertionError(
            f"Unexpected split sizes: {len(development_indices)}, {len(audit_indices)}"
        )
    return PreparedData(
        full=full,
        origins=origins,
        sequences=origin_sequences,
        development_indices=development_indices,
        audit_indices=audit_indices,
    )


def fit_channel_preprocessor(
    prepared: PreparedData,
    training_origin_indices: np.ndarray,
) -> ChannelPreprocessor:
    anchors = prepared.origins.iloc[training_origin_indices][
        "Full_Row_Position"
    ].astype(int)
    used = np.zeros(len(prepared.full), dtype=bool)
    for anchor in anchors:
        used[anchor - LOOKBACK + 1 : anchor + 1] = True
    fit_frame = prepared.full.loc[used, ["Date", *FEATURES]].copy()
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(
        fit_frame[FEATURES].to_numpy(dtype=float)
    )
    scaler = StandardScaler()
    scaler.fit(imputed)
    return ChannelPreprocessor(
        feature_names=list(FEATURES),
        imputer=imputer,
        scaler=scaler,
        fit_start=fit_frame["Date"].min(),
        fit_end=fit_frame["Date"].max(),
        unique_fit_rows=len(fit_frame),
    )


def transform_for_model(
    prepared: PreparedData,
    preprocessor: ChannelPreprocessor,
    origin_indices: np.ndarray,
    model_name: str,
) -> np.ndarray:
    transformed = preprocessor.transform(prepared.sequences[origin_indices])
    if model_name == "Compact LSTM":
        return transformed
    return transformed.reshape(len(transformed), -1)


def class_weights(labels: np.ndarray) -> dict[int, float] | None:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        return None
    total = len(labels)
    return {
        0: float(total / (2 * counts[0])),
        1: float(total / (2 * counts[1])),
    }


def build_tabular_model(
    model_name: str,
    config: dict[str, Any],
    y_train: np.ndarray,
    seed: int,
):
    if model_name == "Lagged Logistic Regression":
        return LogisticRegression(
            C=float(config["C"]),
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
        scale_pos_weight = (
            float(counts[0] / counts[1]) if counts[1] > 0 else 1.0
        )
        return XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            min_child_weight=float(config["min_child_weight"]),
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported tabular model: {model_name}")


def build_lstm(config: dict[str, Any], seed: int):
    tf = tensorflow_module()
    tf.keras.backend.clear_session()
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK, len(FEATURES))),
            tf.keras.layers.LSTM(
                int(config["units"]),
                activation="tanh",
                dropout=float(config["dropout"]),
                recurrent_dropout=0.0,
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="rq3_compact_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(config["learning_rate"]),
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
    )
    return model


def classification_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    return {
        "Accuracy": float(accuracy_score(actual, prediction)),
        "Balanced Accuracy": float(
            balanced_accuracy_score(actual, prediction)
        ),
        "Macro F1": float(
            f1_score(actual, prediction, average="macro", zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(actual, prediction)),
        "Recall Down": float(
            recall_score(actual, prediction, pos_label=0, zero_division=0)
        ),
        "Recall Up": float(
            recall_score(actual, prediction, pos_label=1, zero_division=0)
        ),
        "ROC-AUC": (
            float(roc_auc_score(actual, probability))
            if len(np.unique(actual)) == 2
            else np.nan
        ),
        "Brier Score": float(brier_score_loss(actual, probability)),
        "Predicted Positive Share": float(prediction.mean()),
        "Actual Positive Share": float(actual.mean()),
    }


def fit_lstm_inner(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int, int, float]:
    tf = tensorflow_module()
    model = build_lstm(config, seed)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=100,
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[callback],
        class_weight=class_weights(y_train),
    )
    epochs_run = int(len(history.history["loss"]))
    monitored_epoch = getattr(callback, "best_epoch", None)
    if (
        callback.best_weights is None
        or not isinstance(monitored_epoch, (int, np.integer))
        or not callback.start_from_epoch <= monitored_epoch < epochs_run
        or not np.isfinite(history.history["val_loss"][monitored_epoch])
    ):
        tf.keras.backend.clear_session()
        raise RuntimeError(
            "LSTM early stopping did not record a finite monitored best epoch; "
            "refusing to choose a refit epoch from unmonitored validation loss."
        )
    # Keras records a zero-based epoch for the weights restored at fit end.
    # A raw history argmin would include warm-up epochs and ignore min_delta.
    best_epoch = int(monitored_epoch) + 1
    probability = model.predict(x_validation, verbose=0).reshape(-1)
    elapsed = time.perf_counter() - started
    tf.keras.backend.clear_session()
    return probability, best_epoch, epochs_run, elapsed


def fit_fixed_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: dict[str, Any],
    seed: int,
    epochs: int,
):
    model = build_lstm(config, seed)
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        epochs=int(epochs),
        batch_size=32,
        shuffle=False,
        verbose=0,
        class_weight=class_weights(y_train),
    )
    probability = model.predict(x_test, verbose=0).reshape(-1)
    elapsed = time.perf_counter() - started
    return model, probability, elapsed


def candidate_configurations(
    protocol: dict[str, Any],
    model_name: str,
) -> list[dict[str, Any]]:
    return [
        dict(config)
        for config in protocol["main_models"][model_name][
            "candidate_configurations"
        ]
    ]


def validate_temporal_boundary(
    prepared: PreparedData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label: str,
) -> None:
    train_target_end = prepared.origins.iloc[train_indices]["Target_Date"].max()
    test_start = prepared.origins.iloc[test_indices]["Date"].min()
    if train_target_end >= test_start:
        raise AssertionError(
            f"{label} leaks a training label into evaluation: "
            f"{train_target_end} >= {test_start}"
        )


def tune_model(
    prepared: PreparedData,
    protocol: dict[str, Any],
    training_origin_indices: np.ndarray,
    model_name: str,
    scope: str,
    outer_fold: int | None,
) -> tuple[
    dict[str, Any],
    int | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    configs = candidate_configurations(protocol, model_name)
    splitter = TimeSeriesSplit(
        n_splits=INNER_SPLITS,
        test_size=INNER_TEST_SIZE,
        gap=PURGE_ROWS,
    )
    local_splits = list(splitter.split(training_origin_indices))
    detail_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
        local_splits,
        start=1,
    ):
        inner_train = training_origin_indices[inner_train_local]
        inner_validation = training_origin_indices[inner_validation_local]
        validate_temporal_boundary(
            prepared,
            inner_train,
            inner_validation,
            f"{scope} inner fold {inner_fold}",
        )
        train_rows = prepared.origins.iloc[inner_train]
        validation_rows = prepared.origins.iloc[inner_validation]
        manifest_rows.append(
            {
                "Scope": scope,
                "Outer Fold": outer_fold,
                "Inner Fold": inner_fold,
                "Train Start": train_rows["Date"].min(),
                "Train End": train_rows["Date"].max(),
                "Train Target End": train_rows["Target_Date"].max(),
                "Validation Start": validation_rows["Date"].min(),
                "Validation End": validation_rows["Date"].max(),
                "Train Rows": len(train_rows),
                "Validation Rows": len(validation_rows),
                "Gap Rows": PURGE_ROWS,
            }
        )

    for config_index, config in enumerate(configs, start=1):
        config_id = f"{model_name.replace(' ', '_')}-{config_index:02d}"
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            local_splits,
            start=1,
        ):
            inner_train = training_origin_indices[inner_train_local]
            inner_validation = training_origin_indices[inner_validation_local]
            preprocessor = fit_channel_preprocessor(prepared, inner_train)
            x_train = transform_for_model(
                prepared,
                preprocessor,
                inner_train,
                model_name,
            )
            x_validation = transform_for_model(
                prepared,
                preprocessor,
                inner_validation,
                model_name,
            )
            y_train = prepared.origins.iloc[inner_train][
                "Weekly_Trend_Direction"
            ].astype(int).to_numpy()
            actual = prepared.origins.iloc[inner_validation][
                "Weekly_Trend_Direction"
            ].astype(int).to_numpy()
            best_epoch: int | None = None
            epochs_run: int | None = None
            started = time.perf_counter()
            if model_name == "Compact LSTM":
                probability, best_epoch, epochs_run, elapsed = fit_lstm_inner(
                    x_train,
                    y_train,
                    x_validation,
                    actual,
                    config,
                    TUNING_SEED,
                )
            else:
                model = build_tabular_model(
                    model_name,
                    config,
                    y_train,
                    TUNING_SEED,
                )
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_validation)[:, 1]
                elapsed = time.perf_counter() - started
            prediction = (probability >= THRESHOLD).astype(int)
            metrics = classification_metrics(actual, prediction, probability)
            train_rows = prepared.origins.iloc[inner_train]
            validation_rows = prepared.origins.iloc[inner_validation]
            detail_rows.append(
                {
                    "Scope": scope,
                    "Outer Fold": outer_fold,
                    "Model": model_name,
                    "Candidate": config_id,
                    "Candidate Order": config_index,
                    "Configuration": stable_json(config),
                    "Inner Fold": inner_fold,
                    "Train Start": train_rows["Date"].min(),
                    "Train End": train_rows["Date"].max(),
                    "Train Target End": train_rows["Target_Date"].max(),
                    "Validation Start": validation_rows["Date"].min(),
                    "Validation End": validation_rows["Date"].max(),
                    "Train Rows": len(inner_train),
                    "Validation Rows": len(inner_validation),
                    "Preprocessor Fit End": preprocessor.fit_end,
                    "Best Epoch": best_epoch,
                    "Epochs Run": epochs_run,
                    "Fit and Inference Seconds": elapsed,
                    **metrics,
                }
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(
            ["Candidate", "Candidate Order", "Configuration"],
            as_index=False,
        )
        .agg(
            Mean_Inner_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            SD_Inner_Balanced_Accuracy=("Balanced Accuracy", "std"),
            Mean_Inner_Accuracy=("Accuracy", "mean"),
            Mean_Inner_Macro_F1=("Macro F1", "mean"),
        )
        .sort_values(
            ["Mean_Inner_Balanced_Accuracy", "Candidate Order"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    selected_candidate = summary.iloc[0]["Candidate"]
    selected_order = int(
        detail.loc[detail["Candidate"].eq(selected_candidate), "Candidate Order"]
        .iloc[0]
    )
    selected_config = configs[selected_order - 1]
    selected_epoch: int | None = None
    if model_name == "Compact LSTM":
        selected_epochs = detail.loc[
            detail["Candidate"].eq(selected_candidate), "Best Epoch"
        ].dropna()
        selected_epoch = max(1, int(np.rint(selected_epochs.median())))
    for row in detail_rows:
        row["Selected Candidate"] = row["Candidate"] == selected_candidate
        row["Selected Refit Epoch"] = selected_epoch
    return selected_config, selected_epoch, detail_rows, manifest_rows


def fit_predict_ensemble(
    prepared: PreparedData,
    training_origin_indices: np.ndarray,
    test_origin_indices: np.ndarray,
    model_name: str,
    config: dict[str, Any],
    selected_epoch: int | None,
    retain_models: bool = False,
) -> tuple[
    ChannelPreprocessor,
    list[Any],
    np.ndarray,
    list[dict[str, Any]],
    float,
]:
    preprocessor = fit_channel_preprocessor(prepared, training_origin_indices)
    x_train = transform_for_model(
        prepared,
        preprocessor,
        training_origin_indices,
        model_name,
    )
    x_test = transform_for_model(
        prepared,
        preprocessor,
        test_origin_indices,
        model_name,
    )
    y_train = prepared.origins.iloc[training_origin_indices][
        "Weekly_Trend_Direction"
    ].astype(int).to_numpy()
    seeds = SEEDS if model_name in STOCHASTIC_MODELS else [TUNING_SEED]
    probabilities: list[np.ndarray] = []
    seed_rows: list[dict[str, Any]] = []
    retained: list[Any] = []
    total_seconds = 0.0
    for seed in seeds:
        if model_name == "Compact LSTM":
            if selected_epoch is None:
                raise AssertionError("A fixed LSTM refit epoch was not selected.")
            model, probability, elapsed = fit_fixed_lstm(
                x_train,
                y_train,
                x_test,
                config,
                seed,
                selected_epoch,
            )
        else:
            model = build_tabular_model(
                model_name,
                config,
                y_train,
                seed,
            )
            started = time.perf_counter()
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            elapsed = time.perf_counter() - started
        probabilities.append(np.asarray(probability, dtype=float))
        total_seconds += elapsed
        seed_rows.append(
            {
                "Model": model_name,
                "Seed": seed,
                "Configuration": stable_json(config),
                "Selected Epoch": selected_epoch,
                "Fit and Inference Seconds": elapsed,
            }
        )
        if retain_models:
            retained.append(model)
        elif model_name == "Compact LSTM":
            tensorflow_module().keras.backend.clear_session()
    ensemble = np.mean(np.vstack(probabilities), axis=0)
    for row, probability in zip(seed_rows, probabilities):
        row["Probability Array"] = probability
    return preprocessor, retained, ensemble, seed_rows, total_seconds


def bridge_logistic_predictions(
    prepared: PreparedData,
    training_origin_indices: np.ndarray,
    test_origin_indices: np.ndarray,
) -> np.ndarray:
    x_train = prepared.origins.iloc[training_origin_indices][FEATURES]
    x_test = prepared.origins.iloc[test_origin_indices][FEATURES]
    y_train = prepared.origins.iloc[training_origin_indices][
        "Weekly_Trend_Direction"
    ].astype(int)
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(x_train)
    test_imputed = imputer.transform(x_test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=2_000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(train_scaled, y_train)
    return model.predict_proba(test_scaled)[:, 1]


def append_prediction_frame(
    collection: list[pd.DataFrame],
    prepared: PreparedData,
    origin_indices: np.ndarray,
    comparator: str,
    role: str,
    period: str,
    fold: int | str,
    probability: np.ndarray,
) -> pd.DataFrame:
    rows = prepared.origins.iloc[origin_indices][
        [
            "Date",
            "Target_Date",
            "Eligible_Row_ID",
            "Phase_Offset",
            "Weekly_Trend_Direction",
        ]
    ].copy()
    rows = rows.rename(columns={"Weekly_Trend_Direction": "Actual"})
    rows["Comparator"] = comparator
    rows["Role"] = role
    rows["Period"] = period
    rows["Fold"] = fold
    rows["Probability"] = np.asarray(probability, dtype=float)
    rows["Prediction"] = (rows["Probability"] >= THRESHOLD).astype(int)
    collection.append(rows)
    return rows


def fold_metric_row(
    prediction_frame: pd.DataFrame,
    prepared: PreparedData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    training_seconds: float,
    config: dict[str, Any] | None,
    selected_epoch: int | None,
) -> dict[str, Any]:
    train = prepared.origins.iloc[train_indices]
    test = prepared.origins.iloc[test_indices]
    metrics = classification_metrics(
        prediction_frame["Actual"].astype(int).to_numpy(),
        prediction_frame["Prediction"].astype(int).to_numpy(),
        prediction_frame["Probability"].to_numpy(),
    )
    return {
        "Comparator": prediction_frame["Comparator"].iloc[0],
        "Role": prediction_frame["Role"].iloc[0],
        "Period": prediction_frame["Period"].iloc[0],
        "Fold": prediction_frame["Fold"].iloc[0],
        "Train Start": train["Date"].min(),
        "Train End": train["Date"].max(),
        "Train Target End": train["Target_Date"].max(),
        "Test Start": test["Date"].min(),
        "Test End": test["Date"].max(),
        "Train Rows": len(train),
        "Test Rows": len(test),
        "Configuration": stable_json(config) if config is not None else None,
        "Selected Epoch": selected_epoch,
        "Fit and Inference Seconds": training_seconds,
        **metrics,
    }


def one_standard_error_selection(
    fold_metrics: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    main = fold_metrics.loc[
        fold_metrics["Comparator"].isin(MAIN_MODELS)
        & fold_metrics["Period"].eq("Development OOF")
    ].copy()
    summary = (
        main.groupby("Comparator", as_index=False)
        .agg(
            Mean_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            SD_Balanced_Accuracy=("Balanced Accuracy", "std"),
            Minimum_Balanced_Accuracy=("Balanced Accuracy", "min"),
            Mean_Accuracy=("Accuracy", "mean"),
            SD_Accuracy=("Accuracy", "std"),
        )
        .rename(columns={"Comparator": "Model"})
    )
    best_row = summary.sort_values(
        ["Mean_Balanced_Accuracy", "Model"],
        ascending=[False, True],
    ).iloc[0]
    best_mean = float(best_row["Mean_Balanced_Accuracy"])
    best_se = float(best_row["SD_Balanced_Accuracy"] / math.sqrt(OUTER_SPLITS))
    cutoff = best_mean - best_se
    complexity = {
        model: int(protocol["main_models"][model]["complexity_rank"])
        for model in MAIN_MODELS
    }
    summary["Best Model by Mean"] = summary["Model"].eq(best_row["Model"])
    summary["Best Model Standard Error"] = best_se
    summary["One-SE Cutoff"] = cutoff
    summary["One-SE Eligible"] = summary["Mean_Balanced_Accuracy"].ge(cutoff)
    summary["Complexity Rank"] = summary["Model"].map(complexity)
    eligible = summary.loc[summary["One-SE Eligible"]].sort_values(
        "Complexity Rank"
    )
    selected = eligible.iloc[0]["Model"]
    summary["Selected for RQ4"] = summary["Model"].eq(selected)
    summary["Selection Data"] = "Development outer-fold OOF only"
    summary["Audit Used"] = False
    summary["RQ4 Returns Used"] = False
    return summary.sort_values("Complexity Rank").reset_index(drop=True)


def pooled_summary(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparator, frame in predictions.groupby("Comparator", sort=False):
        dev = frame.loc[frame["Period"].eq("Development OOF")]
        audit = frame.loc[frame["Period"].eq("Retrospective audit")]
        dev_folds = fold_metrics.loc[
            fold_metrics["Comparator"].eq(comparator)
            & fold_metrics["Period"].eq("Development OOF")
        ]
        row: dict[str, Any] = {
            "Comparator": comparator,
            "Role": frame["Role"].iloc[0],
            "Development Rows": len(dev),
            "Development Mean Fold Accuracy": dev_folds["Accuracy"].mean(),
            "Development Accuracy SD": dev_folds["Accuracy"].std(),
            "Development Minimum Fold Accuracy": dev_folds["Accuracy"].min(),
            "Development Mean Fold Balanced Accuracy": dev_folds[
                "Balanced Accuracy"
            ].mean(),
            "Development Balanced Accuracy SD": dev_folds[
                "Balanced Accuracy"
            ].std(),
            "Development Minimum Fold Balanced Accuracy": dev_folds[
                "Balanced Accuracy"
            ].min(),
            "Audit Rows": len(audit),
        }
        if len(dev):
            dev_metrics = classification_metrics(
                dev["Actual"].astype(int).to_numpy(),
                dev["Prediction"].astype(int).to_numpy(),
                dev["Probability"].to_numpy(),
            )
            row.update({f"Development Pooled {k}": v for k, v in dev_metrics.items()})
        if len(audit):
            audit_metrics = classification_metrics(
                audit["Actual"].astype(int).to_numpy(),
                audit["Prediction"].astype(int).to_numpy(),
                audit["Probability"].to_numpy(),
            )
            row.update({f"Audit {k}": v for k, v in audit_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def circular_block_positions(
    positions: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    block_count = int(np.ceil(len(positions) / block_length))
    starts = rng.integers(0, len(positions), size=block_count)
    local = np.concatenate(
        [
            (start + np.arange(block_length)) % len(positions)
            for start in starts
        ]
    )[: len(positions)]
    return positions[local]


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    comparator_a: str,
    comparator_b: str,
    period: str,
    comparison_family: str,
) -> list[dict[str, Any]]:
    period_frame = predictions.loc[predictions["Period"].eq(period)]
    keys = ["Date", "Target_Date", "Actual", "Phase_Offset"]
    if period == "Development OOF":
        keys.append("Fold")
    pivot = period_frame.pivot_table(
        index=keys,
        columns="Comparator",
        values="Prediction",
        aggfunc="first",
    ).reset_index()
    if comparator_a not in pivot or comparator_b not in pivot:
        raise AssertionError(f"Missing bootstrap pair: {comparator_a}, {comparator_b}")
    pivot = pivot.dropna(subset=[comparator_a, comparator_b])
    sort_columns = ["Fold", "Date"] if "Fold" in pivot else ["Date"]
    pivot = pivot.sort_values(sort_columns).reset_index(drop=True)
    actual = pivot["Actual"].astype(int).to_numpy()
    prediction_a = pivot[comparator_a].astype(int).to_numpy()
    prediction_b = pivot[comparator_b].astype(int).to_numpy()

    point_accuracy = accuracy_score(actual, prediction_a) - accuracy_score(
        actual, prediction_b
    )
    point_balanced = balanced_accuracy_score(
        actual, prediction_a
    ) - balanced_accuracy_score(actual, prediction_b)
    rng = np.random.default_rng(42)
    accuracy_distribution: list[float] = []
    balanced_distribution: list[float] = []
    if "Fold" in pivot:
        fold_position_sets = [
            group.index.to_numpy()
            for _, group in pivot.groupby("Fold", sort=True)
        ]
    else:
        fold_position_sets = [pivot.index.to_numpy()]
    attempts = 0
    while (
        len(accuracy_distribution) < BOOTSTRAP_REPLICATES
        and attempts < BOOTSTRAP_REPLICATES * 3
    ):
        attempts += 1
        sampled = np.concatenate(
            [
                circular_block_positions(positions, BLOCK_LENGTH, rng)
                for positions in fold_position_sets
            ]
        )
        sampled_actual = actual[sampled]
        if len(np.unique(sampled_actual)) < 2:
            continue
        sampled_a = prediction_a[sampled]
        sampled_b = prediction_b[sampled]
        accuracy_distribution.append(
            accuracy_score(sampled_actual, sampled_a)
            - accuracy_score(sampled_actual, sampled_b)
        )
        balanced_distribution.append(
            balanced_accuracy_score(sampled_actual, sampled_a)
            - balanced_accuracy_score(sampled_actual, sampled_b)
        )
    if len(accuracy_distribution) != BOOTSTRAP_REPLICATES:
        raise AssertionError("Unable to obtain the requested valid bootstrap draws.")

    rows: list[dict[str, Any]] = []
    for metric, point, distribution in [
        ("Accuracy", point_accuracy, np.asarray(accuracy_distribution)),
        ("Balanced Accuracy", point_balanced, np.asarray(balanced_distribution)),
    ]:
        lower = float(np.quantile(distribution, 0.025))
        upper = float(np.quantile(distribution, 0.975))
        p_value = float(
            min(
                1.0,
                2
                * min(
                    (1 + np.sum(distribution <= 0))
                    / (BOOTSTRAP_REPLICATES + 1),
                    (1 + np.sum(distribution >= 0))
                    / (BOOTSTRAP_REPLICATES + 1),
                ),
            )
        )
        rows.append(
            {
                "Comparison Family": comparison_family,
                "Period": period,
                "Metric": metric,
                "Comparator A": comparator_a,
                "Comparator B": comparator_b,
                "Rows": len(pivot),
                "Point Difference": float(point),
                "CI Lower 95%": lower,
                "CI Upper 95%": upper,
                "Two-sided Bootstrap p": p_value,
                "Bootstrap Replicates": BOOTSTRAP_REPLICATES,
                "Block Length": BLOCK_LENGTH,
                "Fold-aware Development Resampling": (
                    period == "Development OOF"
                ),
            }
        )
    return rows


def add_holm_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    group_columns = ["Comparison Family", "Period", "Metric"]
    for _, group in frame.groupby(group_columns, sort=False):
        current = group.copy()
        values = current["Two-sided Bootstrap p"].to_numpy()
        order = np.argsort(values)
        adjusted = np.empty(len(current), dtype=float)
        running = 0.0
        count = len(current)
        for rank, position in enumerate(order):
            candidate = min(1.0, (count - rank) * values[position])
            running = max(running, candidate)
            adjusted[position] = running
        current["Holm-adjusted p"] = adjusted
        output.append(current)
    return pd.concat(output, ignore_index=True)


def bootstrap_all(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ["Development OOF", "Retrospective audit"]:
        for model in ["Random Forest", "XGBoost", "Compact LSTM"]:
            rows.extend(
                paired_block_bootstrap(
                    predictions,
                    model,
                    "Lagged Logistic Regression",
                    period,
                    "Primary complex-versus-linear",
                )
            )
        for model in MAIN_MODELS:
            for baseline in [
                "Weekly-trend persistence",
                "EMA8/13 state",
            ]:
                rows.extend(
                    paired_block_bootstrap(
                        predictions,
                        model,
                        baseline,
                        period,
                        "Secondary model-versus-transparent-baseline",
                    )
                )
    return add_holm_adjustment(pd.DataFrame(rows))


def phase_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (period, comparator, phase), frame in predictions.groupby(
        ["Period", "Comparator", "Phase_Offset"],
        sort=False,
    ):
        metrics = classification_metrics(
            frame["Actual"].astype(int).to_numpy(),
            frame["Prediction"].astype(int).to_numpy(),
            frame["Probability"].to_numpy(),
        )
        rows.append(
            {
                "Period": period,
                "Comparator": comparator,
                "Role": frame["Role"].iloc[0],
                "Phase Offset": int(phase),
                "Rows": len(frame),
                "Start": frame["Date"].min(),
                "End": frame["Date"].max(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_model: str,
) -> None:
    main = summary.loc[summary["Comparator"].isin(MAIN_MODELS)].copy()
    x = np.arange(len(main))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar(
        x - width / 2,
        main["Development Mean Fold Balanced Accuracy"] * 100,
        width,
        yerr=main["Development Balanced Accuracy SD"] * 100,
        capsize=4,
        label="Development mean fold",
        color="#2563EB",
    )
    axis.bar(
        x + width / 2,
        main["Audit Balanced Accuracy"] * 100,
        width,
        label="Retrospective audit",
        color="#F59E0B",
    )
    axis.set_xticks(x, main["Comparator"], rotation=15, ha="right")
    axis.set_ylabel("Balanced Accuracy (%)")
    axis.set_title("RQ3 model performance with common 20-session information")
    axis.set_ylim(40, 80)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_model_balanced_accuracy.png", dpi=180)
    plt.close(fig)

    main_folds = fold_metrics.loc[
        fold_metrics["Comparator"].isin(MAIN_MODELS)
        & fold_metrics["Period"].eq("Development OOF")
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    for model, frame in main_folds.groupby("Comparator", sort=False):
        axis.plot(
            frame["Fold"].astype(int),
            frame["Balanced Accuracy"] * 100,
            marker="o",
            linewidth=2,
            label=model,
        )
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Balanced Accuracy (%)")
    axis.set_title("Development outer-fold Balanced Accuracy")
    axis.set_xticks(range(1, OUTER_SPLITS + 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_outer_fold_balanced_accuracy.png", dpi=180)
    plt.close(fig)

    primary = bootstrap.loc[
        bootstrap["Comparison Family"].eq("Primary complex-versus-linear")
        & bootstrap["Metric"].eq("Balanced Accuracy")
    ].copy()
    primary["Label"] = (
        primary["Comparator A"] + "\n" + primary["Period"]
    )
    positions = np.arange(len(primary))
    point = primary["Point Difference"].to_numpy() * 100
    lower = primary["CI Lower 95%"].to_numpy() * 100
    upper = primary["CI Upper 95%"].to_numpy() * 100
    fig, axis = plt.subplots(figsize=(9, 5.4))
    axis.errorbar(
        point,
        positions,
        xerr=np.vstack(
            [
                np.maximum(point - lower, 0),
                np.maximum(upper - point, 0),
            ]
        ),
        fmt="o",
        color="#7C3AED",
        ecolor="#94A3B8",
        capsize=4,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions, primary["Label"])
    axis.set_xlabel("Balanced Accuracy difference versus lagged LR (pp)")
    axis.set_title("Paired 20-session block-bootstrap model contrasts")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_complex_vs_lr_bootstrap.png", dpi=180)
    plt.close(fig)

    selected_audit = predictions.loc[
        predictions["Comparator"].eq(selected_model)
        & predictions["Period"].eq("Retrospective audit")
    ]
    matrix = confusion_matrix(
        selected_audit["Actual"].astype(int),
        selected_audit["Prediction"].astype(int),
        labels=[0, 1],
    )
    short_model_name = {
        "Lagged Logistic Regression": "Lagged LR",
        "Random Forest": "RF",
        "XGBoost": "XGBoost",
        "Compact LSTM": "LSTM",
    }.get(selected_model, selected_model)
    fig, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=12,
            )
    axis.set_xticks([0, 1], ["Predicted down", "Predicted up"])
    axis.set_yticks([0, 1], ["Actual down", "Actual up"])
    axis.set_title(
        f"Retrospective audit confusion matrix: {short_model_name}"
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq3_selected_audit_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_selected_model_artifacts(
    selected_model: str,
    selected_config: dict[str, Any],
    selected_epoch: int | None,
    preprocessor: ChannelPreprocessor,
    models: list[Any],
) -> list[Path]:
    paths: list[Path] = []
    preprocessor_path = FREEZE / "rq3_selected_preprocessor.json"
    preprocessor_path.write_text(
        json.dumps(preprocessor.to_dict(), indent=2),
        encoding="utf-8",
    )
    paths.append(preprocessor_path)
    for seed, model in zip(
        SEEDS if selected_model in STOCHASTIC_MODELS else [TUNING_SEED],
        models,
    ):
        if selected_model == "Compact LSTM":
            path = FREEZE / f"rq3_selected_model_seed_{seed}.keras"
            model.save(path)
        else:
            path = FREEZE / f"rq3_selected_model_seed_{seed}.joblib"
            joblib.dump(model, path)
        paths.append(path)
    metadata_path = FREEZE / "rq3_selected_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selected_model": selected_model,
                "configuration": selected_config,
                "selected_epoch": selected_epoch,
                "features": FEATURES,
                "lookback": LOOKBACK,
                "tabular_feature_count": LOOKBACK * len(FEATURES),
                "seeds": (
                    SEEDS
                    if selected_model in STOCHASTIC_MODELS
                    else [TUNING_SEED]
                ),
                "probability_threshold": THRESHOLD,
                "selection_rule": "development-only one-standard-error",
                "audit_used_for_selection": False,
                "rq4_returns_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.append(metadata_path)
    return paths


def main() -> None:
    started_all = time.perf_counter()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_hashes = {
        DATA: protocol["input_snapshot"]["sha256"],
        UPSTREAM_RQ2_PROTOCOL: protocol["upstream_rq2"]["protocol_sha256"],
        UPSTREAM_RQ2_OOF: protocol["upstream_rq2"]["oof_predictions_sha256"],
        UPSTREAM_RQ2_AUDIT: protocol["upstream_rq2"][
            "audit_predictions_sha256"
        ],
    }
    for path, expected in expected_hashes.items():
        actual = file_sha256(path)
        if actual != expected:
            raise AssertionError(f"Frozen input hash mismatch for {path}: {actual}")

    prepared = prepare_data()
    origins = prepared.origins
    sequence_index = origins[
        [
            "Eligible_Row_ID",
            "Date",
            "Sequence_Start",
            "Sequence_End",
            "Target_Date",
            "Phase_Offset",
            "Full_Row_Position",
        ]
    ].copy()
    sequence_index["Sample Partition"] = "Boundary-excluded"
    sequence_index.loc[
        sequence_index["Eligible_Row_ID"].isin(
            origins.iloc[prepared.development_indices]["Eligible_Row_ID"]
        ),
        "Sample Partition",
    ] = "Development"
    sequence_index.loc[
        sequence_index["Eligible_Row_ID"].isin(
            origins.iloc[prepared.audit_indices]["Eligible_Row_ID"]
        ),
        "Sample Partition",
    ] = "Retrospective audit"
    sequence_index.to_csv(OUTPUTS / "rq3_sequence_index.csv", index=False)

    lagged_names = [
        f"{feature}_lag_{LOOKBACK - 1 - step}"
        for step in range(LOOKBACK)
        for feature in FEATURES
    ]
    (OUTPUTS / "rq3_lagged_feature_order.json").write_text(
        json.dumps(
            {
                "order": "chronological time-major, oldest to current",
                "shape_before_flattening": [LOOKBACK, len(FEATURES)],
                "flattened_feature_count": len(lagged_names),
                "flattened_features": lagged_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    outer_splitter = TimeSeriesSplit(
        n_splits=OUTER_SPLITS,
        test_size=OUTER_TEST_SIZE,
        gap=PURGE_ROWS,
    )
    outer_splits = list(outer_splitter.split(prepared.development_indices))
    outer_manifest_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    inner_manifest_rows: list[dict[str, Any]] = []
    recorded_inner_scopes: set[str] = set()
    selected_configuration_rows: list[dict[str, Any]] = []
    seed_prediction_rows: list[pd.DataFrame] = []

    for fold, (train_local, test_local) in enumerate(outer_splits, start=1):
        train_indices = prepared.development_indices[train_local]
        test_indices = prepared.development_indices[test_local]
        validate_temporal_boundary(
            prepared,
            train_indices,
            test_indices,
            f"Outer fold {fold}",
        )
        train_rows = origins.iloc[train_indices]
        test_rows = origins.iloc[test_indices]
        outer_manifest_rows.append(
            {
                "Fold": fold,
                "Train Start": train_rows["Date"].min(),
                "Train End": train_rows["Date"].max(),
                "Train Target End": train_rows["Target_Date"].max(),
                "Test Start": test_rows["Date"].min(),
                "Test End": test_rows["Date"].max(),
                "Train Rows": len(train_rows),
                "Test Rows": len(test_rows),
                "Gap Rows": PURGE_ROWS,
            }
        )

        for model_name in MAIN_MODELS:
            print(f"Outer fold {fold}: tuning {model_name}", flush=True)
            (
                selected_config,
                selected_epoch,
                model_tuning_rows,
                model_inner_manifest,
            ) = tune_model(
                prepared,
                protocol,
                train_indices,
                model_name,
                scope=f"Outer fold {fold}",
                outer_fold=fold,
            )
            tuning_rows.extend(model_tuning_rows)
            scope_name = f"Outer fold {fold}"
            if scope_name not in recorded_inner_scopes:
                inner_manifest_rows.extend(model_inner_manifest)
                recorded_inner_scopes.add(scope_name)
            selected_configuration_rows.append(
                {
                    "Scope": f"Outer fold {fold}",
                    "Outer Fold": fold,
                    "Model": model_name,
                    "Selected Configuration": stable_json(selected_config),
                    "Selected Epoch": selected_epoch,
                    "Selection Metric": "Mean inner-fold Balanced Accuracy",
                    "Audit Used": False,
                }
            )
            (
                preprocessor,
                _,
                probability,
                seed_rows,
                elapsed,
            ) = fit_predict_ensemble(
                prepared,
                train_indices,
                test_indices,
                model_name,
                selected_config,
                selected_epoch,
                retain_models=False,
            )
            prediction_frame = append_prediction_frame(
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
                fold_metric_row(
                    prediction_frame,
                    prepared,
                    train_indices,
                    test_indices,
                    elapsed,
                    selected_config,
                    selected_epoch,
                )
            )
            for seed_row in seed_rows:
                seed_frame = prediction_frame[
                    [
                        "Date",
                        "Target_Date",
                        "Eligible_Row_ID",
                        "Phase_Offset",
                        "Actual",
                    ]
                ].copy()
                seed_frame["Period"] = "Development OOF"
                seed_frame["Fold"] = fold
                seed_frame["Model"] = model_name
                seed_frame["Seed"] = seed_row["Seed"]
                seed_frame["Configuration"] = seed_row["Configuration"]
                seed_frame["Selected Epoch"] = seed_row["Selected Epoch"]
                seed_frame["Probability"] = seed_row["Probability Array"]
                seed_prediction_rows.append(seed_frame)

        y_train = train_rows["Weekly_Trend_Direction"].astype(int).to_numpy()
        actual_test = test_rows["Weekly_Trend_Direction"].astype(int).to_numpy()
        baseline_probabilities = {
            "Training-majority": np.full(
                len(test_indices),
                float(y_train.mean()),
            ),
            "Weekly-trend persistence": test_rows[
                "Weekly_Persistence"
            ].astype(float).to_numpy(),
            "EMA8/13 state": test_rows[
                "EMA8_13_State"
            ].astype(float).to_numpy(),
        }
        for baseline, probability in baseline_probabilities.items():
            frame = append_prediction_frame(
                prediction_frames,
                prepared,
                test_indices,
                baseline,
                "Transparent baseline",
                "Development OOF",
                fold,
                probability,
            )
            fold_metric_rows.append(
                fold_metric_row(
                    frame,
                    prepared,
                    train_indices,
                    test_indices,
                    0.0,
                    None,
                    None,
                )
            )

        bridge_probability = bridge_logistic_predictions(
            prepared,
            train_indices,
            test_indices,
        )
        bridge_frame = append_prediction_frame(
            prediction_frames,
            prepared,
            test_indices,
            BRIDGE_MODEL,
            "RQ2 bridge control",
            "Development OOF",
            fold,
            bridge_probability,
        )
        fold_metric_rows.append(
            fold_metric_row(
                bridge_frame,
                prepared,
                train_indices,
                test_indices,
                0.0,
                {"C": 0.5, "lookback": 1},
                None,
            )
        )
        del actual_test

    outer_manifest = pd.DataFrame(outer_manifest_rows)
    outer_manifest.to_csv(OUTPUTS / "rq3_outer_folds.csv", index=False)
    preliminary_fold_metrics = pd.DataFrame(fold_metric_rows)
    selection = one_standard_error_selection(
        preliminary_fold_metrics,
        protocol,
    )
    selection.to_csv(OUTPUTS / "rq3_one_se_selection.csv", index=False)
    selected_model = selection.loc[
        selection["Selected for RQ4"], "Model"
    ].iloc[0]
    print(f"Development-only selected model: {selected_model}", flush=True)

    final_configurations: dict[str, dict[str, Any]] = {}
    final_epochs: dict[str, int | None] = {}
    selected_fit_preprocessor: ChannelPreprocessor | None = None
    selected_fit_models: list[Any] = []
    audit_indices = prepared.audit_indices
    full_development = prepared.development_indices

    for model_name in MAIN_MODELS:
        print(f"Full development: tuning {model_name}", flush=True)
        (
            selected_config,
            selected_epoch,
            model_tuning_rows,
            model_inner_manifest,
        ) = tune_model(
            prepared,
            protocol,
            full_development,
            model_name,
            scope="Full development",
            outer_fold=None,
        )
        tuning_rows.extend(model_tuning_rows)
        if "Full development" not in recorded_inner_scopes:
            inner_manifest_rows.extend(model_inner_manifest)
            recorded_inner_scopes.add("Full development")
        final_configurations[model_name] = selected_config
        final_epochs[model_name] = selected_epoch
        selected_configuration_rows.append(
            {
                "Scope": "Full development",
                "Outer Fold": None,
                "Model": model_name,
                "Selected Configuration": stable_json(selected_config),
                "Selected Epoch": selected_epoch,
                "Selection Metric": "Mean inner-fold Balanced Accuracy",
                "Audit Used": False,
            }
        )
        retain = model_name == selected_model
        (
            preprocessor,
            fitted_models,
            probability,
            seed_rows,
            elapsed,
        ) = fit_predict_ensemble(
            prepared,
            full_development,
            audit_indices,
            model_name,
            selected_config,
            selected_epoch,
            retain_models=retain,
        )
        if retain:
            selected_fit_preprocessor = preprocessor
            selected_fit_models = fitted_models
        frame = append_prediction_frame(
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
            fold_metric_row(
                frame,
                prepared,
                full_development,
                audit_indices,
                elapsed,
                selected_config,
                selected_epoch,
            )
        )
        for seed_row in seed_rows:
            seed_frame = frame[
                [
                    "Date",
                    "Target_Date",
                    "Eligible_Row_ID",
                    "Phase_Offset",
                    "Actual",
                ]
            ].copy()
            seed_frame["Period"] = "Retrospective audit"
            seed_frame["Fold"] = "Audit"
            seed_frame["Model"] = model_name
            seed_frame["Seed"] = seed_row["Seed"]
            seed_frame["Configuration"] = seed_row["Configuration"]
            seed_frame["Selected Epoch"] = seed_row["Selected Epoch"]
            seed_frame["Probability"] = seed_row["Probability Array"]
            seed_prediction_rows.append(seed_frame)

    audit_rows = origins.iloc[audit_indices]
    development_rows = origins.iloc[full_development]
    y_development = development_rows["Weekly_Trend_Direction"].astype(int)
    audit_baselines = {
        "Training-majority": np.full(
            len(audit_indices),
            float(y_development.mean()),
        ),
        "Weekly-trend persistence": audit_rows[
            "Weekly_Persistence"
        ].astype(float).to_numpy(),
        "EMA8/13 state": audit_rows[
            "EMA8_13_State"
        ].astype(float).to_numpy(),
    }
    for baseline, probability in audit_baselines.items():
        frame = append_prediction_frame(
            prediction_frames,
            prepared,
            audit_indices,
            baseline,
            "Transparent baseline",
            "Retrospective audit",
            "Audit",
            probability,
        )
        fold_metric_rows.append(
            fold_metric_row(
                frame,
                prepared,
                full_development,
                audit_indices,
                0.0,
                None,
                None,
            )
        )
    bridge_probability = bridge_logistic_predictions(
        prepared,
        full_development,
        audit_indices,
    )
    bridge_frame = append_prediction_frame(
        prediction_frames,
        prepared,
        audit_indices,
        BRIDGE_MODEL,
        "RQ2 bridge control",
        "Retrospective audit",
        "Audit",
        bridge_probability,
    )
    fold_metric_rows.append(
        fold_metric_row(
            bridge_frame,
            prepared,
            full_development,
            audit_indices,
            0.0,
            {"C": 0.5, "lookback": 1},
            None,
        )
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    inner_manifest = (
        pd.DataFrame(inner_manifest_rows)
        .drop_duplicates(
            ["Scope", "Outer Fold", "Inner Fold"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    selected_configurations = pd.DataFrame(selected_configuration_rows)
    seed_predictions = pd.concat(seed_prediction_rows, ignore_index=True)

    predictions.to_csv(OUTPUTS / "rq3_all_predictions.csv", index=False)
    predictions.loc[predictions["Period"].eq("Development OOF")].to_csv(
        OUTPUTS / "rq3_oof_predictions.csv",
        index=False,
    )
    predictions.loc[predictions["Period"].eq("Retrospective audit")].to_csv(
        OUTPUTS / "rq3_audit_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(OUTPUTS / "rq3_fold_metrics.csv", index=False)
    tuning.to_csv(OUTPUTS / "rq3_inner_tuning_results.csv", index=False)
    inner_manifest.to_csv(OUTPUTS / "rq3_inner_folds.csv", index=False)
    selected_configurations.to_csv(
        OUTPUTS / "rq3_selected_configurations.csv",
        index=False,
    )
    seed_predictions.to_csv(
        OUTPUTS / "rq3_seed_level_predictions.csv",
        index=False,
    )

    summary = pooled_summary(predictions, fold_metrics)
    summary.to_csv(OUTPUTS / "rq3_performance_summary.csv", index=False)
    bootstrap = bootstrap_all(predictions)
    bootstrap.to_csv(
        OUTPUTS / "rq3_paired_block_bootstrap.csv",
        index=False,
    )
    phases = phase_metrics(predictions)
    phases.to_csv(OUTPUTS / "rq3_phase_offset_metrics.csv", index=False)

    phase_summary = (
        phases.groupby(["Period", "Comparator"], as_index=False)
        .agg(
            Phase_Count=("Phase Offset", "nunique"),
            Total_Rows=("Rows", "sum"),
            Mean_Accuracy=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            Minimum_Accuracy=("Accuracy", "min"),
            Maximum_Accuracy=("Accuracy", "max"),
            Mean_Balanced_Accuracy=("Balanced Accuracy", "mean"),
            Balanced_Accuracy_SD=("Balanced Accuracy", "std"),
            Minimum_Balanced_Accuracy=("Balanced Accuracy", "min"),
            Maximum_Balanced_Accuracy=("Balanced Accuracy", "max"),
        )
    )
    phase_summary.to_csv(
        OUTPUTS / "rq3_phase_offset_summary.csv",
        index=False,
    )

    save_figures(
        summary,
        fold_metrics,
        bootstrap,
        predictions,
        selected_model,
    )

    if selected_fit_preprocessor is None or not selected_fit_models:
        raise AssertionError("The selected final model bundle was not retained.")
    selected_model_paths = save_selected_model_artifacts(
        selected_model,
        final_configurations[selected_model],
        final_epochs[selected_model],
        selected_fit_preprocessor,
        selected_fit_models,
    )
    selected_oof = predictions.loc[
        predictions["Comparator"].eq(selected_model)
        & predictions["Period"].eq("Development OOF")
    ].copy()
    selected_audit = predictions.loc[
        predictions["Comparator"].eq(selected_model)
        & predictions["Period"].eq("Retrospective audit")
    ].copy()
    selected_oof_path = FREEZE / "rq3_selected_oof_predictions.csv"
    selected_audit_path = FREEZE / "rq3_selected_audit_predictions.csv"
    selected_oof.to_csv(selected_oof_path, index=False)
    selected_audit.to_csv(selected_audit_path, index=False)
    selected_model_paths.extend([selected_oof_path, selected_audit_path])

    rq4_signal = pd.concat([selected_oof, selected_audit], ignore_index=True)
    rq4_signal = rq4_signal.rename(
        columns={
            "Comparator": "Selected Model",
            "Prediction": "Binary Signal",
        }
    )
    rq4_signal["Threshold"] = THRESHOLD
    rq4_signal["Protocol SHA256"] = file_sha256(PROTOCOL)
    rq4_signal_path = FREEZE / "rq3_frozen_rq4_signal.csv"
    rq4_signal[
        [
            "Date",
            "Target_Date",
            "Eligible_Row_ID",
            "Phase_Offset",
            "Period",
            "Fold",
            "Actual",
            "Probability",
            "Binary Signal",
            "Selected Model",
            "Threshold",
            "Protocol SHA256",
        ]
    ].to_csv(rq4_signal_path, index=False)
    selected_model_paths.append(rq4_signal_path)

    freeze_manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "run_script_sha256": file_sha256(Path(__file__).resolve()),
        "input_sha256": file_sha256(DATA),
        "upstream_rq2_protocol_sha256": file_sha256(UPSTREAM_RQ2_PROTOCOL),
        "selected_model": selected_model,
        "selection_data": "Development outer-fold OOF only",
        "threshold": THRESHOLD,
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in selected_model_paths
        },
    }
    freeze_manifest_path = FREEZE / "rq3_freeze_manifest.json"
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "run_script_sha256": file_sha256(Path(__file__).resolve()),
        "input_sha256": file_sha256(DATA),
        "status": protocol["status"],
        "features": FEATURES,
        "lookback": LOOKBACK,
        "sequence_shape": [LOOKBACK, len(FEATURES)],
        "eligible_origins": len(origins),
        "development_origins": len(full_development),
        "development_start": development_rows["Date"].min().strftime("%Y-%m-%d"),
        "development_end": development_rows["Date"].max().strftime("%Y-%m-%d"),
        "development_oof_rows_per_comparator": OUTER_SPLITS * OUTER_TEST_SIZE,
        "audit_origins": len(audit_indices),
        "audit_start": audit_rows["Date"].min().strftime("%Y-%m-%d"),
        "audit_end": audit_rows["Date"].max().strftime("%Y-%m-%d"),
        "main_models": MAIN_MODELS,
        "stochastic_seeds": SEEDS,
        "selected_model": selected_model,
        "selected_final_configuration": final_configurations[selected_model],
        "selected_final_epoch": final_epochs[selected_model],
        "selection_rule": "development-only one-standard-error",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_block_length": BLOCK_LENGTH,
        "freeze_manifest_sha256": file_sha256(freeze_manifest_path),
        "total_runtime_seconds": time.perf_counter() - started_all,
        "claim_boundary": protocol["claim_boundary"],
    }
    (OUTPUTS / "rq3_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "Comparator",
        "Development Mean Fold Accuracy",
        "Development Mean Fold Balanced Accuracy",
        "Development Balanced Accuracy SD",
        "Audit Accuracy",
        "Audit Balanced Accuracy",
        "Audit Macro F1",
        "Audit MCC",
    ]
    print("\nRQ3 performance summary")
    print(
        summary.loc[summary["Comparator"].isin(MAIN_MODELS + BASELINES)][
            display_columns
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nDevelopment-only one-SE selection")
    print(
        selection.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(
        f"\nCompleted in {metadata['total_runtime_seconds']:.1f} seconds.",
        flush=True,
    )


if __name__ == "__main__":
    main()
