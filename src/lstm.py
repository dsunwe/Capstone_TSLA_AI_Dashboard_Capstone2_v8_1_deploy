from __future__ import annotations

from dataclasses import dataclass
import os
import random
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    CANDIDATE_FEATURES,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_LOOKBACK,
    DEFAULT_LSTM_SEEDS,
    DEFAULT_LSTM_UNITS,
    LSTM_MODEL_NAME,
    SELECTED_FEATURES,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)
from .ml import (
    MLResult,
    _classification_metrics,
    _feature_set_hash,
    _fold_features,
    _walk_forward_inputs,
)
from .temporal import TemporalSplitPlan, build_temporal_split_plan, rows_for_dates


@dataclass
class SequencePreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_names: list[str]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(frame[self.feature_names])
        return self.scaler.transform(imputed).astype(np.float32)


@dataclass
class SequenceFold:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_rows: pd.DataFrame
    preprocessor: SequencePreprocessor
    train_anchor_indices: np.ndarray
    test_anchor_indices: np.ndarray


def _tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ImportError(
            "TensorFlow is required for live LSTM training. Install "
            "requirements-local.txt first."
        ) from exc
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return tf


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)


def build_lstm_model(
    *,
    lookback: int,
    feature_count: int,
    units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_LSTM_DROPOUT,
    learning_rate: float = 0.001,
    random_state: int = 42,
):
    tf = _tensorflow()
    tf.keras.backend.clear_session()
    set_deterministic_seed(random_state)
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, feature_count)),
            tf.keras.layers.LSTM(
                units,
                activation="tanh",
                dropout=dropout,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizer,
            ),
            tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="next_day_direction_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def fit_sequence_preprocessor(
    model_data: pd.DataFrame,
    feature_names: list[str],
    training_row_indices: np.ndarray,
) -> SequencePreprocessor:
    """Fit on unique training rows, not duplicated overlapping sequence values."""
    training_frame = model_data.iloc[training_row_indices]
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(training_frame[feature_names])
    scaler = StandardScaler()
    scaler.fit(imputed)
    return SequencePreprocessor(
        imputer=imputer,
        scaler=scaler,
        feature_names=list(feature_names),
    )


def build_sequences(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    anchor_indices: Iterable[int],
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    sequence_labels: list[int] = []
    valid_anchors: list[int] = []
    for anchor in np.asarray(list(anchor_indices), dtype=int):
        start = anchor - lookback + 1
        if start < 0:
            continue
        sequences.append(scaled_features[start : anchor + 1])
        sequence_labels.append(int(labels[anchor]))
        valid_anchors.append(int(anchor))
    if not sequences:
        return (
            np.empty((0, lookback, scaled_features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=int),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.int32),
        np.asarray(valid_anchors, dtype=int),
    )


def prepare_sequence_fold(
    model_data: pd.DataFrame,
    feature_names: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    lookback: int,
    target_column: str = TARGET_DIRECTION_COLUMN,
) -> SequenceFold:
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    preprocessor = fit_sequence_preprocessor(model_data, feature_names, train_indices)
    scaled = preprocessor.transform(model_data)
    if target_column not in model_data.columns:
        raise KeyError(f"Missing sequence target column: {target_column}")
    labels = model_data[target_column].astype(int).to_numpy()
    x_train, y_train, train_anchors = build_sequences(
        scaled, labels, train_indices, lookback=lookback
    )
    x_test, y_test, test_anchors = build_sequences(
        scaled, labels, test_indices, lookback=lookback
    )
    return SequenceFold(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        test_rows=model_data.iloc[test_anchors].copy(),
        preprocessor=preprocessor,
        train_anchor_indices=train_anchors,
        test_anchor_indices=test_anchors,
    )


def chronological_train_validation_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x_train) < 40:
        raise ValueError("At least 40 training sequences are required for LSTM validation.")
    validation_start = int(len(x_train) * (1 - validation_fraction))
    validation_start = min(max(validation_start, 20), len(x_train) - 10)
    # Two sequence anchors are purged because an anchor-t label uses
    # The five-session label reaches through Close[t+5].
    return (
        x_train[: validation_start - TARGET_PURGE_ROWS],
        y_train[: validation_start - TARGET_PURGE_ROWS],
        x_train[validation_start:],
        y_train[validation_start:],
    )


def _class_weights(y_train: np.ndarray) -> dict[int, float] | None:
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, values)}


def _fit_one_seed(
    fold: SequenceFold,
    *,
    lookback: int,
    units: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any], pd.DataFrame]:
    tf = _tensorflow()
    x_fit, y_fit, x_validation, y_validation = chronological_train_validation_split(
        fold.x_train, fold.y_train
    )
    model = build_lstm_model(
        lookback=lookback,
        feature_count=fold.x_train.shape[2],
        units=units,
        dropout=dropout,
        learning_rate=learning_rate,
        random_state=seed,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        min_delta=1e-4,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    started = perf_counter()
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        callbacks=[early_stopping],
        class_weight=_class_weights(y_fit),
    )
    validation_seconds = perf_counter() - started
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    final_model = build_lstm_model(
        lookback=lookback,
        feature_count=fold.x_train.shape[2],
        units=units,
        dropout=dropout,
        learning_rate=learning_rate,
        random_state=seed,
    )
    started = perf_counter()
    final_model.fit(
        fold.x_train,
        fold.y_train,
        epochs=best_epoch,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
        class_weight=_class_weights(fold.y_train),
    )
    final_fit_seconds = perf_counter() - started
    started = perf_counter()
    probability = final_model.predict(fold.x_test, verbose=0).reshape(-1)
    inference_seconds = perf_counter() - started
    summary = {
        "Seed": seed,
        "Best Epoch": best_epoch,
        "Epochs Evaluated": len(history.history["loss"]),
        "Validation Training Time (s)": validation_seconds,
        "Final Fit Time (s)": final_fit_seconds,
        "Inference Time (s)": inference_seconds,
    }
    history_table = pd.DataFrame(history.history)
    history_table.insert(0, "Epoch", np.arange(1, len(history_table) + 1))
    history_table.insert(0, "Seed", seed)
    return final_model, probability, summary, history_table


def fit_lstm_fold(
    fold: SequenceFold,
    *,
    lookback: int = DEFAULT_LSTM_LOOKBACK,
    units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_LSTM_DROPOUT,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    learning_rate: float = 0.001,
    seeds: Iterable[int] = DEFAULT_LSTM_SEEDS,
    probability_threshold: float = 0.50,
    return_seed_probabilities: bool = False,
):
    models: list[Any] = []
    probabilities: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for seed in seeds:
        model, probability, summary, history = _fit_one_seed(
            fold,
            lookback=lookback,
            units=units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=int(seed),
        )
        models.append(model)
        probability = np.asarray(probability, dtype=np.float64)
        probabilities.append(probability)
        prediction = (probability >= probability_threshold).astype(int)
        seed_metrics = _classification_metrics(
            fold.y_test,
            prediction,
            probability,
        )
        summary.update(
            {
                "Test Accuracy": seed_metrics["Accuracy"],
                "Test Macro F1": seed_metrics["Macro F1"],
                "Test PR-AUC": seed_metrics["PR-AUC"],
                "Test Brier Score": seed_metrics["Brier Score"],
            }
        )
        summaries.append(summary)
        histories.append(history)
    ensemble_probability = np.mean(np.vstack(probabilities), axis=0)
    result = (
        models,
        ensemble_probability,
        pd.DataFrame(summaries),
        pd.concat(histories, ignore_index=True),
    )
    if return_seed_probabilities:
        return (*result, np.vstack(probabilities))
    return result


def _ensemble_predict(models: list[Any], x_data: np.ndarray) -> np.ndarray:
    return np.mean(
        np.vstack([model.predict(x_data, verbose=0).reshape(-1) for model in models]),
        axis=0,
    )


def sequence_feature_occlusion(
    models: list[Any],
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    base_probability = _ensemble_predict(models, x_test)
    base_prediction = (base_probability >= threshold).astype(int)
    base_score = balanced_accuracy_score(y_test, base_prediction)
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        occluded = x_test.copy()
        # Zero is the training mean after standardisation.
        occluded[:, :, feature_index] = 0.0
        probability = _ensemble_predict(models, occluded)
        prediction = (probability >= threshold).astype(int)
        score = balanced_accuracy_score(y_test, prediction)
        rows.append(
            {
                "Feature": feature_name,
                "Importance": float(base_score - score),
                "Model": LSTM_MODEL_NAME,
                "Method": "Sequence-channel occlusion",
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(
        drop=True
    )


def _fixed_holdout_indices(row_count: int, train_size: float) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(row_count * train_size)
    split_index = min(max(split_index, 30), row_count - 10)
    # Purge five anchors because the classification label reaches Close[t+5].
    return (
        np.arange(0, split_index - TARGET_PURGE_ROWS),
        np.arange(split_index, row_count),
    )


def run_lstm_classification(
    engineered: pd.DataFrame,
    *,
    train_size: float = 0.80,
    probability_threshold: float = 0.50,
    feature_names: list[str] | None = None,
    lookback: int = DEFAULT_LSTM_LOOKBACK,
    units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_LSTM_DROPOUT,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    learning_rate: float = 0.001,
    seeds: Iterable[int] = DEFAULT_LSTM_SEEDS,
    split_plan: TemporalSplitPlan | None = None,
) -> MLResult | None:
    features = list(feature_names or SELECTED_FEATURES)
    plan = split_plan or build_temporal_split_plan(
        engineered,
        train_size=train_size,
    )
    model_data = rows_for_dates(
        engineered,
        plan.eligible_dates,
        required_columns=[
            *features,
            "Price",
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
        ],
    ).reset_index(drop=True)
    if len(model_data) < max(100, lookback + 50):
        return None
    train_indices = np.arange(0, len(plan.train_dates))
    test_start_index = len(plan.train_dates) + len(plan.gap_dates)
    test_indices = np.arange(test_start_index, len(model_data))
    fold = prepare_sequence_fold(
        model_data,
        features,
        train_indices,
        test_indices,
        lookback=lookback,
    )
    if len(fold.x_train) < 40 or len(fold.x_test) < 10:
        return None

    started = perf_counter()
    models, probability, seed_results, training_history = fit_lstm_fold(
        fold,
        lookback=lookback,
        units=units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seeds=seeds,
    )
    total_seconds = perf_counter() - started
    prediction = (probability >= probability_threshold).astype(int)
    metrics = _classification_metrics(fold.y_test, prediction, probability)
    results = fold.test_rows[["Date", "Price", TARGET_RETURN_COLUMN]].copy()
    results["Actual Direction"] = fold.y_test
    results["Predicted Probability"] = probability
    results["Predicted Direction"] = prediction
    results["Signal"] = prediction
    matrix = pd.DataFrame(
        pd.crosstab(
            pd.Series(fold.y_test, name="Actual"),
            pd.Series(prediction, name="Predicted"),
            dropna=False,
        )
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    matrix.index = ["Actual 0", "Actual 1"]
    matrix.columns = ["Predicted 0", "Predicted 1"]
    importance = sequence_feature_occlusion(
        models,
        fold.x_test,
        fold.y_test,
        features,
        threshold=probability_threshold,
    )
    selected_parameters = {
        "lookback": lookback,
        "units": units,
        "dropout": dropout,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seeds": ",".join(str(seed) for seed in seeds),
    }
    return MLResult(
        model=models,
        model_name=LSTM_MODEL_NAME,
        results=results.reset_index(drop=True),
        metrics=metrics,
        baseline_comparison=pd.DataFrame(),
        confusion_matrix=matrix,
        feature_importance=importance,
        train_rows=len(fold.y_train),
        test_rows=len(fold.y_test),
        train_start_date=pd.Timestamp(model_data.iloc[train_indices]["Date"].min()),
        train_end_date=pd.Timestamp(model_data.iloc[train_indices]["Date"].max()),
        test_start_date=pd.Timestamp(fold.test_rows["Date"].min()),
        test_end_date=pd.Timestamp(fold.test_rows["Date"].max()),
        feature_names=features,
        tuning_results=seed_results,
        selected_parameters=selected_parameters,
        training_time_seconds=float(
            seed_results[
                ["Validation Training Time (s)", "Final Fit Time (s)"]
            ].to_numpy().sum()
        ),
        inference_time_seconds=float(seed_results["Inference Time (s)"].sum()),
        training_history=training_history,
        preprocessor=fold.preprocessor,
        model_family="sequence",
        lookback=lookback,
        seed_results=seed_results,
        parameter_count=int(models[0].count_params()),
        wall_time_seconds=total_seconds,
        split_plan_id=plan.plan_id,
        split_plan_hash=plan.plan_hash,
        purge_start_date=plan.gap_start_date,
        purge_end_date=plan.gap_end_date,
    )


def walk_forward_lstm_validate(
    engineered: pd.DataFrame,
    *,
    n_splits: int = 5,
    probability_threshold: float = 0.50,
    feature_names: list[str] | None = None,
    lookback: int = DEFAULT_LSTM_LOOKBACK,
    units: int = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_LSTM_DROPOUT,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    seeds: Iterable[int] = DEFAULT_LSTM_SEEDS,
    foldwise_feature_selection: bool = False,
    candidate_features: list[str] | None = None,
    top_k_features: int = 15,
    return_seed_details: bool = False,
):
    features = list(feature_names or SELECTED_FEATURES)
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one deterministic LSTM seed is required.")
    model_data, selection_candidates = _walk_forward_inputs(
        engineered,
        features,
        foldwise_feature_selection=foldwise_feature_selection,
        candidate_features=candidate_features or CANDIDATE_FEATURES,
    )
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=TARGET_PURGE_ROWS)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    seed_result_frames: list[pd.DataFrame] = []
    seed_prediction_frames: list[pd.DataFrame] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(model_data), start=1
    ):
        fold_features, feature_protocol = _fold_features(
            model_data.iloc[train_indices],
            features,
            selection_candidates,
            foldwise_feature_selection=foldwise_feature_selection,
            top_k_features=top_k_features,
            random_state=int(seed_values[0]),
        )
        fold = prepare_sequence_fold(
            model_data,
            fold_features,
            train_indices,
            test_indices,
            lookback=lookback,
        )
        (
            models,
            probability,
            seed_results,
            _,
            seed_probabilities,
        ) = fit_lstm_fold(
            fold,
            lookback=lookback,
            units=units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            seeds=seed_values,
            probability_threshold=probability_threshold,
            return_seed_probabilities=True,
        )
        prediction = (probability >= probability_threshold).astype(int)
        metrics = _classification_metrics(fold.y_test, prediction, probability)
        majority_class = int(pd.Series(fold.y_train).mode().iloc[0])
        majority_prediction = np.full(len(fold.y_test), majority_class, dtype=int)
        majority_probability = np.full(len(fold.y_test), float(fold.y_train.mean()))
        majority_metrics = _classification_metrics(
            fold.y_test, majority_prediction, majority_probability
        )
        metric_rows.append(
            {
                "Fold": fold_number,
                "Train Rows": len(fold.y_train),
                "Test Rows": len(fold.y_test),
                "Train End": model_data.iloc[train_indices]["Date"].max(),
                "Test Start": fold.test_rows["Date"].min(),
                "Test End": fold.test_rows["Date"].max(),
                "Purge Start": model_data.iloc[train_indices[-1] + 1]["Date"],
                "Purge End": model_data.iloc[test_indices[0] - 1]["Date"],
                "Probability Threshold": probability_threshold,
                "Feature Protocol": feature_protocol,
                "Selected Feature Count": len(fold_features),
                "Selected Features": ", ".join(fold_features),
                "Feature Set Hash": _feature_set_hash(fold_features),
                **metrics,
                "Training Time (s)": float(
                    seed_results[
                        ["Validation Training Time (s)", "Final Fit Time (s)"]
                    ].to_numpy().sum()
                ),
                "Inference Time (s)": float(
                    seed_results["Inference Time (s)"].sum()
                ),
                "Majority Baseline Accuracy": majority_metrics["Accuracy"],
                "Majority Baseline F1-score": majority_metrics["F1-score"],
                "Accuracy Improvement": metrics["Accuracy"]
                - majority_metrics["Accuracy"],
                "F1-score Improvement": metrics["F1-score"]
                - majority_metrics["F1-score"],
                "Lookback": lookback,
                "Mean Best Epoch": seed_results["Best Epoch"].mean(),
                "Nested Tuning Protocol": (
                    "Prespecified LSTM architecture; fold-local purged "
                    "chronological early stopping"
                ),
                "Selected Parameters": str(
                    {
                        "lookback": lookback,
                        "units": units,
                        "dropout": dropout,
                        "batch_size": batch_size,
                        "learning_rate": 0.001,
                    }
                ),
                "Inner Splits": 1,
                "Candidate Count": 1,
                "Selection Metric": "Validation loss for epoch selection",
                "LSTM Seeds": ", ".join(str(seed) for seed in seed_values),
                "Seed Count": len(seed_values),
                "Ensemble Rule": "Arithmetic mean of seed probabilities",
                "Seed Accuracy Mean": seed_results["Test Accuracy"].mean(),
                "Seed Accuracy Std": seed_results["Test Accuracy"].std(ddof=0),
                "Seed Macro F1 Mean": seed_results["Test Macro F1"].mean(),
                "Seed Macro F1 Std": seed_results["Test Macro F1"].std(ddof=0),
            }
        )
        frame = fold.test_rows[["Date", "Price", TARGET_RETURN_COLUMN]].copy()
        frame["Actual Direction"] = fold.y_test
        frame["Predicted Probability"] = probability
        frame["Predicted Direction"] = prediction
        frame["Signal"] = prediction
        frame["Probability Threshold"] = probability_threshold
        frame["Fold"] = fold_number
        frame["Train End"] = model_data.iloc[train_indices]["Date"].max()
        frame["Purge Start"] = model_data.iloc[train_indices[-1] + 1]["Date"]
        frame["Purge End"] = model_data.iloc[test_indices[0] - 1]["Date"]
        frame["Feature Protocol"] = feature_protocol
        frame["Selected Feature Count"] = len(fold_features)
        frame["Selected Features"] = ", ".join(fold_features)
        frame["Feature Set Hash"] = _feature_set_hash(fold_features)
        frame["Selected Parameters"] = str(
            {
                "lookback": lookback,
                "units": units,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": 0.001,
            }
        )
        frame["LSTM Seeds"] = ", ".join(str(seed) for seed in seed_values)
        frame["Seed Count"] = len(seed_values)
        frame["Ensemble Rule"] = "Arithmetic mean of seed probabilities"
        prediction_frames.append(frame)

        fold_seed_results = seed_results.copy()
        fold_seed_results.insert(0, "Fold", fold_number)
        fold_seed_results.insert(0, "Model", LSTM_MODEL_NAME)
        fold_seed_results["Train End"] = model_data.iloc[train_indices]["Date"].max()
        fold_seed_results["Test Start"] = fold.test_rows["Date"].min()
        fold_seed_results["Test End"] = fold.test_rows["Date"].max()
        fold_seed_results["Selected Feature Count"] = len(fold_features)
        fold_seed_results["Selected Features"] = ", ".join(fold_features)
        fold_seed_results["Feature Set Hash"] = _feature_set_hash(fold_features)
        fold_seed_results["Lookback"] = lookback
        fold_seed_results["Units"] = units
        fold_seed_results["Dropout"] = dropout
        fold_seed_results["Batch Size"] = batch_size
        seed_result_frames.append(fold_seed_results)

        for seed_index, seed in enumerate(seed_values):
            seed_frame = fold.test_rows[["Date"]].copy()
            seed_frame["Model"] = LSTM_MODEL_NAME
            seed_frame["Fold"] = fold_number
            seed_frame["Seed"] = int(seed)
            seed_frame["Actual Direction"] = fold.y_test
            seed_frame["Predicted Probability"] = seed_probabilities[seed_index]
            seed_frame["Predicted Direction"] = (
                seed_probabilities[seed_index] >= probability_threshold
            ).astype(int)
            seed_prediction_frames.append(seed_frame)
        del models
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if predictions["Date"].duplicated().any():
        raise ValueError("LSTM OOF folds produced duplicate signal dates.")
    metrics = pd.DataFrame(metric_rows)
    if not return_seed_details:
        return metrics, predictions
    seed_results_output = (
        pd.concat(seed_result_frames, ignore_index=True)
        if seed_result_frames
        else pd.DataFrame()
    )
    seed_predictions_output = (
        pd.concat(seed_prediction_frames, ignore_index=True)
        if seed_prediction_frames
        else pd.DataFrame()
    )
    return metrics, predictions, seed_results_output, seed_predictions_output
