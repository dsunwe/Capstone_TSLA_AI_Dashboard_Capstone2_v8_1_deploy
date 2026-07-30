from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CANDIDATE_FEATURES,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_LOOKBACK,
    DEFAULT_LSTM_SEEDS,
    DEFAULT_LSTM_UNITS,
    LSTM_MODEL_NAME,
    MAJORITY_MODEL_NAME,
    MODEL_LEVEL_NAMES,
    SELECTED_FEATURES,
    TABULAR_MODEL_NAMES,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)
from .selection import select_features_from_rows
from .temporal import (
    TemporalSplitPlan,
    build_temporal_split_plan,
    rows_for_dates,
    split_frame_by_plan,
)


@dataclass
class MLResult:
    model: Any
    model_name: str
    results: pd.DataFrame
    metrics: dict[str, float]
    baseline_comparison: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame
    train_rows: int
    test_rows: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    feature_names: list[str] = field(default_factory=list)
    tuning_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_parameters: dict[str, Any] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    training_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocessor: Any = None
    model_family: str = "tabular"
    lookback: int = 1
    seed_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    parameter_count: int = 0
    wall_time_seconds: float = 0.0
    split_plan_id: str | None = None
    split_plan_hash: str | None = None
    purge_start_date: pd.Timestamp | None = None
    purge_end_date: pd.Timestamp | None = None


def _xgboost_classifier(random_state: int, **parameters: Any):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - deployment installs the declared dependency.
        raise ImportError(
            "XGBoost is required for the XGBoost benchmark. Install requirements.txt first."
        ) from exc
    defaults = {
        "n_estimators": 180,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    defaults.update(parameters)
    return XGBClassifier(**defaults)


def build_classifier(
    model_name: str,
    *,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 6,
    model_parameters: dict[str, Any] | None = None,
) -> Pipeline:
    parameters = dict(model_parameters or {})
    if model_name == "Random Forest Classifier":
        parameters.setdefault("n_estimators", n_estimators)
        parameters.setdefault("max_depth", max_depth)
        parameters.setdefault("min_samples_leaf", 2)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        **parameters,
                    ),
                ),
            ]
        )
    if model_name == "XGBoost Classifier":
        parameters.setdefault("n_estimators", n_estimators)
        parameters.setdefault("max_depth", max_depth if max_depth is not None else 3)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", _xgboost_classifier(random_state, **parameters)),
            ]
        )
    if model_name != "Logistic Regression (Baseline)":
        raise ValueError(f"Unsupported model: {model_name}")
    parameters.setdefault("C", 1.0)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="liblinear",
                    **parameters,
                ),
            ),
        ]
    )


def _classification_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    metrics = {
        "Accuracy": float(accuracy_score(actual_array, predicted_array)),
        "Balanced Accuracy": float(balanced_accuracy_score(actual_array, predicted_array)),
        "Precision": float(precision_score(actual_array, predicted_array, zero_division=0)),
        "Recall": float(recall_score(actual_array, predicted_array, zero_division=0)),
        "F1-score": float(f1_score(actual_array, predicted_array, zero_division=0)),
        "Macro F1": float(
            f1_score(actual_array, predicted_array, average="macro", zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(actual_array, predicted_array)),
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
        "Brier Score": np.nan,
    }
    if probability is not None:
        probability_array = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
        metrics["Brier Score"] = float(brier_score_loss(actual_array, probability_array))
        if len(np.unique(actual_array)) == 2:
            metrics["ROC-AUC"] = float(roc_auc_score(actual_array, probability_array))
            metrics["PR-AUC"] = float(
                average_precision_score(actual_array, probability_array)
            )
    return metrics


def get_feature_importance(
    model: Pipeline,
    model_name: str,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    features = feature_names or SELECTED_FEATURES
    fitted_model = model.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
        label = "Importance"
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(fitted_model.coef_[0])
        label = "Absolute Coefficient"
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Model"])
    result = pd.DataFrame({"Feature": features, label: values, "Model": model_name})
    return result.sort_values(label, ascending=False).reset_index(drop=True)


def _parameter_space(model_name: str) -> dict[str, list[Any]]:
    if model_name == "Logistic Regression (Baseline)":
        return {"model__C": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
    if model_name == "Random Forest Classifier":
        return {
            "model__n_estimators": [120, 200, 320],
            "model__max_depth": [3, 5, 7, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", 0.7, 1.0],
        }
    if model_name == "XGBoost Classifier":
        return {
            "model__n_estimators": [100, 180, 260],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def tune_classifier(
    model: Pipeline,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 3,
    n_iter: int = 6,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    """Tune only inside the training period with chronological CV folds."""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=_parameter_space(model_name),
        n_iter=n_iter,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=n_splits, gap=TARGET_PURGE_ROWS),
        random_state=random_state,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    tuning = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    tuning = tuning.rename(
        columns={
            "params": "Parameters",
            "mean_test_score": "Mean CV Macro F1",
            "std_test_score": "CV Standard Deviation",
            "rank_test_score": "Rank",
        }
    ).sort_values("Rank")
    selected = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    return search.best_estimator_, tuning.reset_index(drop=True), selected


def _fit_and_predict(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, float, float]:
    started = perf_counter()
    model.fit(x_train, y_train)
    training_time = perf_counter() - started
    started = perf_counter()
    probability = model.predict_proba(x_test)[:, 1]
    inference_time = perf_counter() - started
    return probability, training_time, inference_time


def _build_baseline_comparison(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    selected_model_name: str,
    selected_prediction: np.ndarray,
    selected_probability: np.ndarray,
    selected_training_time: float,
    selected_inference_time: float,
    probability_threshold: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
) -> pd.DataFrame:
    majority_class = int(y_train.mode().iloc[0])
    majority_prediction = np.full(len(y_test), majority_class, dtype=int)
    majority_probability = np.full(len(y_test), float(y_train.mean()))
    rows = [
        {
            "Model": "Majority Baseline",
            **_classification_metrics(y_test, majority_prediction, majority_probability),
            "Training Time (s)": 0.0,
            "Inference Time (s)": 0.0,
        }
    ]
    model_specs = [
        ("Logistic Regression", "Logistic Regression (Baseline)"),
        ("Random Forest", "Random Forest Classifier"),
        ("XGBoost", "XGBoost Classifier"),
    ]
    for display_name, classifier_name in model_specs:
        if classifier_name == selected_model_name:
            prediction = selected_prediction
            probability = selected_probability
            training_time = selected_training_time
            inference_time = selected_inference_time
        else:
            comparison_model = build_classifier(
                classifier_name,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
            probability, training_time, inference_time = _fit_and_predict(
                comparison_model, x_train, y_train, x_test
            )
            prediction = (probability >= probability_threshold).astype(int)
        rows.append(
            {
                "Model": display_name,
                **_classification_metrics(y_test, prediction, probability),
                "Training Time (s)": training_time,
                "Inference Time (s)": inference_time,
            }
        )
    return pd.DataFrame(rows)


def _resolve_fixed_holdout(
    engineered: pd.DataFrame,
    feature_names: list[str],
    *,
    train_size: float,
    split_plan: TemporalSplitPlan | None,
) -> tuple[TemporalSplitPlan, pd.DataFrame, pd.DataFrame]:
    """Apply one explicit date plan without feature-dependent row deletion."""
    plan = split_plan or build_temporal_split_plan(
        engineered,
        train_size=train_size,
    )
    required = [
        *feature_names,
        "Price",
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
    ]
    train, _, test = split_frame_by_plan(
        engineered,
        plan,
        required_columns=required,
    )
    return plan, train, test


def _feature_set_hash(feature_names: list[str]) -> str:
    serialised = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _walk_forward_inputs(
    engineered: pd.DataFrame,
    fixed_features: list[str],
    *,
    foldwise_feature_selection: bool,
    candidate_features: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one common OOF date universe for either fixed or nested selection."""
    if foldwise_feature_selection:
        available = [
            feature
            for feature in (candidate_features or CANDIDATE_FEATURES)
            if feature in engineered.columns
        ]
    else:
        available = list(fixed_features)
    plan = build_temporal_split_plan(engineered)
    model_data = rows_for_dates(
        engineered,
        plan.eligible_dates,
        required_columns=[
            *available,
            "Price",
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
        ],
    )
    return model_data.reset_index(drop=True), available


def _fold_features(
    training_rows: pd.DataFrame,
    fixed_features: list[str],
    candidate_features: list[str],
    *,
    foldwise_feature_selection: bool,
    top_k_features: int,
    random_state: int,
) -> tuple[list[str], str]:
    if not foldwise_feature_selection:
        return list(fixed_features), "Pre-specified fixed OOF features"
    selection = select_features_from_rows(
        training_rows,
        candidate_features,
        top_k=top_k_features,
        random_state=random_state,
    )
    return (
        selection.selected_features,
        "Fold-wise training-only MI + correlation selection",
    )


def _build_lstm_baseline_comparison(
    engineered: pd.DataFrame,
    lstm_result: MLResult,
    *,
    feature_names: list[str],
    train_size: float,
    probability_threshold: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    split_plan: TemporalSplitPlan | None,
) -> pd.DataFrame:
    """Build a same-window five-model table when LSTM is explicitly selected."""
    plan, train, test = _resolve_fixed_holdout(
        engineered,
        feature_names,
        train_size=train_size,
        split_plan=split_plan,
    )
    y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
    y_test = test[TARGET_DIRECTION_COLUMN].astype(int)
    majority_class = int(y_train.mode().iloc[0])
    majority_prediction = np.full(len(y_test), majority_class, dtype=int)
    majority_probability = np.full(len(y_test), float(y_train.mean()))
    rows: list[dict[str, Any]] = [
        {
            "Model": "Majority Baseline",
            **_classification_metrics(y_test, majority_prediction, majority_probability),
            "Training Time (s)": 0.0,
            "Inference Time (s)": 0.0,
        }
    ]
    for model_name, display_name in [
        ("Logistic Regression (Baseline)", "Logistic Regression"),
        ("Random Forest Classifier", "Random Forest"),
        ("XGBoost Classifier", "XGBoost"),
    ]:
        result = run_classification(
            engineered,
            model_name=model_name,
            train_size=train_size,
            probability_threshold=probability_threshold,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            feature_names=feature_names,
            include_benchmark=False,
            split_plan=plan,
        )
        if result is None:
            continue
        rows.append(
            {
                "Model": display_name,
                **result.metrics,
                "Training Time (s)": result.training_time_seconds,
                "Inference Time (s)": result.inference_time_seconds,
            }
        )
    rows.append(
        {
            "Model": "LSTM",
            **lstm_result.metrics,
            "Training Time (s)": lstm_result.training_time_seconds,
            "Inference Time (s)": lstm_result.inference_time_seconds,
        }
    )
    return pd.DataFrame(rows)


def run_classification(
    engineered: pd.DataFrame,
    *,
    model_name: str = "Random Forest Classifier",
    train_size: float = 0.80,
    probability_threshold: float = 0.50,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 6,
    feature_names: list[str] | None = None,
    tune: bool = False,
    tuning_splits: int = 3,
    tuning_iterations: int = 6,
    include_benchmark: bool = True,
    lstm_lookback: int = DEFAULT_LSTM_LOOKBACK,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
    lstm_epochs: int = DEFAULT_LSTM_EPOCHS,
    lstm_batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    lstm_seeds: list[int] | tuple[int, ...] = tuple(DEFAULT_LSTM_SEEDS),
    split_plan: TemporalSplitPlan | None = None,
) -> MLResult | None:
    features = list(feature_names or SELECTED_FEATURES)
    if model_name == LSTM_MODEL_NAME:
        from .lstm import run_lstm_classification

        result = run_lstm_classification(
            engineered,
            train_size=train_size,
            probability_threshold=probability_threshold,
            feature_names=features,
            lookback=lstm_lookback,
            units=lstm_units,
            dropout=lstm_dropout,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            seeds=lstm_seeds,
            split_plan=split_plan,
        )
        if result is not None and include_benchmark:
            result.baseline_comparison = _build_lstm_baseline_comparison(
                engineered,
                result,
                feature_names=features,
                train_size=train_size,
                probability_threshold=probability_threshold,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                split_plan=split_plan,
            )
        return result

    plan, train, test = _resolve_fixed_holdout(
        engineered,
        features,
        train_size=train_size,
        split_plan=split_plan,
    )
    if len(train) < 30 or len(test) < 10:
        return None
    x_train = train[features]
    y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
    x_test = test[features]
    y_test = test[TARGET_DIRECTION_COLUMN].astype(int)
    if y_train.nunique() < 2 or y_test.empty:
        return None

    model = build_classifier(
        model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    tuning_results = pd.DataFrame()
    selected_parameters: dict[str, Any] = {}
    if tune:
        started = perf_counter()
        model, tuning_results, selected_parameters = tune_classifier(
            model,
            model_name,
            x_train,
            y_train,
            n_splits=tuning_splits,
            n_iter=tuning_iterations,
            random_state=random_state,
        )
        training_time = perf_counter() - started
        started = perf_counter()
        probability = model.predict_proba(x_test)[:, 1]
        inference_time = perf_counter() - started
    else:
        probability, training_time, inference_time = _fit_and_predict(
            model, x_train, y_train, x_test
        )
    prediction = (probability >= probability_threshold).astype(int)
    metrics = _classification_metrics(y_test, prediction, probability)
    if include_benchmark:
        baseline_comparison = _build_baseline_comparison(
            x_train,
            y_train,
            x_test,
            y_test,
            selected_model_name=model_name,
            selected_prediction=prediction,
            selected_probability=probability,
            selected_training_time=training_time,
            selected_inference_time=inference_time,
            probability_threshold=probability_threshold,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
    else:
        baseline_comparison = pd.DataFrame()

    results = test[["Date", "Price", TARGET_RETURN_COLUMN]].copy()
    results["Actual Direction"] = y_test.values
    results["Predicted Direction"] = prediction
    results["Predicted Probability"] = probability
    results["Signal"] = prediction
    confusion = pd.DataFrame(
        confusion_matrix(y_test, prediction, labels=[0, 1]),
        index=["Actual Down/Flat (0)", "Actual Up (1)"],
        columns=["Predicted Down/Flat (0)", "Predicted Up (1)"],
    )
    return MLResult(
        model=model,
        model_name=model_name,
        results=results,
        metrics=metrics,
        baseline_comparison=baseline_comparison,
        confusion_matrix=confusion,
        feature_importance=get_feature_importance(model, model_name, features),
        train_rows=len(train),
        test_rows=len(test),
        train_start_date=pd.Timestamp(train["Date"].iloc[0]),
        train_end_date=pd.Timestamp(train["Date"].iloc[-1]),
        test_start_date=pd.Timestamp(test["Date"].iloc[0]),
        test_end_date=pd.Timestamp(test["Date"].iloc[-1]),
        feature_names=features,
        tuning_results=tuning_results,
        selected_parameters=selected_parameters,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        split_plan_id=plan.plan_id,
        split_plan_hash=plan.plan_hash,
        purge_start_date=plan.gap_start_date,
        purge_end_date=plan.gap_end_date,
    )


def _walk_forward_tabular_evaluate(
    engineered: pd.DataFrame,
    *,
    model_name: str,
    n_splits: int = 5,
    probability_threshold: float = 0.50,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: int | None = 6,
    feature_names: list[str] | None = None,
    model_parameters: dict[str, Any] | None = None,
    foldwise_feature_selection: bool = False,
    candidate_features: list[str] | None = None,
    top_k_features: int = 15,
    nested_tuning: bool = False,
    inner_splits: int = 3,
    tuning_iterations: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = list(feature_names or SELECTED_FEATURES)
    model_data, selection_candidates = _walk_forward_inputs(
        engineered,
        features,
        foldwise_feature_selection=foldwise_feature_selection,
        candidate_features=candidate_features,
    )
    if len(model_data) < max(120, n_splits * 20):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=TARGET_PURGE_ROWS)
    base_model = build_classifier(
        model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        model_parameters=model_parameters,
    )
    rows: list[dict[str, object]] = []
    outputs: list[pd.DataFrame] = []
    tuning_outputs: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(model_data), start=1):
        train = model_data.iloc[train_idx]
        test = model_data.iloc[test_idx]
        fold_features, feature_protocol = _fold_features(
            train,
            features,
            selection_candidates,
            foldwise_feature_selection=foldwise_feature_selection,
            top_k_features=top_k_features,
            random_state=random_state,
        )
        y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
        y_test = test[TARGET_DIRECTION_COLUMN].astype(int)
        if y_train.nunique() < 2 or y_test.empty:
            continue
        model = clone(base_model)
        selected_parameters = dict(model_parameters or {})
        tuning_table = pd.DataFrame()
        tuning_time = 0.0
        if nested_tuning:
            tuning_started = perf_counter()
            model, tuning_table, selected_parameters = tune_classifier(
                model,
                model_name,
                train[fold_features],
                y_train,
                n_splits=inner_splits,
                n_iter=tuning_iterations,
                random_state=random_state,
            )
            tuning_time = perf_counter() - tuning_started
            tuning_table = tuning_table.copy()
            tuning_table.insert(0, "Model", model_name)
            tuning_table.insert(1, "Outer Fold", fold)
            tuning_table.insert(2, "Outer Train End", train["Date"].max())
            tuning_table.insert(3, "Outer Test Start", test["Date"].min())
            tuning_table["Inner Splits"] = inner_splits
            tuning_table["Purge Gap"] = TARGET_PURGE_ROWS
            tuning_table["Selection Metric"] = "Macro F1"
            tuning_table["Selected"] = tuning_table["Rank"].eq(1)
            tuning_table["Outer Selected Feature Count"] = len(fold_features)
            tuning_table["Outer Selected Features"] = ", ".join(fold_features)
            tuning_table["Outer Feature Set Hash"] = _feature_set_hash(fold_features)
            tuning_outputs.append(tuning_table)
        probability, training_time, inference_time = _fit_and_predict(
            model, train[fold_features], y_train, test[fold_features]
        )
        training_time += tuning_time
        prediction = (probability >= probability_threshold).astype(int)
        metrics = _classification_metrics(y_test, prediction, probability)
        majority_class = int(y_train.mode().iloc[0])
        majority_prediction = np.full(len(y_test), majority_class, dtype=int)
        majority_probability = np.full(len(y_test), float(y_train.mean()))
        majority_metrics = _classification_metrics(
            y_test, majority_prediction, majority_probability
        )
        rows.append(
            {
                "Fold": fold,
                "Train Start": train["Date"].iloc[0],
                "Train End": train["Date"].iloc[-1],
                "Test Start": test["Date"].iloc[0],
                "Test End": test["Date"].iloc[-1],
                "Train Rows": len(train),
                "Test Rows": len(test),
                "Purge Start": model_data.iloc[train_idx[-1] + 1]["Date"],
                "Purge End": model_data.iloc[test_idx[0] - 1]["Date"],
                "Probability Threshold": probability_threshold,
                "Feature Protocol": feature_protocol,
                "Selected Feature Count": len(fold_features),
                "Selected Features": ", ".join(fold_features),
                "Feature Set Hash": _feature_set_hash(fold_features),
                "Nested Tuning Protocol": (
                    "Outer-train-only RandomizedSearchCV with "
                    f"TimeSeriesSplit(n_splits={inner_splits}, gap={TARGET_PURGE_ROWS})"
                    if nested_tuning
                    else "Disabled; prespecified/default parameters"
                ),
                "Selected Parameters": str(selected_parameters or "Default"),
                "Inner Splits": inner_splits if nested_tuning else 0,
                "Candidate Count": len(tuning_table) if nested_tuning else 1,
                "Selection Metric": "Macro F1" if nested_tuning else "Not applicable",
                **metrics,
                "Training Time (s)": training_time,
                "Inference Time (s)": inference_time,
                "Majority Baseline Accuracy": majority_metrics["Accuracy"],
                "Accuracy Improvement": metrics["Accuracy"] - majority_metrics["Accuracy"],
                "Majority Baseline F1-score": majority_metrics["F1-score"],
                "F1-score Improvement": metrics["F1-score"] - majority_metrics["F1-score"],
            }
        )
        output = test[
            ["Date", "Price", TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]
        ].copy()
        output["Actual Direction"] = y_test.to_numpy()
        output["Predicted Probability"] = probability
        output["Predicted Direction"] = prediction
        output["Signal"] = prediction
        output["Probability Threshold"] = probability_threshold
        output["Fold"] = fold
        output["Train End"] = train["Date"].max()
        output["Purge Start"] = model_data.iloc[train_idx[-1] + 1]["Date"]
        output["Purge End"] = model_data.iloc[test_idx[0] - 1]["Date"]
        output["Feature Protocol"] = feature_protocol
        output["Selected Feature Count"] = len(fold_features)
        output["Selected Features"] = ", ".join(fold_features)
        output["Feature Set Hash"] = _feature_set_hash(fold_features)
        output["Nested Tuning Protocol"] = (
            "Outer-train-only chronological nested tuning"
            if nested_tuning
            else "Disabled"
        )
        output["Selected Parameters"] = str(selected_parameters or "Default")
        output["Inner Splits"] = inner_splits if nested_tuning else 0
        output["Candidate Count"] = len(tuning_table) if nested_tuning else 1
        output["Selection Metric"] = (
            "Macro F1" if nested_tuning else "Not applicable"
        )
        outputs.append(output.drop(columns=[TARGET_DIRECTION_COLUMN]))
    predictions = (
        pd.concat(outputs, ignore_index=True).sort_values("Date").reset_index(drop=True)
        if outputs
        else pd.DataFrame()
    )
    tuning = (
        pd.concat(tuning_outputs, ignore_index=True)
        if tuning_outputs
        else pd.DataFrame()
    )
    return pd.DataFrame(rows), predictions, tuning


def _walk_forward_majority_evaluate(
    engineered: pd.DataFrame,
    *,
    n_splits: int = 5,
    feature_names: list[str] | None = None,
    foldwise_feature_selection: bool = False,
    candidate_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a fold-wise training-majority signal on the common OOF dates."""
    features = list(feature_names or SELECTED_FEATURES)
    model_data, _ = _walk_forward_inputs(
        engineered,
        features,
        foldwise_feature_selection=foldwise_feature_selection,
        candidate_features=candidate_features,
    )
    if len(model_data) < max(120, n_splits * 20):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=TARGET_PURGE_ROWS)
    metric_rows: list[dict[str, object]] = []
    outputs: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(model_data), start=1):
        train = model_data.iloc[train_idx]
        test = model_data.iloc[test_idx]
        y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
        y_test = test[TARGET_DIRECTION_COLUMN].astype(int)
        majority_class = int(y_train.mode().iloc[0])
        probability = np.full(len(test), float(y_train.mean()))
        prediction = np.full(len(test), majority_class, dtype=int)
        metrics = _classification_metrics(y_test, prediction, probability)
        metric_rows.append(
            {
                "Fold": fold,
                "Train Start": train["Date"].iloc[0],
                "Train End": train["Date"].iloc[-1],
                "Test Start": test["Date"].iloc[0],
                "Test End": test["Date"].iloc[-1],
                "Train Rows": len(train),
                "Test Rows": len(test),
                "Purge Start": model_data.iloc[train_idx[-1] + 1]["Date"],
                "Purge End": model_data.iloc[test_idx[0] - 1]["Date"],
                "Probability Threshold": 0.50,
                "Feature Protocol": "Training-fold majority class; no input features",
                "Selected Feature Count": 0,
                "Selected Features": "",
                "Feature Set Hash": _feature_set_hash([]),
                "Nested Tuning Protocol": "Not applicable; no tunable parameters",
                "Selected Parameters": "Training-fold majority class",
                "Inner Splits": 0,
                "Candidate Count": 1,
                "Selection Metric": "Not applicable",
                **metrics,
                "Training Time (s)": 0.0,
                "Inference Time (s)": 0.0,
                "Majority Baseline Accuracy": metrics["Accuracy"],
                "Accuracy Improvement": 0.0,
                "Majority Baseline F1-score": metrics["F1-score"],
                "F1-score Improvement": 0.0,
            }
        )
        output = test[
            ["Date", "Price", TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]
        ].copy()
        output["Actual Direction"] = output[TARGET_DIRECTION_COLUMN].astype(int)
        output["Predicted Probability"] = probability
        output["Predicted Direction"] = prediction
        output["Signal"] = output["Predicted Direction"]
        output["Probability Threshold"] = 0.50
        output["Fold"] = fold
        output["Train End"] = train["Date"].max()
        output["Purge Start"] = model_data.iloc[train_idx[-1] + 1]["Date"]
        output["Purge End"] = model_data.iloc[test_idx[0] - 1]["Date"]
        output["Feature Protocol"] = "Training-fold majority class; no input features"
        output["Selected Feature Count"] = 0
        output["Selected Features"] = ""
        output["Feature Set Hash"] = _feature_set_hash([])
        output["Nested Tuning Protocol"] = "Not applicable"
        output["Selected Parameters"] = "Training-fold majority class"
        output["Inner Splits"] = 0
        output["Candidate Count"] = 1
        output["Selection Metric"] = "Not applicable"
        outputs.append(output.drop(columns=[TARGET_DIRECTION_COLUMN]))
    predictions = (
        pd.concat(outputs, ignore_index=True).sort_values("Date").reset_index(drop=True)
        if outputs
        else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), predictions, pd.DataFrame()


def walk_forward_evaluate(
    engineered: pd.DataFrame,
    *,
    model_name: str = "Random Forest Classifier",
    n_splits: int = 5,
    probability_threshold: float = 0.50,
    random_state: int = 42,
    n_estimators: int = 150,
    max_depth: int | None = 6,
    feature_names: list[str] | None = None,
    model_parameters: dict[str, Any] | None = None,
    lstm_lookback: int = DEFAULT_LSTM_LOOKBACK,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
    lstm_epochs: int = DEFAULT_LSTM_EPOCHS,
    lstm_batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    lstm_seeds: list[int] | tuple[int, ...] = tuple(DEFAULT_LSTM_SEEDS),
    foldwise_feature_selection: bool = False,
    candidate_features: list[str] | None = None,
    top_k_features: int = 15,
    nested_tuning: bool = False,
    inner_splits: int = 3,
    tuning_iterations: int = 4,
    return_diagnostics: bool = False,
):
    """Return fold metrics and predictions from one shared OOF training pass."""
    features = list(feature_names or SELECTED_FEATURES)
    if model_name == MAJORITY_MODEL_NAME:
        metrics, predictions, tuning = _walk_forward_majority_evaluate(
            engineered,
            n_splits=n_splits,
            feature_names=features,
            foldwise_feature_selection=foldwise_feature_selection,
            candidate_features=candidate_features,
        )
        details = (
            metrics,
            predictions,
            tuning,
            pd.DataFrame(),
            pd.DataFrame(),
        )
    elif model_name == LSTM_MODEL_NAME:
        from .lstm import walk_forward_lstm_validate

        parameters = model_parameters or {}
        (
            metrics,
            predictions,
            seed_results,
            seed_predictions,
        ) = walk_forward_lstm_validate(
            engineered,
            n_splits=n_splits,
            probability_threshold=probability_threshold,
            feature_names=features,
            lookback=int(parameters.get("lookback", lstm_lookback)),
            units=int(parameters.get("units", lstm_units)),
            dropout=float(parameters.get("dropout", lstm_dropout)),
            epochs=int(parameters.get("epochs_max", lstm_epochs)),
            batch_size=int(parameters.get("batch_size", lstm_batch_size)),
            seeds=lstm_seeds,
            foldwise_feature_selection=foldwise_feature_selection,
            candidate_features=candidate_features,
            top_k_features=top_k_features,
            return_seed_details=True,
        )
        details = (
            metrics,
            predictions,
            pd.DataFrame(),
            seed_results,
            seed_predictions,
        )
    else:
        metrics, predictions, tuning = _walk_forward_tabular_evaluate(
            engineered,
            model_name=model_name,
            n_splits=n_splits,
            probability_threshold=probability_threshold,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            feature_names=features,
            model_parameters=model_parameters,
            foldwise_feature_selection=foldwise_feature_selection,
            candidate_features=candidate_features,
            top_k_features=top_k_features,
            nested_tuning=nested_tuning,
            inner_splits=inner_splits,
            tuning_iterations=tuning_iterations,
        )
        details = (
            metrics,
            predictions,
            tuning,
            pd.DataFrame(),
            pd.DataFrame(),
        )
    if return_diagnostics:
        return details
    return details[0], details[1]


def walk_forward_validate(
    engineered: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    metrics, _ = walk_forward_evaluate(engineered, **kwargs)
    return metrics


def walk_forward_oof_predictions(
    engineered: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return only chronologically out-of-fold predictions for backtesting."""
    _, predictions = walk_forward_evaluate(engineered, **kwargs)
    return predictions


def walk_forward_model_suite(
    engineered: pd.DataFrame,
    *,
    model_names: list[str] | tuple[str, ...] = tuple(MODEL_LEVEL_NAMES),
    return_diagnostics: bool = False,
    **kwargs: Any,
):
    """Evaluate every model level on identical folds, targets and OOF dates."""
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    seed_result_frames: list[pd.DataFrame] = []
    seed_prediction_frames: list[pd.DataFrame] = []
    expected_dates: pd.DatetimeIndex | None = None
    for model_name in model_names:
        (
            metrics,
            predictions,
            tuning,
            seed_results,
            seed_predictions,
        ) = walk_forward_evaluate(
            engineered,
            model_name=model_name,
            return_diagnostics=True,
            **kwargs,
        )
        if metrics.empty or predictions.empty:
            raise RuntimeError(f"OOF evaluation produced no rows for {model_name}.")
        dates = pd.DatetimeIndex(pd.to_datetime(predictions["Date"]))
        if expected_dates is None:
            expected_dates = dates
        elif not dates.equals(expected_dates):
            raise ValueError(
                f"OOF dates for {model_name} do not match the shared model suite."
            )
        metrics.insert(0, "Model", model_name)
        predictions.insert(0, "Model", model_name)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        if not tuning.empty:
            tuning_frames.append(tuning)
        if not seed_results.empty:
            seed_result_frames.append(seed_results)
        if not seed_predictions.empty:
            seed_prediction_frames.append(seed_predictions)
    result = (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        (
            pd.concat(tuning_frames, ignore_index=True)
            if tuning_frames
            else pd.DataFrame()
        ),
        (
            pd.concat(seed_result_frames, ignore_index=True)
            if seed_result_frames
            else pd.DataFrame()
        ),
        (
            pd.concat(seed_prediction_frames, ignore_index=True)
            if seed_prediction_frames
            else pd.DataFrame()
        ),
    )
    if return_diagnostics:
        return result
    return result[0], result[1]


def fixed_holdout_model_suite(
    engineered: pd.DataFrame,
    *,
    feature_names: list[str],
    split_plan: TemporalSplitPlan,
    probability_threshold: float = 0.50,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 6,
    lstm_result: MLResult | None = None,
    lstm_lookback: int = DEFAULT_LSTM_LOOKBACK,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
    lstm_epochs: int = DEFAULT_LSTM_EPOCHS,
    lstm_batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    lstm_seeds: tuple[int, ...] = tuple(DEFAULT_LSTM_SEEDS),
    tune_tabular: bool = False,
    tuning_splits: int = 3,
    tuning_iterations: int = 4,
) -> pd.DataFrame:
    """Return date-aligned fixed-holdout signals for all five model levels."""
    train = rows_for_dates(
        engineered,
        split_plan.train_dates,
        required_columns=[TARGET_DIRECTION_COLUMN],
    )
    test = rows_for_dates(
        engineered,
        split_plan.test_dates,
        required_columns=[TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN],
    )
    y_train = train[TARGET_DIRECTION_COLUMN].astype(int)
    majority_class = int(y_train.mode().iloc[0])
    majority = test[["Date", "Price", TARGET_RETURN_COLUMN]].copy()
    majority["Actual Direction"] = test[TARGET_DIRECTION_COLUMN].astype(int).to_numpy()
    majority["Predicted Probability"] = float(y_train.mean())
    majority["Predicted Direction"] = majority_class
    majority["Signal"] = majority_class
    majority["Probability Threshold"] = 0.50
    majority.insert(0, "Model", MAJORITY_MODEL_NAME)
    frames = [majority]

    expected_dates = pd.DatetimeIndex(pd.to_datetime(test["Date"]))
    for model_name in MODEL_LEVEL_NAMES[1:]:
        result = (
            lstm_result
            if model_name == LSTM_MODEL_NAME and lstm_result is not None
            else run_classification(
                engineered,
                model_name=model_name,
                train_size=split_plan.train_size,
                probability_threshold=probability_threshold,
                random_state=random_state,
                n_estimators=n_estimators,
                max_depth=max_depth,
                feature_names=feature_names,
                tune=tune_tabular and model_name in TABULAR_MODEL_NAMES,
                tuning_splits=tuning_splits,
                tuning_iterations=tuning_iterations,
                include_benchmark=False,
                lstm_lookback=lstm_lookback,
                lstm_units=lstm_units,
                lstm_dropout=lstm_dropout,
                lstm_epochs=lstm_epochs,
                lstm_batch_size=lstm_batch_size,
                lstm_seeds=lstm_seeds,
                split_plan=split_plan,
            )
        )
        if result is None:
            raise RuntimeError(f"Fixed-holdout evaluation failed for {model_name}.")
        frame = result.results.copy()
        if not pd.DatetimeIndex(pd.to_datetime(frame["Date"])).equals(expected_dates):
            raise ValueError(
                f"Fixed-holdout dates for {model_name} do not match the shared plan."
            )
        frame["Probability Threshold"] = probability_threshold
        frame.insert(0, "Model", model_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
