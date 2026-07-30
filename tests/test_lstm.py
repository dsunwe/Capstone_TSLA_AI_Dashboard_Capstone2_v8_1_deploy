import numpy as np
import pandas as pd
import pytest

from src.config import (
    LSTM_MODEL_NAME,
    MODEL_LEVEL_NAMES,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)
from src.lstm import build_sequences, prepare_sequence_fold
from src.ml import (
    run_classification,
    walk_forward_evaluate,
    walk_forward_model_suite,
    walk_forward_oof_predictions,
)


def test_lstm_oof_uses_three_seed_probability_ensemble(engineered_data):
    pytest.importorskip("tensorflow")
    features = [
        "Lag_Return_1",
        "Momentum_5",
        "Volatility_10",
        "Volume_Ratio",
        "SP500_Return",
    ]
    metrics, predictions, _, seed_results, seed_predictions = (
        walk_forward_evaluate(
            engineered_data,
            model_name=LSTM_MODEL_NAME,
            n_splits=2,
            feature_names=features,
            lstm_epochs=1,
            lstm_seeds=(42, 43, 44),
            return_diagnostics=True,
        )
    )
    assert set(seed_results["Seed"]) == {42, 43, 44}
    assert len(seed_results) == 2 * 3
    assert metrics["Seed Count"].eq(3).all()
    assert metrics["Ensemble Rule"].eq(
        "Arithmetic mean of seed probabilities"
    ).all()
    seed_mean = (
        seed_predictions.groupby("Date", as_index=False)["Predicted Probability"]
        .mean()
        .rename(columns={"Predicted Probability": "Seed Mean"})
    )
    comparison = predictions.merge(seed_mean, on="Date", how="inner")
    assert len(comparison) == len(predictions)
    assert np.allclose(
        comparison["Predicted Probability"],
        comparison["Seed Mean"],
        rtol=0.0,
        atol=1e-12,
    )


def test_sequence_anchor_matches_current_row_executable_label():
    scaled = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.arange(12) % 2
    x_data, y_data, anchors = build_sequences(
        scaled,
        labels,
        [3, 4, 8],
        lookback=4,
    )
    assert x_data.shape == (3, 4, 2)
    assert anchors.tolist() == [3, 4, 8]
    assert y_data.tolist() == labels[anchors].tolist()
    assert np.array_equal(x_data[0], scaled[0:4])


def test_sequence_scaler_is_fitted_on_unique_training_rows_only():
    rows = 80
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Price": np.linspace(100, 140, rows),
            "Feature_A": np.r_[np.arange(60), np.full(20, 1_000_000)],
            "Feature_B": np.arange(rows) % 7,
            TARGET_RETURN_COLUMN: np.linspace(-0.01, 0.01, rows),
            TARGET_DIRECTION_COLUMN: np.arange(rows) % 2,
        }
    )
    train_indices = np.arange(0, 60)
    test_indices = np.arange(60 + TARGET_PURGE_ROWS, 80)
    assert test_indices[0] - train_indices[-1] - 1 == TARGET_PURGE_ROWS
    fold = prepare_sequence_fold(
        frame,
        ["Feature_A", "Feature_B"],
        train_indices,
        test_indices,
        lookback=5,
    )
    assert fold.preprocessor.scaler.mean_[0] == pytest.approx(np.arange(60).mean())
    assert fold.x_test[:, :, 0].max() > 1_000


def test_lstm_holdout_returns_valid_probabilities(engineered_data):
    pytest.importorskip("tensorflow")
    features = [
        "Lag_Return_1",
        "Momentum_5",
        "Volatility_10",
        "Volume_Ratio",
        "SP500_Return",
    ]
    result = run_classification(
        engineered_data,
        model_name="LSTM Classifier",
        feature_names=features,
        include_benchmark=False,
        lstm_epochs=1,
        lstm_seeds=(42,),
    )
    assert result is not None
    probabilities = result.results["Predicted Probability"]
    assert probabilities.between(0, 1).all()
    assert result.lookback == 20
    assert result.parameter_count > 0
    assert result.train_end_date < result.purge_start_date
    assert result.purge_start_date < result.purge_end_date
    assert result.purge_end_date < result.test_start_date
    assert TARGET_RETURN_COLUMN in result.results


def test_lstm_walk_forward_outputs_are_unique_and_fold_labelled(engineered_data):
    pytest.importorskip("tensorflow")
    features = [
        "Lag_Return_1",
        "Momentum_5",
        "Volatility_10",
        "Volume_Ratio",
        "SP500_Return",
    ]
    result = walk_forward_oof_predictions(
        engineered_data,
        model_name="LSTM Classifier",
        n_splits=3,
        feature_names=features,
        lstm_epochs=1,
    )
    assert not result.empty
    assert result["Date"].is_unique
    assert result["Fold"].nunique() == 3
    assert result["Predicted Probability"].between(0, 1).all()
    train_end = pd.to_datetime(result["Train End"])
    purge_start = pd.to_datetime(result["Purge Start"])
    purge_end = pd.to_datetime(result["Purge End"])
    test_date = pd.to_datetime(result["Date"])
    assert (train_end < purge_start).all()
    assert (purge_start < purge_end).all()
    assert (purge_end < test_date).all()
    assert TARGET_RETURN_COLUMN in result


def test_five_model_level_oof_suite_uses_identical_dates_folds_and_boundaries(
    engineered_data,
):
    pytest.importorskip("xgboost")
    pytest.importorskip("tensorflow")
    features = [
        "Lag_Return_1",
        "Momentum_5",
        "Volatility_10",
        "Volume_Ratio",
        "SP500_Return",
    ]
    metrics, predictions = walk_forward_model_suite(
        engineered_data,
        n_splits=3,
        probability_threshold=0.50,
        n_estimators=20,
        max_depth=3,
        feature_names=features,
        lstm_epochs=1,
        foldwise_feature_selection=True,
        candidate_features=features,
        top_k_features=len(features),
    )

    assert list(MODEL_LEVEL_NAMES) == [
        "Majority Baseline",
        "Logistic Regression (Baseline)",
        "Random Forest Classifier",
        "XGBoost Classifier",
        LSTM_MODEL_NAME,
    ]
    assert set(metrics["Model"]) == set(MODEL_LEVEL_NAMES)
    assert set(predictions["Model"]) == set(MODEL_LEVEL_NAMES)
    assert metrics.groupby("Model")["Fold"].nunique().eq(3).all()
    assert predictions.groupby("Model")["Fold"].nunique().eq(3).all()
    assert not predictions.duplicated(["Model", "Date"]).any()
    common_metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Training Time (s)",
        "Inference Time (s)",
    ]
    assert set(common_metric_columns).issubset(metrics.columns)
    assert np.isfinite(metrics[common_metric_columns].to_numpy()).all()

    alignment_columns = [
        "Date",
        "Fold",
        "Train End",
        "Purge Start",
        "Purge End",
    ]
    reference = (
        predictions.loc[
            predictions["Model"] == MODEL_LEVEL_NAMES[0],
            alignment_columns,
        ]
        .reset_index(drop=True)
        .copy()
    )
    for model_name in MODEL_LEVEL_NAMES[1:]:
        candidate = predictions.loc[
            predictions["Model"] == model_name,
            alignment_columns,
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(candidate, reference)

    train_end = pd.to_datetime(predictions["Train End"])
    purge_start = pd.to_datetime(predictions["Purge Start"])
    purge_end = pd.to_datetime(predictions["Purge End"])
    test_date = pd.to_datetime(predictions["Date"])
    assert (train_end < purge_start).all()
    assert (purge_start < purge_end).all()
    assert (purge_end < test_date).all()

    predictive = predictions[
        predictions["Model"].isin(MODEL_LEVEL_NAMES[1:])
    ]
    feature_alignment = [
        "Date",
        "Fold",
        "Selected Feature Count",
        "Selected Features",
        "Feature Set Hash",
    ]
    reference_features = (
        predictive.loc[
            predictive["Model"] == MODEL_LEVEL_NAMES[1],
            feature_alignment,
        ]
        .reset_index(drop=True)
        .copy()
    )
    for model_name in MODEL_LEVEL_NAMES[2:]:
        candidate = predictive.loc[
            predictive["Model"] == model_name,
            feature_alignment,
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(candidate, reference_features)

    assert TARGET_RETURN_COLUMN in predictions
    assert np.array_equal(
        predictions["Actual Direction"].to_numpy(),
        (predictions[TARGET_RETURN_COLUMN].to_numpy() > 0).astype(int),
    )
