import numpy as np
import pandas as pd
import pytest

from src.backtesting import ML_STRATEGY_SPEC, RULE_STRATEGY_SPECS
from src.config import MAJORITY_MODEL_NAME, TARGET_RETURN_COLUMN
from src.main_study_inference import (
    build_classification_uncertainty,
    build_return_separation_uncertainty,
    build_strategy_benchmark_uncertainty,
)


def _prediction_fixture() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=80, freq="B")
    actual = np.arange(80) % 2
    returns = np.where(actual == 1, 0.01, -0.01)
    folds = np.repeat(np.arange(1, 5), 20)
    frames = []
    for model, predicted in [
        (MAJORITY_MODEL_NAME, np.tile([0, 0, 1, 1], 20)),
        ("Perfect Model", actual),
    ]:
        frames.append(
            pd.DataFrame(
                {
                    "Model": model,
                    "Date": dates,
                    "Fold": folds,
                    "Actual Direction": actual,
                    "Predicted Direction": predicted,
                    "Signal": predicted,
                    "Predicted Probability": np.where(predicted == 1, 0.8, 0.2),
                    TARGET_RETURN_COLUMN: returns,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _execution_fixture(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model, group in predictions.groupby("Model", sort=False):
        frame = pd.DataFrame(
            {
                "Signal Model": model,
                "Signal Date": pd.to_datetime(group["Date"]),
                ML_STRATEGY_SPEC[1]: np.where(group["Signal"].to_numpy() == 1, 0.01, 0.0),
            }
        )
        for index, (_, return_column, _) in enumerate(RULE_STRATEGY_SPECS):
            frame[return_column] = 0.0001 * index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_main_study_uncertainty_is_deterministic_and_paired():
    predictions = _prediction_fixture()
    first = build_classification_uncertainty(
        predictions,
        n_bootstrap=100,
        block_length=5,
        random_state=9,
    )
    second = build_classification_uncertainty(
        predictions,
        n_bootstrap=100,
        block_length=5,
        random_state=9,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    perfect_accuracy = first.loc[
        first["Model"].eq("Perfect Model")
        & first["Analysis"].eq("Absolute OOF metric")
        & first["Metric"].eq("Accuracy")
    ].iloc[0]
    assert perfect_accuracy["Point Estimate"] == pytest.approx(1.0)
    assert perfect_accuracy["CI Lower"] == pytest.approx(1.0)

    separation = build_return_separation_uncertainty(
        predictions,
        n_bootstrap=100,
        block_length=5,
        random_state=9,
    )
    perfect_separation = separation.loc[
        separation["Model"].eq("Perfect Model")
    ].iloc[0]
    assert perfect_separation["Point Estimate"] == pytest.approx(0.02)
    assert perfect_separation["CI Lower"] > 0


def test_strategy_uncertainty_covers_every_model_and_benchmark():
    predictions = _prediction_fixture()
    execution = _execution_fixture(predictions)
    result = build_strategy_benchmark_uncertainty(
        predictions,
        execution,
        n_bootstrap=80,
        block_length=4,
        random_state=3,
    )
    assert len(result) == 2 * len(RULE_STRATEGY_SPECS)
    assert result.groupby("Model")["Benchmark"].nunique().eq(8).all()
    assert result["Valid Replicates"].eq(80).all()


def test_main_study_inference_rejects_misaligned_models_and_benchmarks():
    predictions = _prediction_fixture()
    bad_predictions = predictions.copy()
    bad_predictions.loc[
        bad_predictions["Model"].eq("Perfect Model")
        & bad_predictions["Date"].eq(bad_predictions["Date"].max()),
        "Fold",
    ] = 99
    with pytest.raises(ValueError, match="folds are not aligned"):
        build_classification_uncertainty(
            bad_predictions,
            n_bootstrap=20,
            block_length=4,
        )

    execution = _execution_fixture(predictions)
    execution.loc[
        execution["Signal Model"].eq("Perfect Model"),
        RULE_STRATEGY_SPECS[0][1],
    ] += 0.001
    with pytest.raises(ValueError, match="differs across model copies"):
        build_strategy_benchmark_uncertainty(
            predictions,
            execution,
            n_bootstrap=20,
            block_length=4,
        )
