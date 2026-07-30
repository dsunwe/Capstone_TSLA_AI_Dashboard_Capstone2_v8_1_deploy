import numpy as np
import pandas as pd

from src.config import (
    NEWS_ATTENTION_FEATURES,
    NEWS_SENTIMENT_FEATURES,
    SENTIMENT_FEATURE_SETS,
    SENTIMENT_PROXY_FEATURES,
    SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
    SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
    SUPPLEMENTARY_TARGET_RETURN_COLUMN,
)
from src.sentiment_study import (
    TARGET_LARGE_MOVE,
    _fold_labels,
    run_dual_target_benchmark,
)


def _study_frame(rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=rows)
    all_features = list(
        dict.fromkeys(
            feature
            for features in SENTIMENT_FEATURE_SETS.values()
            for feature in features
        )
    )
    frame = pd.DataFrame({"Date": dates})
    for feature in all_features:
        frame[feature] = rng.normal(size=rows)
    frame["No_News"] = rng.integers(0, 2, size=rows)
    frame["Unique_Story_Count"] = rng.poisson(3, size=rows)
    frame["Log_News_Count"] = np.log1p(frame["Unique_Story_Count"])
    returns = rng.normal(0, 0.025, size=rows)
    frame[SUPPLEMENTARY_TARGET_RETURN_COLUMN] = returns
    frame[SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN] = np.abs(returns)
    frame[SUPPLEMENTARY_TARGET_DIRECTION_COLUMN] = (returns > 0).astype(int)
    frame["Price"] = 100 * (1 + pd.Series(returns).fillna(0)).cumprod()
    return frame


def test_f0_f5_keep_attention_and_sentiment_separate():
    assert not set(NEWS_ATTENTION_FEATURES) & set(NEWS_SENTIMENT_FEATURES)
    assert not set(SENTIMENT_FEATURE_SETS["F0"]) & set(NEWS_ATTENTION_FEATURES)
    assert not set(SENTIMENT_FEATURE_SETS["F0"]) & set(NEWS_SENTIMENT_FEATURES)
    assert set(SENTIMENT_PROXY_FEATURES).issubset(SENTIMENT_FEATURE_SETS["F1"])
    assert set(NEWS_ATTENTION_FEATURES).issubset(SENTIMENT_FEATURE_SETS["F2"])
    assert set(NEWS_SENTIMENT_FEATURES).issubset(SENTIMENT_FEATURE_SETS["F3"])
    assert set(NEWS_ATTENTION_FEATURES + NEWS_SENTIMENT_FEATURES).issubset(
        SENTIMENT_FEATURE_SETS["F4"]
    )
    assert set(
        SENTIMENT_PROXY_FEATURES + NEWS_ATTENTION_FEATURES + NEWS_SENTIMENT_FEATURES
    ).issubset(SENTIMENT_FEATURE_SETS["F5"])


def test_large_move_threshold_is_fitted_from_training_returns_only():
    frame = _study_frame(120)
    train_indices = np.arange(0, 80)
    test_indices = np.arange(81, 120)
    _, _, threshold_before = _fold_labels(
        frame,
        train_indices,
        test_indices,
        TARGET_LARGE_MOVE,
        large_move_quantile=0.80,
    )
    changed = frame.copy()
    changed.loc[test_indices, SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN] = 1000
    _, _, threshold_after = _fold_labels(
        changed,
        train_indices,
        test_indices,
        TARGET_LARGE_MOVE,
        large_move_quantile=0.80,
    )
    assert threshold_before == threshold_after


def test_f0_f5_oof_rows_use_identical_test_dates():
    frames = {asset: _study_frame() for asset in ["TSLA", "AAPL", "NVDA"]}
    result = run_dual_target_benchmark(
        frames,
        feature_sets=("F0", "F1", "F5"),
        models=("Logistic Regression (Baseline)",),
        targets=("direction", "large_move"),
        n_splits=3,
        include_lstm=False,
    )
    learned = result.oof_predictions.loc[
        result.oof_predictions["Model"] == "Logistic Regression (Baseline)"
    ]
    for (_, _), group in learned.groupby(["Asset", "Target"]):
        dates = {
            feature_set: tuple(values["Date"])
            for feature_set, values in group.groupby("Feature Set")
        }
        assert dates["F0"] == dates["F1"] == dates["F5"]
    assert result.fold_thresholds.loc[
        result.fold_thresholds["Target"] == "large_move", "Large-Move Threshold"
    ].notna().all()
