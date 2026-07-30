import numpy as np
import pandas as pd

from src.config import (
    CANDIDATE_FEATURES,
    SELECTED_FEATURES,
    TARGET_DATE_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_HORIZON_SESSIONS,
    TARGET_RETURN_COLUMN,
    TARGET_TRAILING_SESSIONS,
    TARGET_VALUATION_DATE_COLUMN,
)


def test_selected_features_are_created(engineered_data):
    assert all(feature in engineered_data.columns for feature in SELECTED_FEATURES)


def test_targets_are_not_selected_features():
    assert TARGET_RETURN_COLUMN not in SELECTED_FEATURES
    assert TARGET_DIRECTION_COLUMN not in SELECTED_FEATURES


def test_primary_target_is_the_five_session_forward_vs_trailing_mean_close_class(
    engineered_data,
):
    chronological = engineered_data.sort_values("Date").reset_index(drop=True)
    trailing_mean = chronological["Price"].rolling(
        TARGET_TRAILING_SESSIONS
    ).mean()
    future_mean = (
        chronological["Price"]
        .shift(-1)
        .rolling(TARGET_HORIZON_SESSIONS)
        .mean()
        .shift(-(TARGET_HORIZON_SESSIONS - 1))
    )
    expected_return = future_mean / trailing_mean - 1.0
    expected_direction = np.where(
        expected_return.isna(),
        np.nan,
        (expected_return > 0).astype(int),
    )

    pd.testing.assert_series_equal(
        chronological[TARGET_RETURN_COLUMN],
        expected_return.rename(TARGET_RETURN_COLUMN),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        chronological[TARGET_DIRECTION_COLUMN].to_numpy(),
        expected_direction,
    )
    assert chronological[TARGET_RETURN_COLUMN].tail(5).isna().all()
    assert chronological[TARGET_DIRECTION_COLUMN].tail(5).isna().all()


def test_target_and_decision_dates_use_t_plus_1_t_plus_5_and_t_plus_6(
    engineered_data,
):
    chronological = engineered_data.sort_values("Date").reset_index(drop=True)

    pd.testing.assert_series_equal(
        pd.to_datetime(chronological[TARGET_EXECUTION_DATE_COLUMN]),
        pd.to_datetime(chronological["Date"].shift(-1)).rename(
            TARGET_EXECUTION_DATE_COLUMN
        ),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        pd.to_datetime(chronological[TARGET_DATE_COLUMN]),
        pd.to_datetime(chronological["Date"].shift(-5)).rename(
            TARGET_DATE_COLUMN
        ),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        pd.to_datetime(chronological[TARGET_VALUATION_DATE_COLUMN]),
        pd.to_datetime(chronological["Date"].shift(-6)).rename(
            TARGET_VALUATION_DATE_COLUMN
        ),
        check_exact=True,
    )
    valid = chronological.dropna(
        subset=[
            TARGET_EXECUTION_DATE_COLUMN,
            TARGET_DATE_COLUMN,
            TARGET_VALUATION_DATE_COLUMN,
        ]
    )
    assert (valid["Date"] < valid[TARGET_EXECUTION_DATE_COLUMN]).all()
    assert (
        valid[TARGET_EXECUTION_DATE_COLUMN]
        < valid[TARGET_DATE_COLUMN]
    ).all()
    assert (
        valid[TARGET_DATE_COLUMN]
        < valid[TARGET_VALUATION_DATE_COLUMN]
    ).all()


def test_examiner_requested_candidate_indicators_are_created(engineered_data):
    required = {
        "RSI_14",
        "MACD",
        "MACD_Histogram",
        "Bollinger_Band_Width",
        "ATR_14",
        "Market_Return",
    }
    assert required.issubset(engineered_data.columns)
    assert all(feature in engineered_data.columns for feature in CANDIDATE_FEATURES)
