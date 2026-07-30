from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    REQUIRED_COLUMNS,
    TARGET_DIRECTION_COLUMN,
    TARGET_RETURN_COLUMN,
)
from src.sandbox import (
    SandboxInputError,
    minimum_clean_rows_for_sandbox,
    sandbox_error_message,
    validate_cleaned_sandbox_data,
    validate_engineered_sandbox_data,
)


def cleaned_frame(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=rows)
    values = np.arange(1, rows + 1, dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": values,
            "High": values + 1,
            "Low": values - 0.5,
            "Close": values + 0.25,
            "Volume": values * 1_000,
        }
    )[REQUIRED_COLUMNS]


def engineered_target_frame(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            TARGET_RETURN_COLUMN: np.resize([0.01, -0.01], rows),
            TARGET_DIRECTION_COLUMN: np.resize([1.0, 0.0], rows),
        }
    )
    frame.loc[rows - 5 :, [TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]] = np.nan
    return frame


def test_minimum_clean_rows_matches_canonical_plan():
    assert minimum_clean_rows_for_sandbox() == 99


def test_cleaned_sandbox_validation_rejects_empty_and_short_data():
    with pytest.raises(SandboxInputError, match="No valid OHLCV rows"):
        validate_cleaned_sandbox_data(pd.DataFrame(columns=REQUIRED_COLUMNS))

    with pytest.raises(SandboxInputError, match="at least 50 chronological rows"):
        validate_cleaned_sandbox_data(cleaned_frame(49))


def test_engineered_sandbox_validation_rejects_insufficient_targets():
    with pytest.raises(SandboxInputError, match="Only 44 eligible target rows"):
        validate_engineered_sandbox_data(engineered_target_frame(98))


def test_engineered_sandbox_validation_accepts_minimum_varied_sample():
    validate_engineered_sandbox_data(engineered_target_frame(99))


def test_engineered_sandbox_validation_rejects_one_class():
    frame = engineered_target_frame(99)
    frame[TARGET_DIRECTION_COLUMN] = 1.0
    frame.loc[94:, TARGET_DIRECTION_COLUMN] = np.nan

    with pytest.raises(SandboxInputError, match="only one direction class"):
        validate_engineered_sandbox_data(frame)


def test_sandbox_error_messages_are_actionable():
    empty_message = sandbox_error_message(
        "CSV loading",
        pd.errors.EmptyDataError("No columns to parse from file"),
    )
    schema_message = sandbox_error_message(
        "OHLCV cleaning",
        ValueError("Missing required columns: Volume"),
    )

    assert "empty" in empty_message.lower()
    assert "non-empty CSV" in empty_message
    assert "Expected Date, Open, High, Low, Close" in schema_message
