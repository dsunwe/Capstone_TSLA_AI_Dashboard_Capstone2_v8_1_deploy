from __future__ import annotations

import pandas as pd

from .config import (
    REQUIRED_COLUMNS,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)
from .temporal import (
    DEFAULT_MAX_LOOKBACK,
    DEFAULT_MINIMUM_TEST_ROWS,
    DEFAULT_MINIMUM_TRAIN_ROWS,
)


class SandboxInputError(ValueError):
    """Raised when exploratory input cannot support the dashboard pipeline."""


def minimum_clean_rows_for_sandbox(
    *,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    target_lookahead_rows: int = TARGET_PURGE_ROWS,
    purge_rows: int = TARGET_PURGE_ROWS,
    minimum_train_rows: int = DEFAULT_MINIMUM_TRAIN_ROWS,
    minimum_test_rows: int = DEFAULT_MINIMUM_TEST_ROWS,
) -> int:
    """Return the minimum clean rows required by the canonical temporal plan."""

    return (
        max_lookback
        - 1
        + target_lookahead_rows
        + minimum_train_rows
        + purge_rows
        + minimum_test_rows
    )


def validate_cleaned_sandbox_data(data: pd.DataFrame) -> None:
    """Reject empty, malformed, or clearly too-short cleaned OHLCV input."""

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise SandboxInputError(
            "No valid OHLCV rows remain after cleaning. Upload a non-empty CSV "
            "with valid Date, Open, High, Low, Close (or Adj Close), and Volume values."
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise SandboxInputError(
            "The cleaned dataset is missing required columns: " + ", ".join(missing)
        )

    if len(data) < DEFAULT_MAX_LOOKBACK:
        raise SandboxInputError(
            f"Only {len(data)} clean rows remain, but at least "
            f"{DEFAULT_MAX_LOOKBACK} chronological rows are required for the "
            "longest feature warm-up."
        )


def validate_engineered_sandbox_data(engineered: pd.DataFrame) -> None:
    """Ensure enough labelled targets remain for selection and temporal splitting."""

    target_columns = [TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]
    missing = [column for column in target_columns if column not in engineered.columns]
    if missing:
        raise SandboxInputError(
            "Feature engineering did not create the required executable targets: "
            + ", ".join(missing)
        )

    warmed = engineered.iloc[DEFAULT_MAX_LOOKBACK - 1 :]
    labelled = warmed.dropna(subset=target_columns)
    required_labelled = (
        DEFAULT_MINIMUM_TRAIN_ROWS
        + TARGET_PURGE_ROWS
        + DEFAULT_MINIMUM_TEST_ROWS
    )
    if len(labelled) < required_labelled:
        minimum_clean_rows = minimum_clean_rows_for_sandbox()
        raise SandboxInputError(
            f"Only {len(labelled)} eligible target rows remain after the "
            f"{DEFAULT_MAX_LOOKBACK}-session warm-up and five-session target look-ahead; "
            f"at least {required_labelled} are required. Provide at least "
            f"{minimum_clean_rows} clean chronological daily rows (more if prices "
            "contain invalid or missing values)."
        )

    if labelled[TARGET_DIRECTION_COLUMN].nunique() < 2:
        raise SandboxInputError(
            "The eligible target sample contains only one direction class. "
            "Feature selection and classification require both positive and "
            "non-positive future-mean closing-price outcomes."
        )


def sandbox_error_message(stage: str, error: Exception) -> str:
    """Convert expected parsing and analysis failures into concise UI guidance."""

    if isinstance(error, pd.errors.EmptyDataError):
        detail = (
            "The uploaded file is empty or contains no readable columns. "
            "Choose a non-empty CSV with an OHLCV header and data rows."
        )
    elif isinstance(error, pd.errors.ParserError):
        detail = (
            "The uploaded file could not be parsed as CSV. Check its delimiter, "
            "quoting, and row structure."
        )
    elif isinstance(error, SandboxInputError):
        detail = str(error)
    else:
        raw_detail = str(error).strip()
        if "Missing required columns" in raw_detail:
            detail = (
                f"{raw_detail}. Expected Date, Open, High, Low, Close "
                "(or Adj Close), and Volume columns."
            )
        elif "Not enough common eligible dates" in raw_detail:
            detail = (
                f"{raw_detail} Upload a longer daily history with valid executable "
                "open prices."
            )
        elif "requires both target classes" in raw_detail:
            detail = (
                f"{raw_detail} Upload a longer or more varied price history."
            )
        else:
            detail = raw_detail or "An unexpected input-processing error occurred."

    return f"Sandbox {stage} could not be completed. {detail}"
