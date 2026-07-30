from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from .config import (
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
)

DEFAULT_MAX_LOOKBACK = 50
DEFAULT_MINIMUM_TRAIN_ROWS = 30
DEFAULT_MINIMUM_TEST_ROWS = 10
DEFAULT_TARGET_COLUMNS = (TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN)


def _normalise_dates(values: Sequence[object] | pd.Series) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.Series(values), errors="coerce")
    if dates.isna().any():
        raise ValueError("Temporal data contains an invalid or missing Date value.")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return pd.DatetimeIndex(dates.dt.normalize())


def _as_timestamp_tuple(values: Sequence[object] | pd.DatetimeIndex) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(value) for value in _normalise_dates(values))


@dataclass(frozen=True)
class TemporalSplitPlan:
    """Immutable date contract shared by selection, models, and comparisons."""

    eligible_dates: tuple[pd.Timestamp, ...]
    train_dates: tuple[pd.Timestamp, ...]
    gap_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    train_size: float
    gap: int
    max_lookback: int
    target_columns: tuple[str, ...]
    plan_id: str
    plan_hash: str

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end_date(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def gap_start_date(self) -> pd.Timestamp:
        return self.gap_dates[0]

    @property
    def gap_end_date(self) -> pd.Timestamp:
        return self.gap_dates[-1]

    @property
    def test_start_date(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end_date(self) -> pd.Timestamp:
        return self.test_dates[-1]

    def boundary_metadata(self) -> dict[str, object]:
        return {
            "Plan ID": self.plan_id,
            "Plan Hash": self.plan_hash,
            "Train Start": self.train_start_date,
            "Train End": self.train_end_date,
            "Purge Start": self.gap_start_date,
            "Purge End": self.gap_end_date,
            "Test Start": self.test_start_date,
            "Test End": self.test_end_date,
            "Train Rows": len(self.train_dates),
            "Purge Rows": len(self.gap_dates),
            "Test Rows": len(self.test_dates),
        }

    def manifest(self) -> pd.DataFrame:
        partitions = (
            ("Train", self.train_dates),
            ("Purge", self.gap_dates),
            ("Test", self.test_dates),
        )
        rows = [
            {
                "Plan ID": self.plan_id,
                "Plan Hash": self.plan_hash,
                "Partition": partition,
                "Date": date,
                "Train Fraction": self.train_size,
                "Gap": self.gap,
                "Maximum Prespecified Lookback": self.max_lookback,
            }
            for partition, dates in partitions
            for date in dates
        ]
        return pd.DataFrame(rows)


def _eligible_dates_for_frame(
    frame: pd.DataFrame,
    *,
    max_lookback: int,
    target_columns: Sequence[str],
) -> pd.DatetimeIndex:
    required = {"Date", *target_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Temporal split data is missing required columns: {missing}")
    if max_lookback < 1:
        raise ValueError("max_lookback must be at least 1.")

    data = frame[["Date", *target_columns]].copy()
    data["Date"] = _normalise_dates(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    if data["Date"].duplicated().any():
        raise ValueError("Temporal split data must contain unique trading dates.")
    if len(data) < max_lookback:
        raise ValueError(
            f"At least {max_lookback} chronological rows are required for feature warm-up."
        )

    # Eligibility is defined by the prespecified maximum feature lookback and
    # known target availability. It deliberately does not depend on whether a
    # particular candidate feature happens to be missing.
    warmed = data.iloc[max_lookback - 1 :].copy()
    target_known = warmed[list(target_columns)].notna().all(axis=1)
    return pd.DatetimeIndex(warmed.loc[target_known, "Date"])


def _plan_hash_payload(
    eligible_dates: Sequence[pd.Timestamp],
    *,
    train_dates: Sequence[pd.Timestamp],
    gap_dates: Sequence[pd.Timestamp],
    test_dates: Sequence[pd.Timestamp],
    train_size: float,
    gap: int,
    max_lookback: int,
    target_columns: Sequence[str],
    minimum_train_rows: int,
    minimum_test_rows: int,
) -> str:
    payload = {
        "schema": "temporal-split-plan-v2-executable-open-to-open",
        "eligible_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in eligible_dates],
        "train_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in train_dates],
        "gap_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in gap_dates],
        "test_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in test_dates],
        "train_size": float(train_size),
        "gap": int(gap),
        "max_lookback": int(max_lookback),
        "target_columns": list(target_columns),
        "minimum_train_rows": int(minimum_train_rows),
        "minimum_test_rows": int(minimum_test_rows),
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def build_temporal_split_plan(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    train_size: float = 0.80,
    gap: int = TARGET_PURGE_ROWS,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    minimum_train_rows: int = DEFAULT_MINIMUM_TRAIN_ROWS,
    minimum_test_rows: int = DEFAULT_MINIMUM_TEST_ROWS,
) -> TemporalSplitPlan:
    """Create one canonical chronological split from common eligible dates.

    The eligible universe starts only after the maximum prespecified feature
    lookback and requires known targets. Feature values and their missingness
    never influence the split. When several assets are supplied, their eligible
    date intersection is used so every asset receives the same partition.
    """

    if not 0 < train_size < 1:
        raise ValueError("train_size must be strictly between 0 and 1.")
    if gap < TARGET_PURGE_ROWS:
        raise ValueError(
            f"gap must be at least {TARGET_PURGE_ROWS} because the "
            "classification label uses Close[t+1:t+5]."
        )
    if minimum_train_rows < 1 or minimum_test_rows < 1:
        raise ValueError("minimum train and test rows must be positive.")

    frames = data if isinstance(data, Mapping) else {"primary": data}
    if not frames:
        raise ValueError("At least one frame is required to build a temporal split.")

    eligible_by_frame = [
        _eligible_dates_for_frame(
            frame,
            max_lookback=max_lookback,
            target_columns=target_columns,
        )
        for frame in frames.values()
    ]
    common_dates = set(eligible_by_frame[0])
    for dates in eligible_by_frame[1:]:
        common_dates.intersection_update(dates)
    eligible_dates = tuple(sorted(pd.Timestamp(date) for date in common_dates))

    minimum_total = minimum_train_rows + gap + minimum_test_rows
    if len(eligible_dates) < minimum_total:
        raise ValueError(
            "Not enough common eligible dates for the requested chronological split: "
            f"{len(eligible_dates)} available, {minimum_total} required."
        )

    split_index = int(len(eligible_dates) * train_size)
    split_index = min(
        max(split_index, minimum_train_rows + gap),
        len(eligible_dates) - minimum_test_rows,
    )
    train_dates = eligible_dates[: split_index - gap]
    gap_dates = eligible_dates[split_index - gap : split_index]
    test_dates = eligible_dates[split_index:]

    plan_hash = _plan_hash_payload(
        eligible_dates,
        train_dates=train_dates,
        gap_dates=gap_dates,
        test_dates=test_dates,
        train_size=train_size,
        gap=gap,
        max_lookback=max_lookback,
        target_columns=target_columns,
        minimum_train_rows=minimum_train_rows,
        minimum_test_rows=minimum_test_rows,
    )
    return TemporalSplitPlan(
        eligible_dates=eligible_dates,
        train_dates=train_dates,
        gap_dates=gap_dates,
        test_dates=test_dates,
        train_size=float(train_size),
        gap=int(gap),
        max_lookback=int(max_lookback),
        target_columns=tuple(target_columns),
        plan_id=f"temporal-v2-{plan_hash[:16]}",
        plan_hash=plan_hash,
    )


def rows_for_dates(
    frame: pd.DataFrame,
    dates: Sequence[object],
    *,
    required_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return exactly the requested rows in requested chronological order."""

    missing_columns = sorted({"Date", *required_columns}.difference(frame.columns))
    if missing_columns:
        raise KeyError(f"Frame is missing required columns: {missing_columns}")

    requested = _as_timestamp_tuple(dates)
    if len(set(requested)) != len(requested):
        raise ValueError("Requested temporal dates must be unique.")

    data = frame.copy()
    data["Date"] = _normalise_dates(data["Date"])
    if data["Date"].duplicated().any():
        raise ValueError("Frame must contain unique dates before temporal alignment.")
    available = set(pd.Timestamp(date) for date in data["Date"])
    missing_dates = [date for date in requested if date not in available]
    if missing_dates:
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:5])
        raise ValueError(f"Frame does not cover all dates in the split plan: {preview}")

    indexed = data.set_index("Date", drop=False)
    aligned = indexed.loc[list(requested)].copy().reset_index(drop=True)
    if tuple(aligned["Date"]) != requested:
        raise AssertionError("Temporal alignment did not preserve the exact requested date order.")
    return aligned


def split_frame_by_plan(
    frame: pd.DataFrame,
    plan: TemporalSplitPlan,
    *,
    required_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align one frame to a plan without dropping rows for feature missingness."""

    train = rows_for_dates(frame, plan.train_dates, required_columns=required_columns)
    gap_frame = rows_for_dates(frame, plan.gap_dates, required_columns=required_columns)
    test = rows_for_dates(frame, plan.test_dates, required_columns=required_columns)
    return train, gap_frame, test
