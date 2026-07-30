from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .config import CLEANED_DATA_PATH, DATASET_PATHS, REQUIRED_COLUMNS

KNOWN_SPLIT_EVENTS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "TSLA": (
        ("2020-08-31", "Five-for-one split", 5),
        ("2022-08-25", "Three-for-one split", 3),
    ),
    "AAPL": (("2020-08-31", "Four-for-one split", 4),),
    "NVDA": (
        ("2021-07-20", "Four-for-one split", 4),
        ("2024-06-10", "Ten-for-one split", 10),
    ),
}


def generate_demo_data(rows: int = 750, seed: int = 42) -> pd.DataFrame:
    """Create deterministic TSLA-like OHLCV data for first-run testing only."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="B")
    log_returns = rng.normal(0.0007, 0.034, size=rows)
    close = 220 * np.exp(np.cumsum(log_returns))
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.008, size=rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.002, 0.03, size=rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.002, 0.03, size=rows))
    volume = rng.lognormal(mean=18.1, sigma=0.25, size=rows).astype(int)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def normalize_column_name(column: str) -> str:
    value = str(column).strip().lower()
    for old, new in [("/", "_"), (" ", "_"), ("-", "_"), (".", "_")]:
        value = value.replace(old, new)
    while "__" in value:
        value = value.replace("__", "_")
    return value


# Backward-compatible alias for callers using British spelling.
normalise_column_name = normalize_column_name


def clean_number_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["", "nan", "NaN", "None", "null"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _read_csv(source: str | Path | IO[str] | IO[bytes] | BytesIO | StringIO) -> pd.DataFrame:
    return pd.read_csv(source)


def preprocess_stock_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardise and clean OHLCV data while preserving an audit profile."""
    raw_preview = df.head(10).copy()
    original_columns = list(df.columns)
    aliases = {
        "date": "Date",
        "datetime": "Date",
        "time": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "close_last": "Close",
        "adj_close": "Adj Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }

    rename_map: dict[str, str] = {}
    close_source_field: str | None = None
    adjusted_source_field: str | None = None
    for original in df.columns:
        normalised = normalize_column_name(original)
        if normalised in aliases:
            rename_map[original] = aliases[normalised]
            if aliases[normalised] == "Adj Close":
                adjusted_source_field = str(original)
            elif aliases[normalised] == "Close" and close_source_field is None:
                close_source_field = str(original)

    working = df.rename(columns=rename_map).copy()
    if "Close" not in working.columns and "Adj Close" in working.columns:
        working["Close"] = working["Adj Close"]
        source_price_field = adjusted_source_field or "Adj Close"
        uses_explicit_adjusted_close = True
    else:
        source_price_field = close_source_field or "Unknown"
        uses_explicit_adjusted_close = False

    missing_required = [column for column in REQUIRED_COLUMNS if column not in working.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

    working = working[REQUIRED_COLUMNS].copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        working[column] = clean_number_column(working[column])

    rows_before = len(working)
    missing_before = working.isna().sum().to_dict()
    duplicate_dates_before = int(working.loc[working["Date"].notna(), "Date"].duplicated().sum())

    cleaned = (
        working.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("Date", kind="mergesort")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    cleaned["Price"] = cleaned["Close"]
    cleaned["Daily_Return"] = cleaned["Price"].pct_change()
    cleaned["Cumulative_Return"] = (1 + cleaned["Daily_Return"].fillna(0)).cumprod() - 1

    ordered = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Price",
        "Volume",
        "Daily_Return",
        "Cumulative_Return",
    ]
    cleaned = cleaned[ordered]

    profile: dict[str, Any] = {
        "raw_preview": raw_preview,
        "cleaned_preview": cleaned.head(10).copy(),
        "raw_feature_count": len(original_columns),
        "raw_feature_names": original_columns,
        "source_price_field": source_price_field,
        "uses_explicit_adjusted_close": uses_explicit_adjusted_close,
        "rows_before_preprocessing": rows_before,
        "rows_after_preprocessing": len(cleaned),
        "rows_removed": rows_before - len(cleaned),
        "duplicate_dates_before_removal": duplicate_dates_before,
        "missing_before_required_drop": missing_before,
        "missing_after_preprocessing": cleaned.isna().sum().to_dict(),
    }
    return cleaned, profile


def preprocessing_audit_table(profile: dict[str, Any]) -> pd.DataFrame:
    """Build examiner-facing raw-to-clean evidence from one preprocessing run."""
    missing = profile.get("missing_before_required_drop", {})
    missing_total = int(sum(int(value) for value in missing.values()))
    rows_before = int(profile.get("rows_before_preprocessing", 0))
    rows_after = int(profile.get("rows_after_preprocessing", 0))
    duplicates = int(profile.get("duplicate_dates_before_removal", 0))
    return pd.DataFrame(
        [
            {
                "Step": "Schema mapping",
                "Before": ", ".join(map(str, profile.get("raw_feature_names", []))),
                "After": ", ".join(REQUIRED_COLUMNS),
                "Reason": "Apply one OHLCV schema to every dataset.",
            },
            {
                "Step": "Numeric conversion",
                "Before": "Currency symbols, commas, or text values may be present",
                "After": "Open, High, Low, Close, and Volume are numeric",
                "Reason": "Enable feature calculation and model input validation.",
            },
            {
                "Step": "Date conversion and ordering",
                "Before": "Date strings may be newest-first",
                "After": "Datetime values in ascending chronological order",
                "Reason": "Prevent random ordering and time-series leakage.",
            },
            {
                "Step": "Missing required rows",
                "Before": f"{missing_total} required-field missing values",
                "After": "Rows with incomplete required OHLCV fields removed",
                "Reason": "Avoid invalid prices and unsupported model inputs.",
            },
            {
                "Step": "Duplicate dates",
                "Before": f"{duplicates} duplicate dates",
                "After": "One observation per date",
                "Reason": "Maintain one unambiguous daily record.",
            },
            {
                "Step": "Final row count",
                "Before": f"{rows_before:,} rows",
                "After": f"{rows_after:,} rows",
                "Reason": "Make preprocessing loss explicit and reproducible.",
            },
        ]
    )


def split_continuity_audit(data: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Check known split dates for mechanical price discontinuities.

    This is an empirical continuity check, not provider metadata proving that
    every corporate action or total-return adjustment has been applied.
    """
    events = KNOWN_SPLIT_EVENTS.get(str(asset).upper(), ())
    if not events or data.empty:
        return pd.DataFrame()

    frame = data[["Date", "Price"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame = frame.dropna().sort_values("Date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for effective_date_text, event, factor in events:
        effective_date = pd.Timestamp(effective_date_text)
        matches = frame.index[frame["Date"].eq(effective_date)]
        if len(matches) == 0:
            rows.append(
                {
                    "Effective Date": effective_date,
                    "Corporate Action": event,
                    "Previous Session": pd.NaT,
                    "Previous Price": np.nan,
                    "Effective-Date Price": np.nan,
                    "Observed One-Day Return": np.nan,
                    "Unadjusted Mechanical Return": 1 / factor - 1,
                    "Continuity Status": "Not testable",
                }
            )
            continue

        current_index = int(matches[0])
        if current_index == 0:
            continue
        previous = frame.iloc[current_index - 1]
        current = frame.iloc[current_index]
        observed_return = float(current["Price"] / previous["Price"] - 1)
        rows.append(
            {
                "Effective Date": effective_date,
                "Corporate Action": event,
                "Previous Session": previous["Date"],
                "Previous Price": float(previous["Price"]),
                "Effective-Date Price": float(current["Price"]),
                "Observed One-Day Return": observed_return,
                "Unadjusted Mechanical Return": 1 / factor - 1,
                "Continuity Status": (
                    "Consistent with split-adjusted continuity"
                    if abs(observed_return) < 0.50
                    else "Potential unadjusted split break"
                ),
            }
        )
    return pd.DataFrame(rows)


def load_builtin_datasets() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all bundled real datasets and report their compatible common window."""
    datasets: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for asset, path in DATASET_PATHS.items():
        cleaned, _ = load_and_preprocess_csv(path)
        datasets[asset] = cleaned
        rows.append(
            {
                "Asset": asset,
                "File": path.name,
                "Rows": len(cleaned),
                "Start Date": cleaned["Date"].min(),
                "End Date": cleaned["Date"].max(),
                "Required Fields Present": all(column in cleaned.columns for column in REQUIRED_COLUMNS),
            }
        )
    return datasets, pd.DataFrame(rows)


def align_datasets_to_common_window(
    datasets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    """Restrict datasets to one shared calendar window for fair comparisons."""
    if not datasets:
        raise ValueError("At least one dataset is required.")
    common_start = max(frame["Date"].min() for frame in datasets.values())
    common_end = min(frame["Date"].max() for frame in datasets.values())
    if common_start >= common_end:
        raise ValueError("Datasets do not share a valid date range.")
    aligned = {
        asset: frame.loc[
            frame["Date"].between(common_start, common_end, inclusive="both")
        ].reset_index(drop=True)
        for asset, frame in datasets.items()
    }
    return aligned, pd.Timestamp(common_start), pd.Timestamp(common_end)


def load_and_preprocess_csv(
    source: str | Path | IO[str] | IO[bytes] | BytesIO | StringIO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return preprocess_stock_data(_read_csv(source))


def save_cleaned_dataset(df: pd.DataFrame, path: Path = CLEANED_DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return numerical statistics only, avoiding datetime display issues."""
    return df.select_dtypes(include=np.number).describe().T.round(4)


def date_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Item": ["Start Date", "End Date", "Number of Trading Records"],
            "Value": [
                df["Date"].min().strftime("%Y-%m-%d"),
                df["Date"].max().strftime("%Y-%m-%d"),
                f"{len(df):,}",
            ],
        }
    )
