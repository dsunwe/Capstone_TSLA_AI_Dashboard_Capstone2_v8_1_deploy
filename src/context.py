from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CONTEXT_SERIES_PATHS,
    MARKET_PROXY_PATHS,
    SOURCE_MANIFEST_PATH,
)
from .data import load_and_preprocess_csv


@dataclass
class ContextResult:
    frame: pd.DataFrame
    audit: pd.DataFrame
    manifest: pd.DataFrame


def _load_fred_series(series_id: str) -> pd.DataFrame:
    path = CONTEXT_SERIES_PATHS[series_id]
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return (
        frame.dropna(subset=["Date", series_id])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def _with_available_date(series_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a conservative information-availability date to source observations."""
    result = frame.copy()
    if series_id == "DFF":
        result["Available_Date"] = result["Date"] + pd.Timedelta(days=1)
    elif series_id == "CPIAUCSL":
        result["Available_Date"] = (
            result["Date"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
        )
    elif series_id == "UNRATE":
        result["Available_Date"] = (
            result["Date"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=7)
        )
    elif series_id == "UMCSENT":
        # FRED states that this public series is delayed by one month. The end
        # of the following month is a conservative reproducible cut-off.
        result["Available_Date"] = result["Date"] + pd.offsets.MonthEnd(2)
    else:
        result["Available_Date"] = result["Date"]
    return result


def _asof_merge(
    base: pd.DataFrame,
    source: pd.DataFrame,
    *,
    series_id: str,
    value_columns: list[str],
) -> pd.DataFrame:
    available_column = f"{series_id}_Available_Date"
    prepared = source.rename(columns={"Available_Date": available_column}).sort_values(
        available_column
    )
    merged = pd.merge_asof(
        base.sort_values("Date"),
        prepared[[available_column, *value_columns]],
        left_on="Date",
        right_on=available_column,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def _qqq_context() -> pd.DataFrame:
    qqq, _ = load_and_preprocess_csv(MARKET_PROXY_PATHS["QQQ"])
    result = qqq[["Date", "Close"]].copy().sort_values("Date")
    result["QQQ_Return"] = result["Close"].pct_change()
    result["QQQ_Volatility_20"] = result["QQQ_Return"].rolling(20).std()
    result["Available_Date"] = result["Date"]
    return result.rename(columns={"Close": "QQQ_Level"})


def _daily_market_series(
    series_id: str,
    *,
    level_name: str,
    return_name: str,
    volatility_name: str,
) -> pd.DataFrame:
    frame = _with_available_date(series_id, _load_fred_series(series_id))
    frame[level_name] = frame[series_id]
    frame[return_name] = frame[level_name].pct_change()
    frame[volatility_name] = frame[return_name].rolling(20).std()
    return frame[
        ["Date", "Available_Date", level_name, return_name, volatility_name]
    ]


def _macro_series() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    fed = _with_available_date("DFF", _load_fred_series("DFF")).rename(
        columns={"DFF": "Fed_Funds_Rate"}
    )
    ten_year = _with_available_date("DGS10", _load_fred_series("DGS10")).rename(
        columns={"DGS10": "Treasury_10Y_Yield"}
    )
    two_year = _with_available_date("DGS2", _load_fred_series("DGS2")).rename(
        columns={"DGS2": "Treasury_2Y_Yield"}
    )

    cpi = _load_fred_series("CPIAUCSL")
    cpi["CPI_YoY"] = cpi["CPIAUCSL"].pct_change(12) * 100
    cpi = _with_available_date("CPIAUCSL", cpi)

    unemployment = _with_available_date("UNRATE", _load_fred_series("UNRATE")).rename(
        columns={"UNRATE": "Unemployment_Rate"}
    )
    return {
        "DFF": (fed, ["Fed_Funds_Rate"]),
        "DGS10": (ten_year, ["Treasury_10Y_Yield"]),
        "DGS2": (two_year, ["Treasury_2Y_Yield"]),
        "CPIAUCSL": (cpi, ["CPI_YoY"]),
        "UNRATE": (unemployment, ["Unemployment_Rate"]),
    }


def _sentiment_series() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    vix = _with_available_date("VIXCLS", _load_fred_series("VIXCLS"))
    vix["VIX_Level"] = vix["VIXCLS"]
    vix["VIX_Change_5"] = vix["VIX_Level"].pct_change(5)

    consumer = _load_fred_series("UMCSENT")
    consumer["Consumer_Sentiment"] = consumer["UMCSENT"]
    consumer["Consumer_Sentiment_Change"] = consumer["UMCSENT"].diff()
    consumer = _with_available_date("UMCSENT", consumer)
    return {
        "VIXCLS": (vix, ["VIX_Level", "VIX_Change_5"]),
        "UMCSENT": (
            consumer,
            ["Consumer_Sentiment", "Consumer_Sentiment_Change"],
        ),
    }


def context_availability_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in sorted(c for c in frame.columns if c.endswith("_Available_Date")):
        available = pd.to_datetime(frame[column], errors="coerce")
        violations = int((available > frame["Date"]).fillna(False).sum())
        value_prefix = column.removesuffix("_Available_Date")
        rows.append(
            {
                "Source": value_prefix,
                "Rows": len(frame),
                "Available From": available.min(),
                "Available Through": available.max(),
                "Missing Availability Rows": int(available.isna().sum()),
                "Future Availability Violations": violations,
                "Status": "Pass" if violations == 0 else "Fail",
            }
        )
    return pd.DataFrame(rows)


def load_source_manifest() -> pd.DataFrame:
    if not SOURCE_MANIFEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SOURCE_MANIFEST_PATH)


def build_context_frame(trading_dates: pd.Series | pd.Index) -> ContextResult:
    """Create a point-in-time context frame for the supplied trading dates."""
    dates = pd.to_datetime(pd.Series(trading_dates), errors="coerce").dropna()
    base = pd.DataFrame({"Date": sorted(dates.unique())})

    qqq = _qqq_context()
    frame = _asof_merge(
        base,
        qqq,
        series_id="QQQ",
        value_columns=["QQQ_Level", "QQQ_Return", "QQQ_Volatility_20"],
    )

    for series_id, source, values in [
        (
            "SP500",
            _daily_market_series(
                "SP500",
                level_name="SP500_Level",
                return_name="SP500_Return",
                volatility_name="SP500_Volatility_20",
            ),
            ["SP500_Level", "SP500_Return", "SP500_Volatility_20"],
        ),
        (
            "NASDAQCOM",
            _daily_market_series(
                "NASDAQCOM",
                level_name="NASDAQ_Level",
                return_name="NASDAQ_Return",
                volatility_name="NASDAQ_Volatility_20",
            ),
            ["NASDAQ_Level", "NASDAQ_Return", "NASDAQ_Volatility_20"],
        ),
    ]:
        frame = _asof_merge(frame, source, series_id=series_id, value_columns=values)

    for series_id, (source, values) in {
        **_macro_series(),
        **_sentiment_series(),
    }.items():
        frame = _asof_merge(frame, source, series_id=series_id, value_columns=values)

    frame["Yield_Curve_10Y2Y"] = (
        frame["Treasury_10Y_Yield"] - frame["Treasury_2Y_Yield"]
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    audit = context_availability_audit(frame)
    return ContextResult(frame=frame, audit=audit, manifest=load_source_manifest())
