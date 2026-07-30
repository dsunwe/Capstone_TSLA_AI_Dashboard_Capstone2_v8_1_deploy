from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CANDIDATE_FEATURES,
    FEATURE_GROUPS,
    FEATURES_DATA_PATH,
    SELECTED_FEATURES,
    SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
    SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
    SUPPLEMENTARY_TARGET_RETURN_COLUMN,
    TARGET_ABS_RETURN_COLUMN,
    TARGET_DATE_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_HORIZON_SESSIONS,
    TARGET_RETURN_COLUMN,
    TARGET_TRAILING_SESSIONS,
    TARGET_VALUATION_DATE_COLUMN,
)


def add_features(
    df: pd.DataFrame,
    context_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create time-series features using information available at day t or earlier."""
    data = df.copy()
    if "Price" not in data.columns:
        data["Price"] = data["Close"]

    price = data["Price"].replace(0, np.nan)
    open_price = data["Open"].replace(0, np.nan)

    data["Daily_Return"] = data["Price"].pct_change()
    data["Cumulative_Return"] = (1 + data["Daily_Return"].fillna(0)).cumprod() - 1
    data["Log_Return"] = np.log(price / price.shift(1))
    data["Lag_Return_1"] = data["Daily_Return"].shift(1)
    data["Lag_Return_3"] = data["Daily_Return"].shift(3)
    data["Lag_Return_5"] = data["Daily_Return"].shift(5)
    data["Momentum_5"] = data["Price"].pct_change(5)
    data["Momentum_10"] = data["Price"].pct_change(10)

    for window in [5, 10, 20, 50, 100]:
        data[f"SMA{window}"] = data["Price"].rolling(window=window).mean()

    data["SMA10_Ratio"] = price / data["SMA10"]
    data["SMA20_Ratio"] = price / data["SMA20"]
    data["SMA50_Ratio"] = price / data["SMA50"]

    for span in [8, 13, 21, 50]:
        data[f"EMA{span}"] = data["Price"].ewm(span=span, adjust=False).mean()

    data["EMA8_Ratio"] = price / data["EMA8"]
    data["EMA13_Ratio"] = price / data["EMA13"]
    data["EMA21_Ratio"] = price / data["EMA21"]
    data["EMA8_13_State"] = (data["EMA8"] > data["EMA13"]).astype(int)
    data["EMA_Alignment"] = (
        (data["EMA8"] > data["EMA13"]) & (data["EMA13"] > data["EMA21"])
    ).astype(int)

    ema12 = data["Price"].ewm(span=12, adjust=False).mean()
    ema26 = data["Price"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

    price_change = data["Price"].diff()
    average_gain = price_change.clip(lower=0).rolling(window=14).mean()
    average_loss = -price_change.clip(upper=0).rolling(window=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + relative_strength))

    data["Volatility_10"] = data["Daily_Return"].rolling(window=10).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    previous_close = data["Price"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR_14"] = true_range.rolling(window=14).mean() / price
    rolling_std_20 = data["Price"].rolling(window=20).std()
    data["Bollinger_Band_Width"] = (4 * rolling_std_20) / data["SMA20"].replace(0, np.nan)

    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volume_MA10"] = data["Volume"].rolling(window=10).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA10"].replace(0, np.nan)
    direction = np.sign(data["Daily_Return"].fillna(0))
    data["OBV"] = (direction * data["Volume"]).cumsum()
    data["OBV_Change"] = data["OBV"].diff() / data["Volume_MA10"].replace(0, np.nan)

    data["High_Low_Range"] = (data["High"] - data["Low"]) / price
    data["Open_Close_Change"] = (data["Price"] - data["Open"]) / open_price

    data["Rolling_Peak"] = data["Price"].cummax()
    data["Drawdown"] = price / data["Rolling_Peak"] - 1

    context_columns = [
        "QQQ_Return",
        "QQQ_Volatility_20",
        "SP500_Return",
        "SP500_Volatility_20",
        "NASDAQ_Return",
        "NASDAQ_Volatility_20",
        "Fed_Funds_Rate",
        "Treasury_10Y_Yield",
        "Yield_Curve_10Y2Y",
        "CPI_YoY",
        "Unemployment_Rate",
        "VIX_Level",
        "VIX_Change_5",
        "Consumer_Sentiment",
        "Consumer_Sentiment_Change",
    ]
    if context_data is None:
        # This fallback keeps uploaded/offline technical-only datasets usable.
        # Formal bundled experiments always pass the release-aware context frame.
        data["QQQ_Return"] = data["Daily_Return"]
        data["QQQ_Volatility_20"] = data["Volatility_20"]
        data["SP500_Return"] = data["Daily_Return"]
        data["SP500_Volatility_20"] = data["Volatility_20"]
        data["NASDAQ_Return"] = data["Daily_Return"]
        data["NASDAQ_Volatility_20"] = data["Volatility_20"]
        for column in context_columns[6:]:
            data[column] = np.nan
    else:
        available = [
            column
            for column in context_columns
            if column in context_data.columns
        ]
        context = context_data[["Date", *available]].copy().sort_values("Date")
        data = data.merge(context, on="Date", how="left", validate="one_to_one")
        for column in context_columns:
            if column not in data.columns:
                data[column] = np.nan

    # Retained as non-candidate aliases for compatibility with v5 exports.
    data["Market_Return"] = data["QQQ_Return"]
    data["Market_Volatility_20"] = data["QQQ_Volatility_20"]

    # The main v8 target asks whether the mean of the next five closing prices
    # exceeds the mean of the trailing five closes available at t. This target
    # is based only on the five future sessions and is purged by five origins at
    # every temporal boundary. The downstream application layer is separate:
    # a signal available after Close[t] executes at Open[t+1] and the same
    # sleeve is reconsidered/closed at Open[t+6]. The two concepts share a
    # five-session horizon but are not mathematically identical.
    trailing_mean = data["Price"].rolling(TARGET_TRAILING_SESSIONS).mean()
    future_mean = (
        data["Price"]
        .shift(-1)
        .rolling(TARGET_HORIZON_SESSIONS)
        .mean()
        .shift(-(TARGET_HORIZON_SESSIONS - 1))
    )
    execution_open = data["Open"].shift(-1)
    valuation_open = data["Open"].shift(-(TARGET_HORIZON_SESSIONS + 1))
    data[TARGET_EXECUTION_DATE_COLUMN] = data["Date"].shift(-1)
    data[TARGET_VALUATION_DATE_COLUMN] = data["Date"].shift(
        -(TARGET_HORIZON_SESSIONS + 1)
    )
    data[TARGET_DATE_COLUMN] = data["Date"].shift(-TARGET_HORIZON_SESSIONS)
    data["Trailing_Week_Mean_Close"] = trailing_mean
    data["Future_Week_Mean_Close"] = future_mean
    data[TARGET_RETURN_COLUMN] = future_mean / trailing_mean - 1.0
    data[TARGET_ABS_RETURN_COLUMN] = data[TARGET_RETURN_COLUMN].abs()
    data[TARGET_DIRECTION_COLUMN] = np.where(
        data[TARGET_RETURN_COLUMN].isna(),
        np.nan,
        (data[TARGET_RETURN_COLUMN] > 0).astype(int),
    )
    data["Five_Session_Open_to_Open_Return"] = (
        valuation_open / execution_open - 1.0
    )

    # The bounded FinBERT supplementary study retains its original
    # close-to-close target as a separately named, non-trading outcome.
    data[SUPPLEMENTARY_TARGET_RETURN_COLUMN] = data["Daily_Return"].shift(-1)
    data[SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN] = data[
        SUPPLEMENTARY_TARGET_RETURN_COLUMN
    ].abs()
    data[SUPPLEMENTARY_TARGET_DIRECTION_COLUMN] = np.where(
        data[SUPPLEMENTARY_TARGET_RETURN_COLUMN].isna(),
        np.nan,
        (data[SUPPLEMENTARY_TARGET_RETURN_COLUMN] > 0).astype(int),
    )

    return data.replace([np.inf, -np.inf], np.nan)


def save_feature_dataset(df: pd.DataFrame, path=FEATURES_DATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def feature_group_table() -> pd.DataFrame:
    rows = []
    for group, details in FEATURE_GROUPS.items():
        rows.append(
            {
                "Feature Group": group,
                "Features": ", ".join(details["features"]),
                "Purpose": details["purpose"],
            }
        )
    return pd.DataFrame(rows)


def selected_feature_table(features: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"Selected ML Feature": features or SELECTED_FEATURES})


def candidate_feature_table() -> pd.DataFrame:
    group_lookup = {
        feature: group
        for group, details in FEATURE_GROUPS.items()
        if group != "Target"
        for feature in details["features"]
    }
    return pd.DataFrame(
        {
            "Candidate Feature": CANDIDATE_FEATURES,
            "Feature Group": [group_lookup.get(feature, "Other") for feature in CANDIDATE_FEATURES],
            "Information Cut-off": "Day t or release-aware as-of value",
        }
    )


def feature_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_names: list[str] = []
    for details in FEATURE_GROUPS.values():
        feature_names.extend(details["features"])
    available = [name for name in dict.fromkeys(feature_names) if name in df.columns]
    result = df[available].isna().sum().rename("Missing Values").reset_index()
    result = result.rename(columns={"index": "Feature"})
    result["Missing %"] = (result["Missing Values"] / max(len(df), 1) * 100).round(2)
    return result


def build_model_dataset(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    features = feature_names or SELECTED_FEATURES
    missing_columns = [feature for feature in features if feature not in df.columns]
    if missing_columns:
        raise ValueError(
            "Model features are missing from the engineered dataset: "
            + ", ".join(missing_columns)
        )

    # Keep feature-level missing values for the training-fitted imputers used by
    # every model family. Dropping rows separately for each feature subset would
    # change the chronological 80/20 boundary and make feature/model comparisons
    # use different held-out dates. Only the anchor fields and target labels are
    # required to be present here; the canonical temporal plan applies one
    # common warm-up, purge and test universe across all specifications.
    required = ["Date", "Price", TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]
    return df.dropna(subset=required).copy()
