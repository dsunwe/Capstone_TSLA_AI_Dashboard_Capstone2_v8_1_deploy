from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from .config import (
    EMA_WINDOWS,
    SMA_WINDOWS,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_VALUATION_DATE_COLUMN,
)
from .metrics import calculate_backtest_metrics, equity_drawdown_frame
from .ml import MLResult


RULE_STRATEGY_SPECS = (
    ("Buy-and-Hold", "Buy-and-Hold_Return", "Buy-and-Hold_Position"),
    ("Price-above-SMA10", "Price-above-SMA10_Return", "SMA10_Position"),
    ("Price-above-SMA20", "Price-above-SMA20_Return", "SMA20_Position"),
    (
        "SMA10/SMA20 Crossover",
        "SMA10/SMA20 Crossover_Return",
        "SMA10_SMA20_Crossover_Position",
    ),
    ("Price-above-EMA8", "Price-above-EMA8_Return", "EMA8_Position"),
    ("Price-above-EMA13", "Price-above-EMA13_Return", "EMA13_Position"),
    ("Price-above-EMA21", "Price-above-EMA21_Return", "EMA21_Position"),
    (
        "EMA8/EMA13 Crossover",
        "EMA8/EMA13 Crossover_Return",
        "EMA8_EMA13_Crossover_Position",
    ),
)
ML_STRATEGY_SPEC = (
    "ML Prediction Strategy",
    "ML Prediction Strategy_Return",
    "ML_Position",
)


def available_strategy_specs(
    strategy_frame: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """Return the canonical strategies whose position and return columns exist."""
    specifications = [*RULE_STRATEGY_SPECS, ML_STRATEGY_SPEC]
    return [
        specification
        for specification in specifications
        if specification[1] in strategy_frame.columns
        and specification[2] in strategy_frame.columns
    ]


def _net_strategy_return(
    open_to_open_return: pd.Series,
    position: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Apply exact one-way frictions to next-open position changes.

    ``position`` is the desired long/cash position established at an execution
    open. It earns the return from that open to the subsequent trading open.
    Transaction cost and adverse slippage are each interpreted as one-way basis
    point rates. The four possible binary transitions are treated explicitly:

    * 0 -> 0: remain in cash;
    * 0 -> 1: buy at the execution open, paying entry slippage and cost;
    * 1 -> 1: remain invested without another charge;
    * 1 -> 0: sell at the execution open, paying exit slippage and cost.
    """
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction costs and slippage must be non-negative")
    if transaction_cost_bps >= 10_000 or slippage_bps >= 10_000:
        raise ValueError("transaction costs and slippage must each be below 10,000 bps")

    numeric_position = pd.to_numeric(
        pd.Series(position).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0)
    if not numeric_position.isin([0, 1]).all():
        raise ValueError("position must be binary (0 or 1)")
    numeric_position = numeric_position.astype(int)

    numeric_return = pd.to_numeric(
        pd.Series(open_to_open_return).reset_index(drop=True),
        errors="coerce",
    )
    if len(numeric_position) != len(numeric_return):
        raise ValueError("position and open-to-open return must have the same length")
    if numeric_return.isna().any() or not np.isfinite(numeric_return).all():
        raise ValueError("open-to-open return must contain finite numeric values")
    if (numeric_return <= -1).any():
        raise ValueError("open-to-open return values must be greater than -1.0")

    previous_position = numeric_position.shift(1).fillna(0).astype(int)
    price_ratio = 1.0 + numeric_return
    transaction_rate = transaction_cost_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    entry_factor = (1.0 + slippage_rate) * (1.0 + transaction_rate)
    exit_factor = (1.0 - slippage_rate) * (1.0 - transaction_rate)

    wealth_factor = pd.Series(1.0, index=numeric_position.index, dtype=float)
    entering = previous_position.eq(0) & numeric_position.eq(1)
    staying_long = previous_position.eq(1) & numeric_position.eq(1)
    exiting = previous_position.eq(1) & numeric_position.eq(0)
    wealth_factor.loc[entering] = price_ratio.loc[entering] / entry_factor
    wealth_factor.loc[staying_long] = price_ratio.loc[staying_long]
    wealth_factor.loc[exiting] = exit_factor
    return wealth_factor - 1.0


def build_trade_ledger(
    dates: pd.Series,
    position: pd.Series,
    next_day_return: pd.Series,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    prices: pd.Series | None = None,
    signal_dates: pd.Series | None = None,
    valuation_dates: pd.Series | None = None,
    valuation_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert next-open binary positions into an auditable trade ledger.

    For backward API compatibility, ``dates``, ``next_day_return`` and
    ``prices`` retain their original argument names. Under the corrected
    execution semantics they mean execution-open dates, subsequent
    open-to-open returns, and raw execution-open prices respectively.

    A position still active at the end is marked ``Open`` and valued at the
    supplied final ``valuation_date``/``valuation_price`` without a
    hypothetical exit charge.

    ``Winning Trade`` is defined using net return for closed trades. Open
    trades have a missing win flag and are excluded from the default
    trade-level win rate.
    """
    ledger_columns = [
        "Trade",
        "Entry Date",
        "Exit Date",
        "Status",
        "Holding Period (Trading Days)",
        "Gross Return",
        "Net Return",
        "Winning Trade",
        "Entry Signal Date",
        "Entry Execution Date",
        "Exit Signal Date",
        "Exit Execution Date",
        "Mark Date",
        "Mark Price",
        "Effective Entry Fill",
        "Effective Exit Fill",
        "Transaction Cost (bps)",
        "Slippage (bps)",
    ]
    if prices is not None:
        ledger_columns.extend(["Entry Price", "Exit Price"])

    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    raw_position = pd.Series(position).reset_index(drop=True)
    next_day_return = pd.Series(next_day_return).reset_index(drop=True)
    series_lengths = {len(dates), len(raw_position), len(next_day_return)}
    if prices is not None:
        prices = pd.Series(prices).reset_index(drop=True)
        series_lengths.add(len(prices))
    if signal_dates is not None:
        signal_dates = pd.to_datetime(
            pd.Series(signal_dates).reset_index(drop=True),
            errors="coerce",
        )
        series_lengths.add(len(signal_dates))
    if valuation_dates is not None:
        valuation_dates = pd.to_datetime(
            pd.Series(valuation_dates).reset_index(drop=True),
            errors="coerce",
        )
        series_lengths.add(len(valuation_dates))
    if valuation_prices is not None:
        valuation_prices = pd.to_numeric(
            pd.Series(valuation_prices).reset_index(drop=True),
            errors="coerce",
        )
        series_lengths.add(len(valuation_prices))
    if len(series_lengths) != 1:
        raise ValueError("all ledger inputs must have the same length")
    if dates.isna().any():
        raise ValueError("dates must contain valid date-like values")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be in chronological order")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction costs and slippage must be non-negative")
    if dates.empty:
        return pd.DataFrame(columns=ledger_columns)
    if signal_dates is None:
        signal_dates = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    elif signal_dates.isna().any():
        raise ValueError("signal_dates must contain valid date-like values")
    if valuation_dates is None:
        valuation_dates = dates.shift(-1)
    elif valuation_dates.isna().any():
        raise ValueError("valuation_dates must contain valid date-like values")
    if (signal_dates.notna() & signal_dates.ge(dates)).any():
        raise ValueError("each signal date must be earlier than its execution date")
    if (valuation_dates.notna() & valuation_dates.le(dates)).any():
        raise ValueError("each valuation date must be later than its execution date")

    numeric_position = pd.to_numeric(raw_position, errors="coerce").fillna(0.0)
    if not numeric_position.isin([0, 1]).all():
        raise ValueError("position must be binary (0 or 1)")
    numeric_position = numeric_position.astype(int)
    numeric_return = pd.to_numeric(next_day_return, errors="coerce").fillna(0.0)
    if not np.isfinite(numeric_return).all():
        raise ValueError("next_day_return must contain finite numeric values")
    if (numeric_return <= -1).any():
        raise ValueError("next_day_return values must be greater than -1.0")
    if prices is not None:
        prices = pd.to_numeric(prices, errors="coerce")
        if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("prices must contain finite positive execution-open values")
    if valuation_prices is None and prices is not None:
        valuation_prices = prices.shift(-1)
    if valuation_prices is not None:
        non_missing_valuation_prices = valuation_prices.dropna()
        if (
            not np.isfinite(non_missing_valuation_prices).all()
            or (non_missing_valuation_prices <= 0).any()
        ):
            raise ValueError("valuation_prices must contain finite positive values")

    net_daily_return = _net_strategy_return(
        numeric_return,
        numeric_position,
        transaction_cost_bps,
        slippage_bps,
    )
    previous_position = numeric_position.shift(1).fillna(0).astype(int)
    entry_indices = numeric_position.index[
        (numeric_position == 1) & (previous_position == 0)
    ].tolist()

    rows: list[dict[str, object]] = []
    for trade_number, entry_index in enumerate(entry_indices, start=1):
        later_flat = numeric_position.iloc[entry_index + 1 :]
        exit_candidates = later_flat.index[later_flat == 0]
        exit_index = int(exit_candidates[0]) if len(exit_candidates) else None
        active_end = exit_index - 1 if exit_index is not None else len(dates) - 1
        gross_return = float(
            (1.0 + numeric_return.iloc[entry_index : active_end + 1]).prod() - 1.0
        )
        net_end = exit_index if exit_index is not None else active_end
        net_return = float(
            (1.0 + net_daily_return.iloc[entry_index : net_end + 1]).prod() - 1.0
        )
        is_closed = exit_index is not None
        mark_date = (
            dates.iloc[exit_index]
            if is_closed
            else valuation_dates.iloc[active_end]
        )
        mark_price = np.nan
        entry_price = np.nan
        exit_price = np.nan
        effective_entry_fill = np.nan
        effective_exit_fill = np.nan
        if prices is not None:
            entry_price = float(prices.iloc[entry_index])
            if is_closed:
                exit_price = float(prices.iloc[exit_index])
                mark_price = exit_price
            elif valuation_prices is not None and pd.notna(
                valuation_prices.iloc[active_end]
            ):
                mark_price = float(valuation_prices.iloc[active_end])
            effective_entry_fill = entry_price * (1.0 + slippage_bps / 10_000)
            if is_closed:
                effective_exit_fill = exit_price * (1.0 - slippage_bps / 10_000)

        row: dict[str, object] = {
            "Trade": trade_number,
            "Entry Date": dates.iloc[entry_index],
            "Exit Date": dates.iloc[exit_index] if is_closed else pd.NaT,
            "Status": "Closed" if is_closed else "Open",
            "Holding Period (Trading Days)": active_end - entry_index + 1,
            "Gross Return": gross_return,
            "Net Return": net_return,
            "Winning Trade": bool(net_return > 0) if is_closed else pd.NA,
            "Entry Signal Date": signal_dates.iloc[entry_index],
            "Entry Execution Date": dates.iloc[entry_index],
            "Exit Signal Date": (
                signal_dates.iloc[exit_index] if is_closed else pd.NaT
            ),
            "Exit Execution Date": (
                dates.iloc[exit_index] if is_closed else pd.NaT
            ),
            "Mark Date": mark_date,
            "Mark Price": mark_price,
            "Effective Entry Fill": effective_entry_fill,
            "Effective Exit Fill": effective_exit_fill,
            "Transaction Cost (bps)": float(transaction_cost_bps),
            "Slippage (bps)": float(slippage_bps),
        }
        if prices is not None:
            row["Entry Price"] = entry_price
            row["Exit Price"] = exit_price
        rows.append(row)

    ledger = pd.DataFrame(rows, columns=ledger_columns)
    if ledger.empty:
        return ledger
    ledger["Trade"] = ledger["Trade"].astype("Int64")
    ledger["Holding Period (Trading Days)"] = ledger[
        "Holding Period (Trading Days)"
    ].astype("Int64")
    ledger["Winning Trade"] = ledger["Winning Trade"].astype("boolean")
    return ledger


def calculate_trade_level_win_rate(
    ledger: pd.DataFrame,
    *,
    return_column: str = "Net Return",
    include_open: bool = False,
) -> float:
    """Return the share of evaluated trades whose return is strictly positive."""
    if ledger.empty or return_column not in ledger.columns:
        return float("nan")
    evaluated = ledger
    if not include_open and "Status" in evaluated.columns:
        evaluated = evaluated[evaluated["Status"] == "Closed"]
    trade_returns = pd.to_numeric(evaluated[return_column], errors="coerce").dropna()
    return (
        float((trade_returns > 0).mean())
        if not trade_returns.empty
        else float("nan")
    )


def _execution_aligned_frame(engineered: pd.DataFrame) -> pd.DataFrame:
    """Attach causal next-open execution and subsequent-open valuation fields."""
    required = {
        "Date",
        "Open",
        "Price",
        "SMA10",
        "SMA20",
        "EMA8",
        "EMA13",
        "EMA21",
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_EXECUTION_DATE_COLUMN,
        TARGET_VALUATION_DATE_COLUMN,
    }
    missing = sorted(required.difference(engineered.columns))
    if missing:
        raise ValueError(
            "Next-open backtesting requires columns: " + ", ".join(missing)
        )

    frame = engineered[
        [
            "Date",
            "Open",
            "Price",
            "SMA10",
            "SMA20",
            "EMA8",
            "EMA13",
            "EMA21",
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
            TARGET_EXECUTION_DATE_COLUMN,
            TARGET_VALUATION_DATE_COLUMN,
        ]
    ].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Open"] = pd.to_numeric(frame["Open"], errors="coerce")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    if frame["Date"].isna().any() or frame["Date"].duplicated().any():
        raise ValueError("engineered dates must be valid and unique")
    if frame["Open"].isna().any() or not np.isfinite(frame["Open"]).all():
        raise ValueError("Open must contain finite numeric values")
    if (frame["Open"] <= 0).any():
        raise ValueError("Open values must be positive")

    frame["Signal Date"] = frame["Date"]
    frame["Signal Price"] = frame["Price"]
    frame["Execution Date"] = frame["Date"].shift(-1)
    frame["Execution Open"] = frame["Open"].shift(-1)
    frame["Valuation Date"] = frame["Date"].shift(-2)
    frame["Valuation Open"] = frame["Open"].shift(-2)
    frame["Open-to-Open Return"] = (
        frame["Valuation Open"] / frame["Execution Open"] - 1.0
    )
    comparable = frame.dropna(
        subset=[
            TARGET_RETURN_COLUMN,
            TARGET_EXECUTION_DATE_COLUMN,
            TARGET_VALUATION_DATE_COLUMN,
            "Open-to-Open Return",
        ]
    )
    returns_match = np.allclose(
        comparable[TARGET_RETURN_COLUMN],
        comparable["Open-to-Open Return"],
        rtol=0.0,
        atol=1e-12,
    )
    dates_match = (
        pd.to_datetime(comparable[TARGET_EXECUTION_DATE_COLUMN]).reset_index(drop=True)
        .equals(
            pd.to_datetime(comparable["Execution Date"]).reset_index(drop=True)
        )
        and pd.to_datetime(
            comparable[TARGET_VALUATION_DATE_COLUMN]
        ).reset_index(drop=True).equals(
            pd.to_datetime(comparable["Valuation Date"]).reset_index(drop=True)
        )
    )
    if comparable.empty or not returns_match or not dates_match:
        raise ValueError(
            "Prediction target and backtest execution interval are not identical."
        )
    return frame


def build_strategy_frame(
    engineered: pd.DataFrame,
    ml_result: MLResult | None,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    full_period: bool = False,
) -> pd.DataFrame:
    """Build an executable long/cash strategy frame.

    Day-t end-of-day information creates a signal. The desired position is
    established at the next trading open and valued at the subsequent trading
    open. Compatibility aliases ``Date`` and ``Price`` refer to execution date
    and execution open in the returned backtest frame. The canonical target
    return remains explicitly named and is asserted equal to the realised
    gross ``Open-to-Open Return`` on every retained row.
    """
    frame = _execution_aligned_frame(engineered)

    frame["Buy-and-Hold_Position"] = 1
    frame["SMA10_Position"] = (frame["Price"] > frame["SMA10"]).astype(int)
    frame["SMA20_Position"] = (frame["Price"] > frame["SMA20"]).astype(int)
    frame["SMA10_SMA20_Crossover_Position"] = (
        frame["SMA10"] > frame["SMA20"]
    ).astype(int)
    frame["EMA8_Position"] = (frame["Price"] > frame["EMA8"]).astype(int)
    frame["EMA13_Position"] = (frame["Price"] > frame["EMA13"]).astype(int)
    frame["EMA21_Position"] = (frame["Price"] > frame["EMA21"]).astype(int)
    frame["EMA8_EMA13_Crossover_Position"] = (
        frame["EMA8"] > frame["EMA13"]
    ).astype(int)

    if ml_result is not None and not full_period:
        available_signal_columns = [
            column
            for column in ["Date", "Predicted Probability", "Signal"]
            if column in ml_result.results.columns
        ]
        if not {"Date", "Signal"}.issubset(available_signal_columns):
            return frame.iloc[0:0]
        ml_signal = ml_result.results[available_signal_columns].copy()
        ml_signal["Date"] = pd.to_datetime(ml_signal["Date"], errors="coerce")
        ml_signal = ml_signal.rename(columns={"Date": "Signal Date"})
        frame = frame.merge(
            ml_signal,
            on="Signal Date",
            how="inner",
            validate="one_to_one",
        )
        frame["ML_Position"] = frame["Signal"].astype(int)
    elif not full_period:
        return frame.iloc[0:0]

    frame = frame.dropna(
        subset=[
            "Signal Date",
            "Execution Date",
            "Valuation Date",
            "Execution Open",
            "Valuation Open",
            "Open-to-Open Return",
            "Price",
            "SMA10",
            "SMA20",
            "EMA8",
            "EMA13",
            "EMA21",
        ]
    ).reset_index(drop=True)
    frame["Date"] = frame["Execution Date"]
    frame["Price"] = frame["Execution Open"]

    frame["Buy-and-Hold_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["Buy-and-Hold_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["Price-above-SMA10_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["SMA10_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["Price-above-SMA20_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["SMA20_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["SMA10/SMA20 Crossover_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["SMA10_SMA20_Crossover_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["Price-above-EMA8_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["EMA8_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["Price-above-EMA13_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["EMA13_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["Price-above-EMA21_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["EMA21_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    frame["EMA8/EMA13 Crossover_Return"] = _net_strategy_return(
        frame["Open-to-Open Return"],
        frame["EMA8_EMA13_Crossover_Position"],
        transaction_cost_bps,
        slippage_bps,
    )
    # Backward-compatible return aliases for older callers. Public model labels
    # use the accurate Price-above-SMA names below.
    frame["SMA10 Rule_Return"] = frame["Price-above-SMA10_Return"]
    frame["SMA20 Rule_Return"] = frame["Price-above-SMA20_Return"]
    if "ML_Position" in frame.columns:
        frame["ML Prediction Strategy_Return"] = _net_strategy_return(
            frame["Open-to-Open Return"],
            frame["ML_Position"],
            transaction_cost_bps,
            slippage_bps,
        )
    return frame


def build_backtest_comparison(
    engineered: pd.DataFrame,
    ml_result: MLResult | None,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    full_period: bool = False,
    ml_strategy_name: str = "ML Prediction Strategy",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = build_strategy_frame(
        engineered,
        ml_result,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        full_period=full_period,
    )
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), frame

    strategies = available_strategy_specs(frame)

    metrics_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    for model_name, return_column, position_column in strategies:
        display_name = (
            ml_strategy_name if model_name == ML_STRATEGY_SPEC[0] else model_name
        )
        metrics = calculate_backtest_metrics(
            frame[return_column],
            frame[position_column],
            frame["Valuation Date"],
        )
        ledger = build_trade_ledger(
            frame["Execution Date"],
            frame[position_column],
            frame["Open-to-Open Return"],
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            prices=frame["Execution Open"],
            signal_dates=frame["Signal Date"],
            valuation_dates=frame["Valuation Date"],
            valuation_prices=frame["Valuation Open"],
        )
        metrics["Closed Trades"] = int(ledger["Status"].eq("Closed").sum())
        metrics["Open Trades"] = int(ledger["Status"].eq("Open").sum())
        metrics["Trade-Level Win Rate"] = calculate_trade_level_win_rate(ledger)
        metrics_rows.append({"Model": display_name, **metrics})
        equity_frames.append(
            equity_drawdown_frame(
                frame["Valuation Date"],
                frame[return_column],
                display_name,
            )
        )

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(equity_frames, ignore_index=True),
        frame,
    )


def build_oof_backtest_comparison(
    engineered: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest only expanding-window out-of-fold predictions."""
    required = {"Date", "Predicted Probability", "Signal"}
    if oof_predictions.empty or not required.issubset(oof_predictions.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    result_proxy = SimpleNamespace(results=oof_predictions[list(required)].copy())
    metrics, equity, frame = build_backtest_comparison(
        engineered,
        result_proxy,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    if not metrics.empty:
        metrics.insert(0, "Validation", "Walk-forward out-of-fold")
    if not equity.empty:
        equity.insert(0, "Validation", "Walk-forward out-of-fold")
    return metrics, equity, frame


def build_multi_model_backtest_comparison(
    engineered: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    validation_label: str = "Walk-forward out-of-fold",
    strategy_suffix: str = "OOF Signal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest date-aligned signals from every model level on one market path."""
    required = {
        "Model",
        "Date",
        "Predicted Probability",
        "Signal",
        TARGET_RETURN_COLUMN,
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prepared = predictions.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="raise")
    if prepared.duplicated(["Model", "Date"]).any():
        raise ValueError("Each model may provide at most one prediction per signal date.")

    model_order = prepared["Model"].drop_duplicates().tolist()
    expected_dates: pd.DatetimeIndex | None = None
    metrics_parts: list[pd.DataFrame] = []
    equity_parts: list[pd.DataFrame] = []
    execution_parts: list[pd.DataFrame] = []
    for index, model_name in enumerate(model_order):
        model_predictions = (
            prepared.loc[prepared["Model"].eq(model_name)]
            .drop(columns=["Model"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
        dates = pd.DatetimeIndex(model_predictions["Date"])
        if expected_dates is None:
            expected_dates = dates
        elif not dates.equals(expected_dates):
            raise ValueError("All model signals must use exactly the same OOF dates.")
        strategy_name = f"{model_name} {strategy_suffix}"
        proxy = SimpleNamespace(
            results=model_predictions[["Date", "Predicted Probability", "Signal"]]
        )
        metrics, equity, execution = build_backtest_comparison(
            engineered,
            proxy,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            ml_strategy_name=strategy_name,
        )
        if metrics.empty or execution.empty:
            raise RuntimeError(f"No executable backtest rows for {model_name}.")
        if index == 0:
            metrics_parts.append(
                metrics.loc[metrics["Model"].ne(strategy_name)]
            )
            equity_parts.append(
                equity.loc[equity["Model"].ne(strategy_name)]
            )
        metrics_parts.append(metrics.loc[metrics["Model"].eq(strategy_name)])
        equity_parts.append(equity.loc[equity["Model"].eq(strategy_name)])
        execution.insert(0, "Signal Model", model_name)
        execution_parts.append(execution)

    metrics_result = pd.concat(metrics_parts, ignore_index=True)
    equity_result = pd.concat(equity_parts, ignore_index=True)
    execution_result = pd.concat(execution_parts, ignore_index=True)
    metrics_result.insert(0, "Validation", validation_label)
    equity_result.insert(0, "Validation", validation_label)
    return metrics_result, equity_result, execution_result


def build_oof_signal_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise whether each OOF signal separates executable gross returns."""
    required = {
        "Model",
        "Signal",
        "Actual Direction",
        TARGET_RETURN_COLUMN,
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("Model", sort=False):
        signal = pd.to_numeric(group["Signal"], errors="raise").astype(int)
        actual = pd.to_numeric(
            group["Actual Direction"], errors="raise"
        ).astype(int)
        returns = pd.to_numeric(
            group[TARGET_RETURN_COLUMN], errors="raise"
        ).astype(float)
        long_returns = returns.loc[signal.eq(1)]
        cash_returns = returns.loc[signal.eq(0)]
        long_mean = float(long_returns.mean()) if not long_returns.empty else np.nan
        cash_mean = float(cash_returns.mean()) if not cash_returns.empty else np.nan
        rows.append(
            {
                "Model": model_name,
                "OOF Observations": len(group),
                "Direction Accuracy": float(signal.eq(actual).mean()),
                "Long Exposure": float(signal.mean()),
                "Long-Signal Mean Gross Return": long_mean,
                "Cash-Signal Market Mean Gross Return": cash_mean,
                "Gross Return Separation": long_mean - cash_mean,
            }
        )
    return pd.DataFrame(rows)


def run_single_window_ma_sensitivity(
    engineered: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    specifications = [
        *[
            (f"Price-above-SMA{window}", f"SMA{window}", "SMA")
            for window in SMA_WINDOWS
        ],
        *[
            (f"Price-above-EMA{window}", f"EMA{window}", "EMA")
            for window in EMA_WINDOWS
        ],
    ]
    ma_columns = [ma_column for _, ma_column, _ in specifications]

    # Calculate every moving average before selecting a shared evaluation period.
    # This prevents short-window rules from receiving more observations than
    # long-window rules simply because their indicators become available sooner.
    required = {"Date", "Open", "Price"}
    missing = sorted(required.difference(engineered.columns))
    if missing:
        raise ValueError(
            "Next-open sensitivity requires columns: " + ", ".join(missing)
        )
    frame = engineered[["Date", "Open", "Price"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    for window in SMA_WINDOWS:
        frame[f"SMA{window}"] = frame["Price"].rolling(window=window).mean()
    for window in EMA_WINDOWS:
        frame[f"EMA{window}"] = frame["Price"].ewm(span=window, adjust=False).mean()

    frame["Signal Date"] = frame["Date"]
    frame["Execution Date"] = frame["Date"].shift(-1)
    frame["Execution Open"] = frame["Open"].shift(-1)
    frame["Valuation Date"] = frame["Date"].shift(-2)
    frame["Valuation Open"] = frame["Open"].shift(-2)
    frame["Open-to-Open Return"] = (
        frame["Valuation Open"] / frame["Execution Open"] - 1.0
    )
    frame = frame.dropna(
        subset=[
            "Signal Date",
            "Execution Date",
            "Valuation Date",
            "Price",
            "Execution Open",
            "Valuation Open",
            "Open-to-Open Return",
            *ma_columns,
        ]
    ).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    signal_start = frame["Signal Date"].min()
    signal_end = frame["Signal Date"].max()
    evaluation_start = frame["Execution Date"].min()
    evaluation_end = frame["Valuation Date"].max()
    observations = len(frame)
    rows: list[dict[str, object]] = []

    for model_name, ma_column, ma_type in specifications:
        position = (frame["Price"] > frame[ma_column]).astype(int)
        returns = _net_strategy_return(
            frame["Open-to-Open Return"],
            position,
            transaction_cost_bps,
            slippage_bps,
        )
        metrics = calculate_backtest_metrics(
            returns,
            position,
            frame["Valuation Date"],
        )
        rows.append(
            {
                "Model": model_name,
                "Type": ma_type,
                "Window": int("".join(character for character in ma_column if character.isdigit())),
                "Signal Start": signal_start,
                "Signal End": signal_end,
                "Evaluation Start": evaluation_start,
                "Evaluation End": evaluation_end,
                "Observations": observations,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def run_single_window_sensitivity(
    engineered: pd.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Backward-compatible wrapper for the original public function name."""
    return run_single_window_ma_sensitivity(
        engineered,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
