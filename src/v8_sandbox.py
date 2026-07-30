"""Exploratory five-session downstream evaluation for the v8 Sandbox.

This module is deliberately separate from the frozen RQ4 research engine.  It
provides a reusable, side-effect-free implementation for uploaded or bundled
single-asset data.  A signal observed after ``Close[t]`` can first trade at
``Open[t+1]``.  It is assigned to one of five equal, self-financing sleeves and
that sleeve is reconsidered at ``Open[t+6]``.

The public metrics are decimal returns/exposures except for ``Cost Drag (pp)``,
which is expressed in percentage points.  No function writes files or mutates
the caller's frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HORIZON = 5
SLEEVE_COUNT = 5
TRADING_SESSIONS_PER_YEAR = 252
EXPLORATORY_SCOPE = "Interactive Sandbox analysis using the active settings."

EMA_STRATEGY = "EMA8/EMA13 State"
SMA_STRATEGY = "SMA10/SMA20 State"
BUY_HOLD_STRATEGY = "Buy-and-Hold"

PAST_MEAN_COLUMN = "Past 5-Session Mean Close"
FUTURE_MEAN_COLUMN = "Future 5-Session Mean Close"
FIVE_SESSION_TARGET_COLUMN = "Five-Session Mean Target"
FIVE_SESSION_TARGET_DATE_COLUMN = "Five-Session Target Date"


@dataclass(frozen=True)
class FiveSessionSandboxResult:
    """Tables returned by :func:`run_five_session_sandbox`.

    ``signal_ledger`` includes both executable and excluded prediction dates.
    ``sleeve_trade_ledger`` records only position changes; a scheduled
    ``1 -> 1`` continuation therefore produces no fee-bearing trade row.
    """

    metrics: pd.DataFrame
    equity_rows: pd.DataFrame
    sleeve_trade_ledger: pd.DataFrame
    signal_ledger: pd.DataFrame

    @property
    def equity(self) -> pd.DataFrame:
        """Backward-friendly alias for ``equity_rows``."""

        return self.equity_rows

    @property
    def trades(self) -> pd.DataFrame:
        """Backward-friendly alias for ``sleeve_trade_ledger``."""

        return self.sleeve_trade_ledger

    @property
    def signals(self) -> pd.DataFrame:
        """Backward-friendly alias for ``signal_ledger``."""

        return self.signal_ledger

    @property
    def excluded_signals(self) -> pd.DataFrame:
        """Return prediction rows excluded from the common executable grid."""

        return self.signal_ledger.loc[
            ~self.signal_ledger["Five-Session Executable"]
        ].copy()


def _normalise_dates(values: pd.Series, *, label: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError(f"{label} contains invalid dates.")
    return dates.dt.tz_convert(None).dt.normalize()


def _validate_rate(value: float, *, label: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0 or numeric >= 10_000:
        raise ValueError(f"{label} must be finite and in [0, 10,000) bps.")
    return numeric / 10_000.0


def _entry_exit_factors(
    transaction_cost_bps: float,
    slippage_bps: float,
) -> tuple[float, float]:
    transaction_rate = _validate_rate(
        transaction_cost_bps,
        label="transaction_cost_bps",
    )
    slippage_rate = _validate_rate(
        slippage_bps,
        label="slippage_bps",
    )
    entry_factor = 1.0 / (
        (1.0 + transaction_rate) * (1.0 + slippage_rate)
    )
    exit_factor = (1.0 - transaction_rate) * (1.0 - slippage_rate)
    return entry_factor, exit_factor


def _prepare_prices(
    prices: pd.DataFrame,
    *,
    date_column: str,
    open_column: str,
    close_column: str,
) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame.")
    required = [date_column, open_column, close_column]
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise ValueError("prices is missing columns: " + ", ".join(missing))

    frame = prices.copy(deep=True)
    frame[date_column] = _normalise_dates(
        frame[date_column],
        label=f"prices[{date_column!r}]",
    )
    frame[open_column] = pd.to_numeric(frame[open_column], errors="coerce")
    frame[close_column] = pd.to_numeric(frame[close_column], errors="coerce")
    if (
        frame[[open_column, close_column]].isna().any().any()
        or not np.isfinite(frame[[open_column, close_column]].to_numpy()).all()
        or (frame[[open_column, close_column]] <= 0).any().any()
    ):
        raise ValueError("Open and Close prices must be finite and positive.")

    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)
    if frame[date_column].duplicated().any():
        raise ValueError("prices contains duplicate trading dates.")

    close = frame[close_column].astype(float)
    derived = {
        "EMA8": close.ewm(span=8, adjust=False).mean(),
        "EMA13": close.ewm(span=13, adjust=False).mean(),
        "SMA10": close.rolling(window=10).mean(),
        "SMA20": close.rolling(window=20).mean(),
    }
    for column, computed in derived.items():
        if column in frame.columns:
            existing = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = existing.where(np.isfinite(existing), np.nan)
            frame[column] = frame[column].combine_first(computed)
        else:
            frame[column] = computed

    frame["_Price Row"] = np.arange(len(frame), dtype=int)
    return frame


def add_five_session_mean_target(
    prices: pd.DataFrame,
    *,
    date_column: str = "Date",
    close_column: str = "Close",
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Add the RQ3-style future-mean classification target.

    The target equals one when the mean of ``Close[t+1:t+horizon]`` exceeds the
    trailing mean ``Close[t-horizon+1:t]``.  This helper is provided for
    exploratory Sandbox use; the downstream engine itself only consumes
    already-generated predictions.
    """

    if int(horizon) != horizon or horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    horizon = int(horizon)
    frame = _prepare_prices(
        prices,
        date_column=date_column,
        open_column="Open",
        close_column=close_column,
    )
    close = frame[close_column].astype(float)
    past_mean = close.rolling(window=horizon).mean()
    future_mean = sum(close.shift(-step) for step in range(1, horizon + 1))
    future_mean = future_mean / float(horizon)
    eligible = past_mean.notna() & future_mean.notna()
    target = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    target.loc[eligible] = (
        future_mean.loc[eligible] > past_mean.loc[eligible]
    ).astype(int)

    frame[PAST_MEAN_COLUMN] = past_mean
    frame[FUTURE_MEAN_COLUMN] = future_mean
    frame[FIVE_SESSION_TARGET_COLUMN] = target
    frame[FIVE_SESSION_TARGET_DATE_COLUMN] = frame[date_column].shift(-horizon)
    return frame.drop(columns=["_Price Row"])


def _detect_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"predictions is missing column: {requested}")
        return requested
    return next((column for column in candidates if column in frame.columns), None)


def _prepare_predictions(
    predictions: pd.DataFrame,
    *,
    prediction_date_column: str,
    probability_column: str | None,
    signal_column: str | None,
    probability_threshold: float,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty pandas DataFrame.")
    if prediction_date_column not in predictions.columns:
        raise ValueError(
            f"predictions is missing column: {prediction_date_column}"
        )
    threshold = float(probability_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1.")

    frame = predictions.copy(deep=True)
    frame["Signal Date"] = _normalise_dates(
        frame[prediction_date_column],
        label=f"predictions[{prediction_date_column!r}]",
    )
    if frame["Signal Date"].duplicated().any():
        raise ValueError("predictions contains duplicate signal dates.")

    resolved_signal = _detect_column(
        frame,
        signal_column,
        ("Signal", "Prediction"),
    )
    resolved_probability = _detect_column(
        frame,
        probability_column,
        ("Predicted Probability", "Probability", "probability"),
    )
    if resolved_signal is None and resolved_probability is None:
        raise ValueError(
            "predictions must contain a binary Signal/Prediction column or a "
            "Predicted Probability/Probability column."
        )

    if resolved_probability is not None:
        probability = pd.to_numeric(
            frame[resolved_probability],
            errors="coerce",
        )
        if (
            probability.isna().any()
            or not np.isfinite(probability).all()
            or not probability.between(0.0, 1.0).all()
        ):
            raise ValueError("prediction probabilities must be finite in [0, 1].")
        frame["Prediction Probability"] = probability.astype(float)
    else:
        frame["Prediction Probability"] = np.nan

    if resolved_signal is not None:
        signal = pd.to_numeric(frame[resolved_signal], errors="coerce")
        if signal.isna().any() or not signal.isin([0, 1]).all():
            raise ValueError("prediction signals must be binary (0 or 1).")
        frame["Prediction Signal"] = signal.astype(int)
    else:
        frame["Prediction Signal"] = (
            frame["Prediction Probability"] >= threshold
        ).astype(int)

    return (
        frame[["Signal Date", "Prediction Probability", "Prediction Signal"]]
        .sort_values("Signal Date", kind="stable")
        .reset_index(drop=True)
    )


def _build_signal_ledger(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    date_column: str,
) -> pd.DataFrame:
    price_index = pd.Series(
        prices["_Price Row"].to_numpy(),
        index=prices[date_column],
    )
    missing_dates = predictions.loc[
        ~predictions["Signal Date"].isin(price_index.index),
        "Signal Date",
    ]
    if not missing_dates.empty:
        rendered = ", ".join(
            timestamp.strftime("%Y-%m-%d")
            for timestamp in missing_dates.iloc[:5]
        )
        raise ValueError(
            "prediction dates are absent from the price calendar: " + rendered
        )

    ledger = predictions.copy()
    ledger["Signal Price Row"] = (
        ledger["Signal Date"].map(price_index).astype(int)
    )
    ledger["Execution Price Row"] = ledger["Signal Price Row"] + 1
    ledger["End Price Row"] = ledger["Signal Price Row"] + HORIZON + 1
    ledger["Sleeve"] = ledger["Execution Price Row"] % SLEEVE_COUNT

    signal_rows = prices.iloc[
        ledger["Signal Price Row"].to_numpy(dtype=int)
    ].reset_index(drop=True)
    for column in ["Close", "EMA8", "EMA13", "SMA10", "SMA20"]:
        source_column = column
        if column == "Close" and column not in prices.columns:
            raise RuntimeError("Internal Close-column mapping failed.")
        ledger[f"Signal {column}"] = signal_rows[source_column].to_numpy()

    ledger["EMA8/EMA13 State Signal"] = (
        ledger["Signal EMA8"] > ledger["Signal EMA13"]
    ).astype(int)
    ledger["SMA10/SMA20 State Signal"] = (
        ledger["Signal SMA10"] > ledger["Signal SMA20"]
    ).astype(int)

    has_end = ledger["End Price Row"] < len(prices)
    has_moving_averages = ledger[
        ["Signal EMA8", "Signal EMA13", "Signal SMA10", "Signal SMA20"]
    ].notna().all(axis=1)
    ledger["Five-Session Executable"] = has_end & has_moving_averages
    ledger["Exclusion Reason"] = np.select(
        [
            ~has_end,
            has_end & ~has_moving_averages,
        ],
        [
            "missing Open[t+6]",
            "moving-average warm-up incomplete",
        ],
        default="",
    )

    ledger["Execution Date"] = pd.NaT
    ledger["Scheduled End Date"] = pd.NaT
    executable = ledger["Five-Session Executable"]
    execution_rows = ledger.loc[executable, "Execution Price Row"].astype(int)
    end_rows = ledger.loc[executable, "End Price Row"].astype(int)
    ledger.loc[executable, "Execution Date"] = prices.iloc[
        execution_rows.to_numpy()
    ][date_column].to_numpy()
    ledger.loc[executable, "Scheduled End Date"] = prices.iloc[
        end_rows.to_numpy()
    ][date_column].to_numpy()
    ledger["Execution Date"] = pd.to_datetime(ledger["Execution Date"])
    ledger["Scheduled End Date"] = pd.to_datetime(
        ledger["Scheduled End Date"]
    )
    return ledger


def _maximum_drawdown(values: np.ndarray) -> float:
    path = np.asarray(values, dtype=float)
    running_peak = np.maximum.accumulate(path)
    return float(np.min(path / running_peak - 1.0))


def _annualised_return(final_nav: float, intervals: int) -> float:
    if final_nav <= 0 or intervals <= 0:
        return np.nan
    return float(
        final_nav ** (TRADING_SESSIONS_PER_YEAR / float(intervals)) - 1.0
    )


def _strategy_metric(
    *,
    strategy: str,
    family: str,
    executable_rows: int,
    excluded_rows: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    intervals: int,
    net_path: np.ndarray,
    gross_path: np.ndarray,
    average_exposure: float,
    changes: int,
    terminal_all_cash: bool,
) -> dict[str, Any]:
    gross_return = float(gross_path[-1] - 1.0)
    net_return = float(net_path[-1] - 1.0)
    return {
        "Scope": EXPLORATORY_SCOPE,
        "Strategy": strategy,
        "Family": family,
        "Executable Prediction Dates": int(executable_rows),
        "Excluded Prediction Dates": int(excluded_rows),
        "Path Start": start_date,
        "Path End": end_date,
        "Market Intervals": int(intervals),
        "Gross Total Return": gross_return,
        "Net Total Return": net_return,
        "Annualized Net Return": _annualised_return(
            float(net_path[-1]),
            intervals,
        ),
        "Maximum Drawdown": _maximum_drawdown(net_path),
        "Average Exposure": float(average_exposure),
        "Cost Drag (pp)": float((gross_return - net_return) * 100.0),
        "One-way Position Changes": int(changes),
        "Terminal All Cash": bool(terminal_all_cash),
    }


def _simulate_sleeves(
    prices: pd.DataFrame,
    signal_ledger: pd.DataFrame,
    *,
    strategy: str,
    family: str,
    signal_column: str,
    date_column: str,
    open_column: str,
    entry_factor: float,
    exit_factor: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    executable = (
        signal_ledger.loc[signal_ledger["Five-Session Executable"]]
        .sort_values("Signal Date", kind="stable")
        .reset_index(drop=True)
    )
    if executable.empty:
        raise ValueError(
            "No prediction date has both moving-average history and Open[t+6]."
        )

    desired_values = executable[signal_column].astype(int)
    if not desired_values.isin([0, 1]).all():
        raise ValueError(f"{strategy} contains non-binary signals.")

    schedules: dict[int, dict[int, dict[str, Any]]] = {
        sleeve: {} for sleeve in range(SLEEVE_COUNT)
    }
    end_events: list[tuple[int, int, pd.Timestamp]] = []
    for (_, row), desired in zip(executable.iterrows(), desired_values):
        sleeve = int(row["Sleeve"])
        execution_row = int(row["Execution Price Row"])
        end_row = int(row["End Price Row"])
        schedules[sleeve][execution_row] = {
            "desired": int(desired),
            "event": "Signal decision",
            "origin_signal_date": row["Signal Date"],
            "scheduled_end_date": row["Scheduled End Date"],
        }
        end_events.append((sleeve, end_row, row["Signal Date"]))

    # A decision exactly at a preceding signal's t+6 endpoint takes precedence.
    # Otherwise that sleeve returns to cash at the endpoint, which is essential
    # when the uploaded prediction dates contain gaps.
    for sleeve, end_row, origin_signal_date in end_events:
        schedules[sleeve].setdefault(
            end_row,
            {
                "desired": 0,
                "event": "Five-session terminal close",
                "origin_signal_date": origin_signal_date,
                "scheduled_end_date": prices.iloc[end_row][date_column],
            },
        )

    start_row = int(executable["Execution Price Row"].min())
    final_row = int(executable["End Price Row"].max())
    grid = np.arange(start_row, final_row + 1, dtype=int)
    open_prices = prices[open_column].to_numpy(dtype=float)
    dates = prices[date_column].reset_index(drop=True)

    net_nav = np.full(SLEEVE_COUNT, 1.0 / SLEEVE_COUNT, dtype=float)
    gross_nav = np.full(SLEEVE_COUNT, 1.0 / SLEEVE_COUNT, dtype=float)
    positions = np.zeros(SLEEVE_COUNT, dtype=int)
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for grid_index, price_row in enumerate(grid):
        for sleeve in range(SLEEVE_COUNT):
            event = schedules[sleeve].get(int(price_row))
            if event is None:
                continue
            old_position = int(positions[sleeve])
            new_position = int(event["desired"])
            if old_position == new_position:
                continue
            nav_before = float(net_nav[sleeve])
            applied_factor = (
                entry_factor if (old_position, new_position) == (0, 1)
                else exit_factor
            )
            net_nav[sleeve] *= applied_factor
            positions[sleeve] = new_position
            trade_rows.append(
                {
                    "Scope": EXPLORATORY_SCOPE,
                    "Strategy": strategy,
                    "Open Date": dates.iloc[price_row],
                    "Price Row": int(price_row),
                    "Sleeve": int(sleeve),
                    "Event": event["event"],
                    "Origin Signal Date": event["origin_signal_date"],
                    "Scheduled End Date": event["scheduled_end_date"],
                    "Old Position": old_position,
                    "New Position": new_position,
                    "Transition": f"{old_position}->{new_position}",
                    "Net Sleeve NAV Before": nav_before,
                    "Applied Factor": float(applied_factor),
                    "Net Sleeve NAV After": float(net_nav[sleeve]),
                    "Friction Amount": float(nav_before - net_nav[sleeve]),
                    "Transaction Cost (bps)": float(transaction_cost_bps),
                    "Slippage (bps)": float(slippage_bps),
                }
            )

        net_total = float(net_nav.sum())
        gross_total = float(gross_nav.sum())
        net_exposure = float(
            np.dot(net_nav, positions) / net_total
            if net_total > 0
            else np.nan
        )
        gross_exposure = float(
            np.dot(gross_nav, positions) / gross_total
            if gross_total > 0
            else np.nan
        )
        equity_rows.append(
            {
                "Scope": EXPLORATORY_SCOPE,
                "Strategy": strategy,
                "Open Date": dates.iloc[price_row],
                "Price Row": int(price_row),
                "Open Price": float(open_prices[price_row]),
                "Net NAV": net_total,
                "Gross NAV": gross_total,
                "Net Exposure": net_exposure,
                "Gross Exposure": gross_exposure,
                "Active Sleeves": int(positions.sum()),
                **{
                    f"Sleeve {sleeve} Net NAV": float(net_nav[sleeve])
                    for sleeve in range(SLEEVE_COUNT)
                },
                **{
                    f"Sleeve {sleeve} Position": int(positions[sleeve])
                    for sleeve in range(SLEEVE_COUNT)
                },
            }
        )

        if grid_index == len(grid) - 1:
            break
        market_factor = open_prices[price_row + 1] / open_prices[price_row]
        net_nav *= np.where(positions == 1, market_factor, 1.0)
        gross_nav *= np.where(positions == 1, market_factor, 1.0)

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    terminal_all_cash = bool(equity.iloc[-1]["Active Sleeves"] == 0)
    if not terminal_all_cash:
        raise RuntimeError(f"{strategy} did not terminate fully in cash.")

    phase_nav_columns = [
        f"Sleeve {sleeve} Net NAV" for sleeve in range(SLEEVE_COUNT)
    ]
    if not np.isclose(
        equity.iloc[-1][phase_nav_columns].astype(float).sum(),
        float(equity.iloc[-1]["Net NAV"]),
        atol=1e-12,
    ):
        raise RuntimeError("Sleeve NAVs do not reconcile to portfolio NAV.")

    intervals = final_row - start_row
    interval_exposure = equity.iloc[:-1]["Net Exposure"].to_numpy(dtype=float)
    net_path = np.r_[1.0, equity["Net NAV"].to_numpy(dtype=float)]
    gross_path = np.r_[1.0, equity["Gross NAV"].to_numpy(dtype=float)]
    metric = _strategy_metric(
        strategy=strategy,
        family=family,
        executable_rows=len(executable),
        excluded_rows=int((~signal_ledger["Five-Session Executable"]).sum()),
        start_date=dates.iloc[start_row],
        end_date=dates.iloc[final_row],
        intervals=intervals,
        net_path=net_path,
        gross_path=gross_path,
        average_exposure=float(interval_exposure.mean()),
        changes=len(trades),
        terminal_all_cash=terminal_all_cash,
    )
    return metric, equity, trades


def _simulate_buy_and_hold(
    prices: pd.DataFrame,
    signal_ledger: pd.DataFrame,
    *,
    date_column: str,
    open_column: str,
    entry_factor: float,
    exit_factor: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    executable = signal_ledger.loc[signal_ledger["Five-Session Executable"]]
    start_row = int(executable["Execution Price Row"].min())
    final_row = int(executable["End Price Row"].max())
    grid = np.arange(start_row, final_row + 1, dtype=int)
    dates = prices[date_column].reset_index(drop=True)
    opens = prices[open_column].to_numpy(dtype=float)
    price_ratio = opens[grid] / opens[start_row]

    gross_nav = price_ratio.astype(float)
    net_nav = entry_factor * price_ratio
    net_nav[-1] *= exit_factor
    net_exposure = np.ones(len(grid), dtype=float)
    gross_exposure = np.ones(len(grid), dtype=float)
    net_exposure[-1] = 0.0
    gross_exposure[-1] = 0.0
    active = np.ones(len(grid), dtype=int)
    active[-1] = 0

    equity = pd.DataFrame(
        {
            "Scope": EXPLORATORY_SCOPE,
            "Strategy": BUY_HOLD_STRATEGY,
            "Open Date": dates.iloc[grid].to_numpy(),
            "Price Row": grid,
            "Open Price": opens[grid],
            "Net NAV": net_nav,
            "Gross NAV": gross_nav,
            "Net Exposure": net_exposure,
            "Gross Exposure": gross_exposure,
            "Active Sleeves": active,
        }
    )
    origin_signal_date = executable.sort_values("Signal Date").iloc[0][
        "Signal Date"
    ]
    exit_nav_before = float(entry_factor * price_ratio[-1])
    trades = pd.DataFrame(
        [
            {
                "Scope": EXPLORATORY_SCOPE,
                "Strategy": BUY_HOLD_STRATEGY,
                "Open Date": dates.iloc[start_row],
                "Price Row": start_row,
                "Sleeve": "Portfolio",
                "Event": "Initial entry",
                "Origin Signal Date": origin_signal_date,
                "Scheduled End Date": dates.iloc[final_row],
                "Old Position": 0,
                "New Position": 1,
                "Transition": "0->1",
                "Net Sleeve NAV Before": 1.0,
                "Applied Factor": entry_factor,
                "Net Sleeve NAV After": entry_factor,
                "Friction Amount": 1.0 - entry_factor,
                "Transaction Cost (bps)": float(transaction_cost_bps),
                "Slippage (bps)": float(slippage_bps),
            },
            {
                "Scope": EXPLORATORY_SCOPE,
                "Strategy": BUY_HOLD_STRATEGY,
                "Open Date": dates.iloc[final_row],
                "Price Row": final_row,
                "Sleeve": "Portfolio",
                "Event": "Terminal close",
                "Origin Signal Date": origin_signal_date,
                "Scheduled End Date": dates.iloc[final_row],
                "Old Position": 1,
                "New Position": 0,
                "Transition": "1->0",
                "Net Sleeve NAV Before": exit_nav_before,
                "Applied Factor": exit_factor,
                "Net Sleeve NAV After": exit_nav_before * exit_factor,
                "Friction Amount": exit_nav_before * (1.0 - exit_factor),
                "Transaction Cost (bps)": float(transaction_cost_bps),
                "Slippage (bps)": float(slippage_bps),
            },
        ]
    )

    intervals = final_row - start_row
    metric = _strategy_metric(
        strategy=BUY_HOLD_STRATEGY,
        family="Market-context baseline",
        executable_rows=len(executable),
        excluded_rows=int((~signal_ledger["Five-Session Executable"]).sum()),
        start_date=dates.iloc[start_row],
        end_date=dates.iloc[final_row],
        intervals=intervals,
        net_path=np.r_[1.0, net_nav],
        gross_path=np.r_[1.0, gross_nav],
        average_exposure=float(net_exposure[:-1].mean()),
        changes=2,
        terminal_all_cash=bool(active[-1] == 0),
    )
    return metric, equity, trades


def run_five_session_sandbox(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    date_column: str = "Date",
    open_column: str = "Open",
    close_column: str = "Close",
    prediction_date_column: str = "Date",
    probability_column: str | None = None,
    signal_column: str | None = None,
    probability_threshold: float = 0.50,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    prediction_strategy_name: str = "Prediction Strategy",
) -> FiveSessionSandboxResult:
    """Run the exploratory five-session prediction-to-decision layer.

    Parameters
    ----------
    prices:
        Single-asset daily data containing at least Date, Open, and Close.
        EMA8, EMA13, SMA10, and SMA20 are reused when present and otherwise
        calculated causally from Close.
    predictions:
        A possibly non-contiguous subset of signal dates.  It must contain a
        binary Signal/Prediction column, a probability column, or both.
    probability_threshold:
        Used only when a binary signal column is absent.
    transaction_cost_bps, slippage_bps:
        Non-negative one-way friction assumptions applied on actual 0->1 and
        1->0 changes.  A 1->1 continuation does not pay another fee.

    Returns
    -------
    FiveSessionSandboxResult
        Metrics, portfolio equity rows, fee-bearing trade transitions, and the
        complete signal/exclusion ledger for the prediction strategy,
        EMA8/EMA13 state, SMA10/SMA20 state, and Buy-and-Hold.
    """

    clean_prices = _prepare_prices(
        prices,
        date_column=date_column,
        open_column=open_column,
        close_column=close_column,
    )
    # The signal-ledger column is intentionally canonical even when the caller
    # uses a differently named Close column.
    if close_column != "Close":
        clean_prices["Close"] = clean_prices[close_column]

    clean_predictions = _prepare_predictions(
        predictions,
        prediction_date_column=prediction_date_column,
        probability_column=probability_column,
        signal_column=signal_column,
        probability_threshold=probability_threshold,
    )
    signal_ledger = _build_signal_ledger(
        clean_prices,
        clean_predictions,
        date_column=date_column,
    )
    if not signal_ledger["Five-Session Executable"].any():
        raise ValueError(
            "No prediction date has both moving-average history and Open[t+6]."
        )

    entry_factor, exit_factor = _entry_exit_factors(
        transaction_cost_bps,
        slippage_bps,
    )
    strategy_specs = [
        (
            prediction_strategy_name,
            "Predictive model",
            "Prediction Signal",
        ),
        (
            EMA_STRATEGY,
            "Moving-average baseline",
            "EMA8/EMA13 State Signal",
        ),
        (
            SMA_STRATEGY,
            "Moving-average baseline",
            "SMA10/SMA20 State Signal",
        ),
    ]
    metrics: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for strategy, family, strategy_signal_column in strategy_specs:
        metric, equity, trades = _simulate_sleeves(
            clean_prices,
            signal_ledger,
            strategy=strategy,
            family=family,
            signal_column=strategy_signal_column,
            date_column=date_column,
            open_column=open_column,
            entry_factor=entry_factor,
            exit_factor=exit_factor,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        metrics.append(metric)
        equity_frames.append(equity)
        trade_frames.append(trades)

    metric, equity, trades = _simulate_buy_and_hold(
        clean_prices,
        signal_ledger,
        date_column=date_column,
        open_column=open_column,
        entry_factor=entry_factor,
        exit_factor=exit_factor,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    metrics.append(metric)
    equity_frames.append(equity)
    trade_frames.append(trades)

    return FiveSessionSandboxResult(
        metrics=pd.DataFrame(metrics),
        equity_rows=pd.concat(equity_frames, ignore_index=True, sort=False),
        sleeve_trade_ledger=pd.concat(
            trade_frames,
            ignore_index=True,
            sort=False,
        ),
        signal_ledger=signal_ledger,
    )
