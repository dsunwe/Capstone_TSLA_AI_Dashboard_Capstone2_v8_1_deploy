"""Run the five-session, three-asset RQ4 downstream evaluation.

The script consumes frozen RQ3-compatible probabilities.  It does not train,
tune, select, or rethreshold a model.  Four predictive models and two moving-
average baselines use one common five-sleeve execution engine.  Full and
exposure-matched Buy-and-Hold paths provide descriptive context.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq4-five-session-matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
PREDICTIONS_PATH = ROOT / "upstream" / "rq4_model_predictions.csv"
PREDICTION_MANIFEST_PATH = (
    ROOT / "upstream" / "cross_asset_prediction_manifest.json"
)
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
FREEZE = ROOT / "freeze"
for directory in [OUTPUTS, FIGURES, FREEZE]:
    directory.mkdir(parents=True, exist_ok=True)

ASSETS = ["TSLA", "AAPL", "NVDA"]
PERIODS = ["Development OOF", "Retrospective audit"]
MODEL_STRATEGIES = [
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
]
MA_STRATEGIES = [
    "EMA8/EMA13 State",
    "SMA10/SMA20 State",
]
SLEEVE_STRATEGIES = MODEL_STRATEGIES + MA_STRATEGIES
ALL_HEADLINE_STRATEGIES = SLEEVE_STRATEGIES + ["Buy-and-Hold"]
STRATEGY_FAMILY = {
    **{model: "Predictive model" for model in MODEL_STRATEGIES},
    **{strategy: "Moving-average baseline" for strategy in MA_STRATEGIES},
    "Buy-and-Hold": "Market-context baseline",
}
THRESHOLD = 0.50
HORIZON = 5
SLEEVES = 5
TRANSACTION_COST = 0.0010
SLIPPAGE = 0.0005
ENTRY_FACTOR = 1.0 / ((1.0 + TRANSACTION_COST) * (1.0 + SLIPPAGE))
EXIT_FACTOR = (1.0 - TRANSACTION_COST) * (1.0 - SLIPPAGE)
TRADING_SESSIONS_PER_YEAR = 252


@dataclass
class StrategyPath:
    metric: dict[str, Any]
    equity: pd.DataFrame
    trades: pd.DataFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_protocol_alignment(protocol: dict[str, Any]) -> None:
    mapping = protocol["prediction_to_decision_mapping"]
    upstream = protocol["upstream_prediction_protocol"]
    frictions = protocol["frictions"]
    observed = {
        "probability_threshold": float(upstream["probability_threshold"]),
        "sleeves": int(mapping["sleeves"]),
        "initial_weight_per_sleeve": float(
            mapping["initial_weight_per_sleeve"]
        ),
        "holding_open_to_open_intervals": int(
            mapping["holding_open_to_open_intervals"]
        ),
        "transaction_cost_bps": float(frictions["transaction_cost_bps"]),
        "adverse_slippage_bps": float(frictions["adverse_slippage_bps"]),
    }
    expected = {
        "probability_threshold": THRESHOLD,
        "sleeves": SLEEVES,
        "initial_weight_per_sleeve": 1.0 / SLEEVES,
        "holding_open_to_open_intervals": HORIZON,
        "transaction_cost_bps": TRANSACTION_COST * 10_000,
        "adverse_slippage_bps": SLIPPAGE * 10_000,
    }
    for field, expected_value in expected.items():
        if not np.isclose(observed[field], expected_value):
            raise RuntimeError(
                f"Protocol/code mismatch for {field}: "
                f"{observed[field]} != {expected_value}."
            )
    if protocol["research_positioning"]["statistical_inference"].lower().find(
        "no rq4 bootstrap"
    ) < 0:
        raise RuntimeError("Protocol no longer declares the no-bootstrap scope.")
    if protocol["research_positioning"]["excluded_extension"].lower().find(
        "no static-model versus quarterly rolling-retraining"
    ) < 0:
        raise RuntimeError(
            "Protocol no longer excludes quarterly rolling retraining."
        )


def validate_frozen_inputs(protocol: dict[str, Any]) -> None:
    expected: dict[Path, str] = {}
    for asset in ASSETS:
        item = protocol["data_snapshots"][asset]
        expected[ROOT / item["file"]] = item["sha256"]
    upstream = protocol["upstream_prediction_protocol"]
    expected[ROOT / upstream["rq3_protocol_file"]] = upstream[
        "rq3_protocol_sha256"
    ]
    expected[ROOT / upstream["rq3_engine_file"]] = upstream[
        "rq3_engine_sha256"
    ]
    expected[ROOT / upstream["tsla_frozen_predictions_file"]] = upstream[
        "tsla_frozen_predictions_sha256"
    ]
    mismatches = []
    for path, expected_hash in expected.items():
        if not path.exists():
            mismatches.append(
                {"path": str(path), "expected": expected_hash, "actual": None}
            )
            continue
        actual = sha256(path)
        if actual != expected_hash:
            mismatches.append(
                {"path": str(path), "expected": expected_hash, "actual": actual}
            )
    if mismatches:
        raise RuntimeError(
            "Frozen input mismatch: " + json.dumps(mismatches, sort_keys=True)
        )

    if not PREDICTIONS_PATH.exists() or not PREDICTION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Generate and combine AAPL/NVDA predictions before running RQ4."
        )
    prediction_manifest = json.loads(
        PREDICTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if prediction_manifest["combined_prediction_sha256"] != sha256(
        PREDICTIONS_PATH
    ):
        raise RuntimeError("Combined cross-asset prediction hash is invalid.")
    expected_manifest_hashes = {
        ROOT / "upstream" / "TSLA_rq3_all_predictions.csv": prediction_manifest[
            "source_prediction_sha256"
        ]["TSLA"],
        ROOT / "upstream" / "AAPL_rq3_model_predictions.csv": prediction_manifest[
            "source_prediction_sha256"
        ]["AAPL"],
        ROOT / "upstream" / "NVDA_rq3_model_predictions.csv": prediction_manifest[
            "source_prediction_sha256"
        ]["NVDA"],
        ROOT / "upstream" / "rq3_engine.py": prediction_manifest[
            "rq3_engine_sha256"
        ],
        ROOT / "upstream" / "rq3_protocol.json": prediction_manifest[
            "rq3_protocol_sha256"
        ],
    }
    for asset in ASSETS:
        expected_manifest_hashes[
            ROOT / "data" / f"{asset}_features_common_window.csv"
        ] = prediction_manifest["data_sha256"][asset]
    for path, expected_hash in expected_manifest_hashes.items():
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(
                f"Prediction manifest source hash is invalid for {path}."
            )
    if prediction_manifest["return_metrics_used_for_training_or_selection"]:
        raise RuntimeError("Return-informed prediction generation is prohibited.")


def load_price_data(protocol: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    required = {
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA8",
        "EMA13",
        "SMA10",
        "SMA20",
    }
    common_calendar: pd.DatetimeIndex | None = None
    for asset in ASSETS:
        path = ROOT / protocol["data_snapshots"][asset]["file"]
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame = frame.sort_values("Date").reset_index(drop=True)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{asset} price data misses fields: {missing}")
        if frame["Date"].duplicated().any():
            raise RuntimeError(f"{asset} has duplicate trading dates.")
        if not frame["Date"].is_monotonic_increasing:
            raise RuntimeError(f"{asset} dates are not increasing.")
        if frame[["Open", "High", "Low", "Close"]].isna().any().any():
            raise RuntimeError(f"{asset} contains missing OHLC values.")
        if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
            raise RuntimeError(f"{asset} contains non-positive OHLC values.")
        if (frame["Volume"] < 0).any():
            raise RuntimeError(f"{asset} contains negative Volume.")
        if (
            frame["High"]
            < frame[["Open", "Close", "Low"]].max(axis=1)
        ).any() or (
            frame["Low"]
            > frame[["Open", "Close", "High"]].min(axis=1)
        ).any():
            raise RuntimeError(f"{asset} contains inconsistent OHLC ranges.")
        calendar = pd.DatetimeIndex(frame["Date"])
        if common_calendar is None:
            common_calendar = calendar
        elif not calendar.equals(common_calendar):
            raise RuntimeError("The three asset calendars are not identical.")
        frame["Price_Row"] = np.arange(len(frame), dtype=int)
        prices[asset] = frame
    return prices


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["Date", "Target_Date"],
    )
    required = {
        "Asset",
        "Date",
        "Target_Date",
        "Eligible_Row_ID",
        "Phase_Offset",
        "Actual",
        "Comparator",
        "Role",
        "Period",
        "Fold",
        "Probability",
        "Prediction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction ledger misses fields: {missing}")
    frame = frame.loc[
        frame["Asset"].isin(ASSETS)
        & frame["Period"].isin(PERIODS)
        & frame["Comparator"].isin(MODEL_STRATEGIES)
        & frame["Role"].eq("Main model")
    ].copy()
    if frame.duplicated(["Asset", "Period", "Comparator", "Date"]).any():
        raise RuntimeError("Model predictions are not unique.")
    if (
        ~np.isfinite(frame["Probability"])
        | frame["Probability"].lt(0)
        | frame["Probability"].gt(1)
    ).any():
        raise RuntimeError("Prediction probabilities must be finite in [0, 1].")
    if not np.array_equal(
        frame["Prediction"].astype(int).to_numpy(),
        (frame["Probability"].to_numpy() >= THRESHOLD).astype(int),
    ):
        raise RuntimeError("Prediction threshold does not reproduce at 0.50.")
    if not np.array_equal(
        frame["Phase_Offset"].astype(int).to_numpy(),
        frame["Eligible_Row_ID"].astype(int).to_numpy() % HORIZON,
    ):
        raise RuntimeError("Phase offsets do not equal Eligible_Row_ID modulo 5.")

    expected = {"Development OOF": 1250, "Retrospective audit": 407}
    for asset in ASSETS:
        for period in PERIODS:
            for model in MODEL_STRATEGIES:
                observed = int(
                    (
                        frame["Asset"].eq(asset)
                        & frame["Period"].eq(period)
                        & frame["Comparator"].eq(model)
                    ).sum()
                )
                if observed != expected[period]:
                    raise RuntimeError(
                        f"{asset} {period} {model}: {observed} rows."
                    )
    if len(frame) != 19_884:
        raise RuntimeError(f"Expected 19,884 model rows, observed {len(frame)}.")
    return frame


def build_signal_ledger(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metadata_columns = [
        "Date",
        "Target_Date",
        "Eligible_Row_ID",
        "Phase_Offset",
        "Actual",
        "Period",
        "Fold",
    ]
    for asset in ASSETS:
        asset_prices = prices[asset]
        calendar = pd.DatetimeIndex(asset_prices["Date"])
        for period in PERIODS:
            current = predictions.loc[
                predictions["Asset"].eq(asset)
                & predictions["Period"].eq(period)
            ].copy()
            reference = (
                current.loc[
                    current["Comparator"].eq(MODEL_STRATEGIES[0]),
                    metadata_columns,
                ]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            if not reference["Date"].is_monotonic_increasing:
                raise RuntimeError(f"{asset} {period} signals are not ordered.")
            if not np.all(
                np.diff(reference["Eligible_Row_ID"].to_numpy(dtype=int)) == 1
            ):
                raise RuntimeError(
                    f"{asset} {period} signal origins are not consecutive."
                )
            for column in [
                "Target_Date",
                "Eligible_Row_ID",
                "Phase_Offset",
                "Actual",
                "Fold",
            ]:
                if (
                    current.groupby("Date")[column].nunique(dropna=False).max()
                    != 1
                ):
                    raise RuntimeError(
                        f"{asset} {period} models disagree on {column}."
                    )

            signal_locations = calendar.get_indexer(
                pd.DatetimeIndex(reference["Date"])
            )
            if (signal_locations < 0).any():
                raise RuntimeError(f"{asset} has signal dates absent from prices.")
            if (signal_locations < 4).any() or (
                signal_locations + HORIZON >= len(asset_prices)
            ).any():
                raise RuntimeError(f"{asset} signal lacks a complete RQ3 label window.")
            close_values = asset_prices["Close"].to_numpy(dtype=float)
            rebuilt_actual = np.asarray(
                [
                    int(
                        close_values[position + 1 : position + HORIZON + 1].mean()
                        > close_values[position - HORIZON + 1 : position + 1].mean()
                    )
                    for position in signal_locations
                ],
                dtype=int,
            )
            rebuilt_target_dates = asset_prices.iloc[
                signal_locations + HORIZON
            ]["Date"].reset_index(drop=True)
            if not np.array_equal(
                rebuilt_actual,
                reference["Actual"].astype(int).to_numpy(),
            ):
                raise RuntimeError(
                    f"{asset} {period} Actual labels do not reproduce from Close."
                )
            if not pd.DatetimeIndex(rebuilt_target_dates).equals(
                pd.DatetimeIndex(reference["Target_Date"])
            ):
                raise RuntimeError(
                    f"{asset} {period} Target_Date does not equal trading t+5."
                )
            reference["Recomputed Actual"] = rebuilt_actual
            reference["Signal Price Row"] = signal_locations
            reference["Execution Price Row"] = signal_locations + 1
            reference["End Price Row"] = signal_locations + 6
            reference["Five-Session Executable"] = (
                reference["End Price Row"] < len(asset_prices)
            )
            reference["Exclusion Reason"] = np.where(
                reference["Five-Session Executable"],
                "",
                "missing Open[t+6]",
            )
            reference["Asset"] = asset

            for model in MODEL_STRATEGIES:
                model_rows = (
                    current.loc[
                        current["Comparator"].eq(model),
                        ["Date", "Probability", "Prediction"],
                    ]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                if not pd.DatetimeIndex(model_rows["Date"]).equals(
                    pd.DatetimeIndex(reference["Date"])
                ):
                    raise RuntimeError(
                        f"{asset} {period} {model} has a different date grid."
                    )
                reference[f"{model} Probability"] = model_rows[
                    "Probability"
                ].to_numpy(dtype=float)
                reference[f"{model} Signal"] = model_rows[
                    "Prediction"
                ].to_numpy(dtype=int)

            signal_price_rows = asset_prices.iloc[signal_locations].reset_index(
                drop=True
            )
            if signal_price_rows[
                ["Close", "EMA8", "EMA13", "SMA10", "SMA20"]
            ].isna().any().any():
                raise RuntimeError(
                    f"{asset} {period} has missing signal-date moving averages."
                )
            reference["Signal Close"] = signal_price_rows["Close"].to_numpy()
            reference["EMA8"] = signal_price_rows["EMA8"].to_numpy()
            reference["EMA13"] = signal_price_rows["EMA13"].to_numpy()
            reference["SMA10"] = signal_price_rows["SMA10"].to_numpy()
            reference["SMA20"] = signal_price_rows["SMA20"].to_numpy()
            reference["EMA8/EMA13 State Signal"] = (
                reference["EMA8"] > reference["EMA13"]
            ).astype(int)
            reference["SMA10/SMA20 State Signal"] = (
                reference["SMA10"] > reference["SMA20"]
            ).astype(int)

            executable = reference["Five-Session Executable"]
            reference.loc[executable, "Execution Date"] = asset_prices.iloc[
                reference.loc[executable, "Execution Price Row"].astype(int)
            ]["Date"].to_numpy()
            reference.loc[executable, "Scheduled End Date"] = asset_prices.iloc[
                reference.loc[executable, "End Price Row"].astype(int)
            ]["Date"].to_numpy()
            rows.append(reference)

    ledger = pd.concat(rows, ignore_index=True)
    ledger["Date"] = pd.to_datetime(ledger["Date"])
    ledger["Execution Date"] = pd.to_datetime(ledger["Execution Date"])
    ledger["Scheduled End Date"] = pd.to_datetime(
        ledger["Scheduled End Date"]
    )
    executable_counts = (
        ledger.groupby(["Asset", "Period"])["Five-Session Executable"].sum()
    )
    for asset in ASSETS:
        if int(executable_counts.loc[(asset, "Development OOF")]) != 1250:
            raise RuntimeError(f"{asset} Development executable count changed.")
        if int(executable_counts.loc[(asset, "Retrospective audit")]) != 406:
            raise RuntimeError(f"{asset} audit executable count changed.")
    excluded = ledger.loc[~ledger["Five-Session Executable"]]
    if len(excluded) != 3 or set(excluded["Date"]) != {
        pd.Timestamp("2026-07-15")
    }:
        raise RuntimeError("Unexpected five-session terminal exclusions.")
    return ledger


def maximum_drawdown(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    running_peak = np.maximum.accumulate(values)
    drawdown = values / running_peak - 1.0
    return float(drawdown.min())


def annualised_return(final_nav: float, intervals: int) -> float:
    if final_nav <= 0 or intervals <= 0:
        return np.nan
    return float(final_nav ** (TRADING_SESSIONS_PER_YEAR / intervals) - 1.0)


def simulate_five_sleeves(
    asset: str,
    period: str,
    strategy: str,
    signal_rows: pd.DataFrame,
    asset_prices: pd.DataFrame,
    signal_column: str,
) -> StrategyPath:
    current = signal_rows.loc[
        signal_rows["Asset"].eq(asset)
        & signal_rows["Period"].eq(period)
        & signal_rows["Five-Session Executable"]
    ].sort_values("Date")
    if current.empty:
        raise RuntimeError(f"No executable rows for {asset} {period}.")
    signal_values = current[signal_column].astype(int)
    if not signal_values.isin([0, 1]).all():
        raise RuntimeError(f"{strategy} has non-binary signals.")

    schedule: dict[int, dict[int, int]] = {phase: {} for phase in range(SLEEVES)}
    phase_last_end: dict[int, int] = {}
    for (_, row), desired in zip(current.iterrows(), signal_values):
        phase = int(row["Phase_Offset"])
        execution_row = int(row["Execution Price Row"])
        end_row = int(row["End Price Row"])
        schedule[phase][execution_row] = int(desired)
        phase_last_end[phase] = end_row
    if set(phase_last_end) != set(range(SLEEVES)):
        raise RuntimeError(f"{asset} {period} does not cover all five phases.")
    for phase, end_row in phase_last_end.items():
        if end_row in schedule[phase]:
            raise RuntimeError("Terminal close collides with a later phase signal.")
        schedule[phase][end_row] = 0

    start_row = int(current["Execution Price Row"].min())
    final_row = int(max(phase_last_end.values()))
    grid = np.arange(start_row, final_row + 1, dtype=int)
    open_prices = asset_prices["Open"].to_numpy(dtype=float)
    dates = pd.to_datetime(asset_prices["Date"])
    if final_row >= len(asset_prices):
        raise RuntimeError("Simulation end exceeds the price snapshot.")

    net_nav = np.full(SLEEVES, 1.0 / SLEEVES, dtype=float)
    gross_nav = np.full(SLEEVES, 1.0 / SLEEVES, dtype=float)
    positions = np.zeros(SLEEVES, dtype=int)
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    changes = 0

    for grid_index, price_row in enumerate(grid):
        for phase in range(SLEEVES):
            if price_row not in schedule[phase]:
                continue
            old = int(positions[phase])
            new = int(schedule[phase][price_row])
            if old == new:
                continue
            net_before = float(net_nav[phase])
            factor = ENTRY_FACTOR if (old, new) == (0, 1) else EXIT_FACTOR
            net_nav[phase] *= factor
            positions[phase] = new
            changes += 1
            trade_rows.append(
                {
                    "Asset": asset,
                    "Period": period,
                    "Strategy": strategy,
                    "Open Date": dates.iloc[price_row],
                    "Phase": phase,
                    "Old Position": old,
                    "New Position": new,
                    "Transition": f"{old}->{new}",
                    "Net Sleeve NAV Before": net_before,
                    "Applied Factor": factor,
                    "Net Sleeve NAV After": float(net_nav[phase]),
                    "Cost Amount": float(net_before - net_nav[phase]),
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
                "Asset": asset,
                "Period": period,
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
                    f"Phase {phase} Net NAV": float(net_nav[phase])
                    for phase in range(SLEEVES)
                },
                **{
                    f"Phase {phase} Position": int(positions[phase])
                    for phase in range(SLEEVES)
                },
            }
        )
        if grid_index == len(grid) - 1:
            break
        market_factor = open_prices[price_row + 1] / open_prices[price_row]
        net_nav = net_nav * np.where(positions == 1, market_factor, 1.0)
        gross_nav = gross_nav * np.where(positions == 1, market_factor, 1.0)

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    if equity.iloc[-1]["Active Sleeves"] != 0:
        raise RuntimeError(f"{asset} {period} {strategy} did not end in cash.")
    if not np.isclose(
        equity.iloc[-1][[f"Phase {phase} Net NAV" for phase in range(SLEEVES)]]
        .astype(float)
        .sum(),
        equity.iloc[-1]["Net NAV"],
        atol=1e-12,
    ):
        raise RuntimeError("Phase NAVs do not reconcile to portfolio NAV.")

    net_path = np.r_[1.0, equity["Net NAV"].to_numpy(dtype=float)]
    gross_path = np.r_[1.0, equity["Gross NAV"].to_numpy(dtype=float)]
    net_returns = (
        equity["Net NAV"].astype(float).pct_change().dropna().to_numpy()
    )
    intervals = int(final_row - start_row)
    interval_exposure = equity.iloc[:-1]["Net Exposure"].to_numpy(dtype=float)
    metric = {
        "Asset": asset,
        "Period": period,
        "Strategy": strategy,
        "Family": STRATEGY_FAMILY[strategy],
        "Signal Rows": len(current),
        "Path Start": dates.iloc[start_row],
        "Path End": dates.iloc[final_row],
        "Market Intervals": intervals,
        "Gross Total Return": float(gross_path[-1] - 1.0),
        "Net Total Return": float(net_path[-1] - 1.0),
        "Cost Drag": float(gross_path[-1] - net_path[-1]),
        "Annualised Net Return": annualised_return(net_path[-1], intervals),
        "Annualised Volatility": float(
            np.std(net_returns, ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)
        ),
        "Maximum Drawdown": maximum_drawdown(net_path),
        "Average Exposure": float(np.mean(interval_exposure)),
        "One-way Sleeve Changes": changes,
        "Terminal All Cash": bool(equity.iloc[-1]["Active Sleeves"] == 0),
    }
    return StrategyPath(metric=metric, equity=equity, trades=trades)


def buy_hold_path(
    asset: str,
    period: str,
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
    alpha: float,
    label: str,
) -> dict[str, Any]:
    opens = prices["Open"].to_numpy(dtype=float)
    dates = pd.to_datetime(prices["Date"])
    grid = np.arange(start_row, end_row + 1, dtype=int)
    price_ratio = opens[grid] / opens[start_row]
    net_stock = alpha * ENTRY_FACTOR * price_ratio
    gross_stock = alpha * price_ratio
    cash = 1.0 - alpha
    net_nav_pre_exit = cash + net_stock
    gross_nav_pre_exit = cash + gross_stock
    net_exposure = net_stock[:-1] / net_nav_pre_exit[:-1]
    gross_exposure = gross_stock[:-1] / gross_nav_pre_exit[:-1]
    net_nav = net_nav_pre_exit.copy()
    gross_nav = gross_nav_pre_exit.copy()
    net_nav[-1] = cash + net_stock[-1] * EXIT_FACTOR
    # The gross path has no entry or terminal cost.
    net_path = np.r_[1.0, net_nav]
    gross_path = np.r_[1.0, gross_nav]
    net_returns = pd.Series(net_nav).pct_change().dropna().to_numpy()
    intervals = end_row - start_row
    return {
        "Asset": asset,
        "Period": period,
        "Strategy": label,
        "Initial Stock Allocation": float(alpha),
        "Path Start": dates.iloc[start_row],
        "Path End": dates.iloc[end_row],
        "Market Intervals": int(intervals),
        "Gross Total Return": float(gross_path[-1] - 1.0),
        "Net Total Return": float(net_path[-1] - 1.0),
        "Cost Drag": float(gross_path[-1] - net_path[-1]),
        "Annualised Net Return": annualised_return(net_path[-1], intervals),
        "Annualised Volatility": float(
            np.std(net_returns, ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)
        ),
        "Maximum Drawdown": maximum_drawdown(net_path),
        "Average Exposure": float(np.mean(net_exposure)),
        "Gross Average Exposure": float(np.mean(gross_exposure)),
        "Net NAV": net_nav,
        "Gross NAV": gross_nav,
        "Net Exposure Path": np.r_[net_exposure, 0.0],
        "Gross Exposure Path": np.r_[gross_exposure, 0.0],
        "Open Dates": dates.iloc[grid].to_numpy(),
    }


def solve_exposure_matched_alpha(
    target_exposure: float,
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
) -> float:
    if target_exposure <= 0:
        return 0.0
    if target_exposure >= 1:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        result = buy_hold_path(
            "MATCH",
            "MATCH",
            prices,
            start_row,
            end_row,
            midpoint,
            "Exposure-Matched Buy-and-Hold",
        )
        if result["Average Exposure"] < target_exposure:
            lower = midpoint
        else:
            upper = midpoint
    alpha = (lower + upper) / 2.0
    result = buy_hold_path(
        "MATCH",
        "MATCH",
        prices,
        start_row,
        end_row,
        alpha,
        "Exposure-Matched Buy-and-Hold",
    )
    if not np.isclose(
        result["Average Exposure"],
        target_exposure,
        atol=1e-10,
    ):
        raise RuntimeError("Exposure-matched alpha solver did not converge.")
    return float(alpha)


def constant_exposure_passive_path(
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
    exposure: float,
) -> dict[str, Any]:
    """Synthetic fixed-exposure reference with conservative friction treatment.

    The path compounds ``1 + exposure * market_return`` at every open-to-open
    interval.  It is a comparison device rather than literal Buy-and-Hold.
    Entry and terminal costs are charged to the exposed fraction; intermediate
    rebalancing costs are deliberately omitted, which favours the benchmark.
    """

    if not 0.0 <= exposure <= 1.0:
        raise ValueError("Constant exposure must be in [0, 1].")
    opens = prices["Open"].to_numpy(dtype=float)
    market_returns = (
        opens[start_row + 1 : end_row + 1]
        / opens[start_row:end_row]
        - 1.0
    )
    gross_interval_factors = 1.0 + exposure * market_returns
    gross_nav = np.r_[1.0, np.cumprod(gross_interval_factors)]
    entry_portfolio_factor = (1.0 - exposure) + exposure * ENTRY_FACTOR
    exit_portfolio_factor = (1.0 - exposure) + exposure * EXIT_FACTOR
    net_nav = gross_nav * entry_portfolio_factor
    net_nav[-1] *= exit_portfolio_factor
    intervals = end_row - start_row
    return {
        "Exposure": float(exposure),
        "Gross Total Return": float(gross_nav[-1] - 1.0),
        "Net Total Return": float(net_nav[-1] - 1.0),
        "Maximum Drawdown": maximum_drawdown(np.r_[1.0, net_nav]),
        "Annualised Net Return": annualised_return(net_nav[-1], intervals),
        "Entry Portfolio Factor": float(entry_portfolio_factor),
        "Exit Portfolio Factor": float(exit_portfolio_factor),
        "Intermediate Rebalancing Costs Included": False,
    }


def add_heatmap(
    table: pd.DataFrame,
    period: str,
    value_column: str,
    title: str,
    filename: str,
    unit: str = "%",
) -> None:
    value_suffix = unit if unit == "%" else f" {unit}"
    pivot = (
        table.loc[
            table["Period"].eq(period)
            & table["Strategy"].isin(MODEL_STRATEGIES)
        ]
        .pivot(index="Asset", columns="Strategy", values=value_column)
        .reindex(index=ASSETS, columns=MODEL_STRATEGIES)
    )
    values = pivot.to_numpy(dtype=float) * 100.0
    maximum = float(np.nanmax(np.abs(values)))
    maximum = max(maximum, 1.0)
    figure, axis = plt.subplots(figsize=(11, 4.6))
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
        aspect="auto",
    )
    axis.set_xticks(range(len(MODEL_STRATEGIES)))
    axis.set_xticklabels(
        ["Lagged LR", "Random Forest", "XGBoost", "Compact LSTM"],
        rotation=15,
        ha="right",
    )
    axis.set_yticks(range(len(ASSETS)))
    axis.set_yticklabels(ASSETS)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:,.1f}{value_suffix}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    axis.set_title(title)
    axis.set_xlabel("Predictive model")
    axis.set_ylabel("Asset")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(f"{value_column} ({unit})")
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=190, bbox_inches="tight")
    plt.close(figure)


def add_development_comparison_chart(metrics: pd.DataFrame) -> None:
    current = metrics.loc[
        metrics["Period"].eq("Development OOF")
        & metrics["Strategy"].isin(
            MODEL_STRATEGIES
            + ["EMA8/EMA13 State", "SMA10/SMA20 State", "Buy-and-Hold"]
        )
    ].copy()
    labels = [
        "Lagged LR",
        "Random Forest",
        "XGBoost",
        "Compact LSTM",
        "EMA8/13",
        "SMA10/20",
        "Buy-and-Hold",
    ]
    strategies = MODEL_STRATEGIES + MA_STRATEGIES + ["Buy-and-Hold"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = [
        "#2563eb",
        "#0ea5e9",
        "#7c3aed",
        "#db2777",
        "#64748b",
        "#475569",
        "#111827",
    ]
    for axis, asset in zip(axes, ASSETS):
        asset_rows = current.loc[current["Asset"].eq(asset)].set_index(
            "Strategy"
        )
        values = [
            float(asset_rows.loc[strategy, "Net Total Return"]) * 100
            for strategy in strategies
        ]
        bars = axis.bar(range(len(values)), values, color=colors)
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        axis.set_title(asset)
        axis.set_ylabel("Net total return (%)")
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,.0f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    figure.suptitle(
        "Development OOF five-session strategy returns after 15 bps one-way friction"
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES / "rq4_development_strategy_comparison.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    protocol = load_protocol()
    validate_protocol_alignment(protocol)
    validate_frozen_inputs(protocol)
    prices = load_price_data(protocol)
    predictions = load_predictions()
    signal_ledger = build_signal_ledger(predictions, prices)
    signal_ledger.to_csv(OUTPUTS / "rq4_signal_ledger.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    matched_rows: list[dict[str, Any]] = []

    signal_columns = {
        **{model: f"{model} Signal" for model in MODEL_STRATEGIES},
        "EMA8/EMA13 State": "EMA8/EMA13 State Signal",
        "SMA10/SMA20 State": "SMA10/SMA20 State Signal",
    }

    for asset in ASSETS:
        for period in PERIODS:
            subset = signal_ledger.loc[
                signal_ledger["Asset"].eq(asset)
                & signal_ledger["Period"].eq(period)
                & signal_ledger["Five-Session Executable"]
            ]
            start_row = int(subset["Execution Price Row"].min())
            end_row = int(subset["End Price Row"].max())
            strategy_paths: dict[str, StrategyPath] = {}
            for strategy in SLEEVE_STRATEGIES:
                result = simulate_five_sleeves(
                    asset,
                    period,
                    strategy,
                    signal_ledger,
                    prices[asset],
                    signal_columns[strategy],
                )
                strategy_paths[strategy] = result
                metric_rows.append(result.metric)
                equity_frames.append(result.equity)
                if not result.trades.empty:
                    trade_frames.append(result.trades)

            buy_hold = buy_hold_path(
                asset,
                period,
                prices[asset],
                start_row,
                end_row,
                1.0,
                "Buy-and-Hold",
            )
            metric_rows.append(
                {
                    "Asset": asset,
                    "Period": period,
                    "Strategy": "Buy-and-Hold",
                    "Family": STRATEGY_FAMILY["Buy-and-Hold"],
                    "Signal Rows": len(subset),
                    "Path Start": buy_hold["Path Start"],
                    "Path End": buy_hold["Path End"],
                    "Market Intervals": buy_hold["Market Intervals"],
                    "Gross Total Return": buy_hold["Gross Total Return"],
                    "Net Total Return": buy_hold["Net Total Return"],
                    "Cost Drag": buy_hold["Cost Drag"],
                    "Annualised Net Return": buy_hold["Annualised Net Return"],
                    "Annualised Volatility": buy_hold["Annualised Volatility"],
                    "Maximum Drawdown": buy_hold["Maximum Drawdown"],
                    "Average Exposure": buy_hold["Average Exposure"],
                    "One-way Sleeve Changes": 2,
                    "Terminal All Cash": True,
                }
            )
            equity_frames.append(
                pd.DataFrame(
                    {
                        "Asset": asset,
                        "Period": period,
                        "Strategy": "Buy-and-Hold",
                        "Open Date": buy_hold["Open Dates"],
                        "Price Row": np.arange(start_row, end_row + 1),
                        "Open Price": prices[asset].iloc[
                            start_row : end_row + 1
                        ]["Open"].to_numpy(),
                        "Net NAV": buy_hold["Net NAV"],
                        "Gross NAV": buy_hold["Gross NAV"],
                        "Net Exposure": buy_hold["Net Exposure Path"],
                        "Gross Exposure": buy_hold["Gross Exposure Path"],
                        "Active Sleeves": np.r_[
                            np.ones(end_row - start_row, dtype=int),
                            0,
                        ],
                    }
                )
            )

            for model in MODEL_STRATEGIES:
                model_metric = strategy_paths[model].metric
                target_exposure = float(model_metric["Average Exposure"])
                alpha = solve_exposure_matched_alpha(
                    target_exposure,
                    prices[asset],
                    start_row,
                    end_row,
                )
                matched = buy_hold_path(
                    asset,
                    period,
                    prices[asset],
                    start_row,
                    end_row,
                    alpha,
                    f"Exposure-Matched Buy-and-Hold ({model})",
                )
                constant_exposure = constant_exposure_passive_path(
                    prices[asset],
                    start_row,
                    end_row,
                    target_exposure,
                )
                matched_rows.append(
                    {
                        "Asset": asset,
                        "Period": period,
                        "Model": model,
                        "Model Average Exposure": target_exposure,
                        "Matched Initial Stock Allocation": alpha,
                        "Matched Average Exposure": matched["Average Exposure"],
                        "Model Gross Total Return": model_metric[
                            "Gross Total Return"
                        ],
                        "Matched Gross Total Return": matched[
                            "Gross Total Return"
                        ],
                        "Gross Return Difference vs Static Matched B&H": model_metric[
                            "Gross Total Return"
                        ]
                        - matched["Gross Total Return"],
                        "Model Net Total Return": model_metric[
                            "Net Total Return"
                        ],
                        "Matched Net Total Return": matched["Net Total Return"],
                        "Net Return Difference vs Static Matched B&H": model_metric[
                            "Net Total Return"
                        ]
                        - matched["Net Total Return"],
                        "Matched Maximum Drawdown": matched[
                            "Maximum Drawdown"
                        ],
                        "Constant-Exposure Passive Gross Total Return": constant_exposure[
                            "Gross Total Return"
                        ],
                        "Constant-Exposure Passive Net Total Return": constant_exposure[
                            "Net Total Return"
                        ],
                        "Gross Timing Value vs Constant-Exposure Passive": model_metric[
                            "Gross Total Return"
                        ]
                        - constant_exposure["Gross Total Return"],
                        "Net Timing Value vs Constant-Exposure Passive": model_metric[
                            "Net Total Return"
                        ]
                        - constant_exposure["Net Total Return"],
                        "Constant-Exposure Passive Maximum Drawdown": constant_exposure[
                            "Maximum Drawdown"
                        ],
                        "Constant-Exposure Intermediate Rebalancing Costs Included": constant_exposure[
                            "Intermediate Rebalancing Costs Included"
                        ],
                        "Path Start": matched["Path Start"],
                        "Path End": matched["Path End"],
                        "Market Intervals": matched["Market Intervals"],
                        "Interpretation": "Ex-post descriptive exposure match; not used for model tuning or selection",
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics["Path Start"] = pd.to_datetime(metrics["Path Start"])
    metrics["Path End"] = pd.to_datetime(metrics["Path End"])
    metrics["Period Order"] = pd.Categorical(
        metrics["Period"], categories=PERIODS, ordered=True
    )
    metrics["Asset Order"] = pd.Categorical(
        metrics["Asset"], categories=ASSETS, ordered=True
    )
    metrics["Strategy Order"] = metrics["Strategy"].map(
        {strategy: index for index, strategy in enumerate(ALL_HEADLINE_STRATEGIES)}
    )
    metrics = (
        metrics.sort_values(
            ["Period Order", "Asset Order", "Strategy Order"]
        )
        .drop(columns=["Period Order", "Asset Order", "Strategy Order"])
        .reset_index(drop=True)
    )
    matched = pd.DataFrame(matched_rows)
    equity = pd.concat(equity_frames, ignore_index=True)
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    )

    model_results = metrics.loc[
        metrics["Strategy"].isin(MODEL_STRATEGIES)
    ].merge(
        matched[
            [
                "Asset",
                "Period",
                "Model",
                "Matched Initial Stock Allocation",
                "Matched Net Total Return",
                "Net Return Difference vs Static Matched B&H",
                "Constant-Exposure Passive Net Total Return",
                "Net Timing Value vs Constant-Exposure Passive",
            ]
        ],
        left_on=["Asset", "Period", "Strategy"],
        right_on=["Asset", "Period", "Model"],
        how="left",
        validate="one_to_one",
    )
    model_results["Return Rank within Asset-Period"] = (
        model_results.groupby(["Asset", "Period"])["Net Total Return"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    metrics.to_csv(OUTPUTS / "rq4_headline_metrics.csv", index=False)
    matched.to_csv(
        OUTPUTS / "rq4_exposure_matched_buy_hold.csv",
        index=False,
    )
    model_results.to_csv(
        OUTPUTS / "rq4_model_results.csv",
        index=False,
    )
    equity.to_csv(OUTPUTS / "rq4_equity_curves.csv", index=False)
    trades.to_csv(OUTPUTS / "rq4_sleeve_trade_ledger.csv", index=False)

    add_heatmap(
        metrics,
        "Development OOF",
        "Net Total Return",
        "Development OOF model net returns: five-session implementation",
        "rq4_development_model_returns.png",
    )
    add_heatmap(
        metrics,
        "Retrospective audit",
        "Net Total Return",
        "Later-period model net returns: five-session implementation",
        "rq4_later_period_model_returns.png",
    )
    add_heatmap(
        model_results.rename(
            columns={
                "Net Timing Value vs Constant-Exposure Passive": "Timing Value"
            }
        ),
        "Development OOF",
        "Timing Value",
        "Development OOF timing value versus constant-exposure passive reference "
        "(percentage points)",
        "rq4_development_timing_value.png",
        unit="pp",
    )
    add_development_comparison_chart(metrics)

    metadata = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "run_script_sha256": sha256(Path(__file__)),
        "prediction_manifest_sha256": sha256(PREDICTION_MANIFEST_PATH),
        "combined_predictions_sha256": sha256(PREDICTIONS_PATH),
        "assets": ASSETS,
        "models": MODEL_STRATEGIES,
        "main_moving_average_baselines": MA_STRATEGIES,
        "raw_model_prediction_rows": len(predictions),
        "asset_signal_rows": len(signal_ledger),
        "development_executable_signal_rows_per_asset": 1250,
        "audit_raw_signal_rows_per_asset": 407,
        "audit_executable_signal_rows_per_asset": 406,
        "excluded_terminal_signals": signal_ledger.loc[
            ~signal_ledger["Five-Session Executable"],
            ["Asset", "Period", "Date", "Exclusion Reason"],
        ].astype(str).to_dict(orient="records"),
        "five_self_financing_sleeves": True,
        "daily_cross_sleeve_rebalancing": False,
        "entry_factor": ENTRY_FACTOR,
        "exit_factor": EXIT_FACTOR,
        "bootstrap_used": False,
        "quarterly_rolling_retraining_used": False,
        "return_metrics_used_for_model_training_or_selection": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (OUTPUTS / "rq4_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    formal_files = [
        PROTOCOL_PATH,
        Path(__file__),
        PREDICTIONS_PATH,
        PREDICTION_MANIFEST_PATH,
        *sorted((ROOT / "data").glob("*.csv")),
        *sorted(OUTPUTS.glob("rq4_*")),
        *sorted(FIGURES.glob("rq4_*")),
    ]
    result_manifest = {
        "protocol_id": protocol["protocol_id"],
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in formal_files
            if path.is_file()
        ],
    }
    (FREEZE / "rq4_result_manifest.json").write_text(
        json.dumps(result_manifest, indent=2),
        encoding="utf-8",
    )
    print(
        "RQ4 complete: "
        f"{len(metrics)} strategy metrics, {len(matched)} exposure matches.",
        flush=True,
    )


if __name__ == "__main__":
    main()
