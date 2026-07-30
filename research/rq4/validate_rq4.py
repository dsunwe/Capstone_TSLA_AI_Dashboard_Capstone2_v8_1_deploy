"""Independently validate the saved five-session, cross-asset RQ4 outputs.

This validator deliberately does not import ``run_rq4_experiment``. It reads
the frozen price snapshots and combined prediction ledger, reconstructs the
RQ3 target and five-session signal mapping, and then runs a second,
stand-alone implementation of the five self-financing sleeves, fixed
frictions, Buy-and-Hold, and exposure-matched Buy-and-Hold calculations.

The default mode is read-only. Pass ``--write-artifacts`` after an intentional
formal recomputation to replace:

* outputs/rq4_validation_checks.csv
* outputs/rq4_validation_report.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocol.json"
RUN_SCRIPT_PATH = ROOT / "run_rq4_experiment.py"
PREDICTIONS_PATH = ROOT / "upstream" / "rq4_model_predictions.csv"
PREDICTION_MANIFEST_PATH = (
    ROOT / "upstream" / "cross_asset_prediction_manifest.json"
)
TSLA_FROZEN_PREDICTIONS_PATH = (
    ROOT / "upstream" / "TSLA_rq3_all_predictions.csv"
)
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

SIGNAL_LEDGER_PATH = OUTPUTS / "rq4_signal_ledger.csv"
HEADLINE_METRICS_PATH = OUTPUTS / "rq4_headline_metrics.csv"
MODEL_RESULTS_PATH = OUTPUTS / "rq4_model_results.csv"
EQUITY_PATH = OUTPUTS / "rq4_equity_curves.csv"
TRADES_PATH = OUTPUTS / "rq4_sleeve_trade_ledger.csv"
MATCHED_PATH = OUTPUTS / "rq4_exposure_matched_buy_hold.csv"
METADATA_PATH = OUTPUTS / "rq4_experiment_metadata.json"
CHECKS_PATH = OUTPUTS / "rq4_validation_checks.csv"
REPORT_PATH = OUTPUTS / "rq4_validation_report.json"
RESULT_MANIFEST_PATH = ROOT / "freeze" / "rq4_result_manifest.json"

ASSETS = ["TSLA", "AAPL", "NVDA"]
PERIODS = ["Development OOF", "Retrospective audit"]
MODELS = [
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
]
MOVING_AVERAGES = [
    "EMA8/EMA13 State",
    "SMA10/SMA20 State",
]
SLEEVE_STRATEGIES = MODELS + MOVING_AVERAGES
HEADLINE_STRATEGIES = SLEEVE_STRATEGIES + ["Buy-and-Hold"]
FAMILY = {
    **{model: "Predictive model" for model in MODELS},
    **{
        baseline: "Moving-average baseline"
        for baseline in MOVING_AVERAGES
    },
    "Buy-and-Hold": "Market-context baseline",
}

THRESHOLD = 0.50
HORIZON = 5
SLEEVE_COUNT = 5
TRANSACTION_COST = 0.0010
SLIPPAGE = 0.0005
ENTRY_FACTOR = 1.0 / (
    (1.0 + TRANSACTION_COST) * (1.0 + SLIPPAGE)
)
EXIT_FACTOR = (1.0 - TRANSACTION_COST) * (1.0 - SLIPPAGE)
SESSIONS_PER_YEAR = 252
ABSOLUTE_TOLERANCE = 5e-12

EXPECTED_FIGURES = [
    "rq4_development_model_returns.png",
    "rq4_later_period_model_returns.png",
    "rq4_development_timing_value.png",
    "rq4_development_strategy_comparison.png",
]


@dataclass
class IndependentPath:
    metric: dict[str, Any]
    equity: pd.DataFrame
    trades: pd.DataFrame


class ValidationRecorder:
    """Collect concise, independently evidenced validation checks."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] = {}

    def add(
        self,
        check_id: str,
        category: str,
        description: str,
        passed: bool,
        expected: Any,
        observed: Any,
        details: str = "",
        severity: str = "Critical",
    ) -> None:
        self.rows.append(
            {
                "Check ID": check_id,
                "Category": category,
                "Description": description,
                "Status": "Pass" if bool(passed) else "Fail",
                "Severity": severity,
                "Expected": serialise_cell(expected),
                "Observed": serialise_cell(observed),
                "Details": details,
            }
        )


def serialise_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refresh_result_manifest() -> None:
    """Refresh the existing RQ4 formal-file manifest after explicit writes."""

    payload = json.loads(RESULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if not files:
        raise RuntimeError("RQ4 result manifest does not declare any files.")
    refreshed = []
    for entry in files:
        relative = Path(entry["path"])
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"RQ4 result manifest file is missing: {relative.as_posix()}"
            )
        refreshed.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    payload["file_count"] = len(refreshed)
    payload["files"] = refreshed
    RESULT_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def verify_result_manifest() -> tuple[bool, int]:
    """Return whether every file declared by the RQ4 manifest still matches."""

    payload = json.loads(RESULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    matches = [
        (ROOT / entry["path"]).is_file()
        and (ROOT / entry["path"]).stat().st_size == int(entry["bytes"])
        and sha256(ROOT / entry["path"]) == entry["sha256"]
        for entry in files
    ]
    return bool(files) and all(matches), len(files)


def maximum_drawdown(values: np.ndarray) -> float:
    path = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(path)
    return float(np.min(path / peak - 1.0))


def annualised_return(final_nav: float, intervals: int) -> float:
    if final_nav <= 0.0 or intervals <= 0:
        return float("nan")
    return float(final_nav ** (SESSIONS_PER_YEAR / intervals) - 1.0)


def numeric_difference(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
) -> tuple[bool, float]:
    actual = np.asarray(left, dtype=float)
    expected = np.asarray(right, dtype=float)
    if actual.shape != expected.shape:
        return False, float("inf")
    finite_match = np.array_equal(np.isnan(actual), np.isnan(expected))
    if not finite_match:
        return False, float("inf")
    valid = ~(np.isnan(actual) | np.isnan(expected))
    if not valid.any():
        return True, 0.0
    difference = float(np.max(np.abs(actual[valid] - expected[valid])))
    return bool(
        np.allclose(
            actual[valid],
            expected[valid],
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
    ), difference


def normalise_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def read_production_numeric_constants() -> dict[str, int | float]:
    """Read literal numeric constants from the production source via AST.

    Parsing the source keeps this validator independent: the production
    module is never imported or executed.
    """

    requested = {
        "THRESHOLD",
        "HORIZON",
        "SLEEVES",
        "TRANSACTION_COST",
        "SLIPPAGE",
        "TRADING_SESSIONS_PER_YEAR",
    }
    tree = ast.parse(
        RUN_SCRIPT_PATH.read_text(encoding="utf-8"),
        filename=str(RUN_SCRIPT_PATH),
    )
    observed: dict[str, int | float] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        else:
            targets = [node.target]
            value_node = node.value
        if value_node is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in requested:
                continue
            value = ast.literal_eval(value_node)
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Production constant {target.id} is not numeric."
                )
            observed[target.id] = value
    return observed


def load_prices(
    protocol: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        path = ROOT / protocol["data_snapshots"][asset]["file"]
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame = frame.sort_values("Date").reset_index(drop=True)
        frame["Independent Price Row"] = np.arange(len(frame), dtype=int)
        output[asset] = frame
    return output


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(PREDICTIONS_PATH)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Target_Date"] = pd.to_datetime(
        frame["Target_Date"], errors="raise"
    )
    return frame.loc[
        frame["Asset"].isin(ASSETS)
        & frame["Period"].isin(PERIODS)
        & frame["Comparator"].isin(MODELS)
        & frame["Role"].eq("Main model")
    ].copy()


def reconstruct_signal_ledger(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rebuild the wide signal ledger without using the production script."""

    frames: list[pd.DataFrame] = []
    metadata = [
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
                current.loc[current["Comparator"].eq(MODELS[0]), metadata]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            locations = calendar.get_indexer(
                pd.DatetimeIndex(reference["Date"])
            )
            reference["Signal Price Row"] = locations
            reference["Execution Price Row"] = locations + 1
            reference["End Price Row"] = locations + 6
            reference["Five-Session Executable"] = (
                reference["End Price Row"] < len(asset_prices)
            )
            reference["Exclusion Reason"] = np.where(
                reference["Five-Session Executable"],
                "",
                "missing Open[t+6]",
            )
            reference["Asset"] = asset

            for model in MODELS:
                model_rows = (
                    current.loc[
                        current["Comparator"].eq(model),
                        ["Date", "Probability", "Prediction"],
                    ]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                reference[f"{model} Probability"] = model_rows[
                    "Probability"
                ].to_numpy(dtype=float)
                reference[f"{model} Signal"] = model_rows[
                    "Prediction"
                ].to_numpy(dtype=int)

            price_rows = asset_prices.iloc[locations].reset_index(drop=True)
            reference["Signal Close"] = price_rows["Close"].to_numpy(
                dtype=float
            )
            for column in ["EMA8", "EMA13", "SMA10", "SMA20"]:
                reference[column] = price_rows[column].to_numpy(dtype=float)
            reference["EMA8/EMA13 State Signal"] = (
                reference["EMA8"] > reference["EMA13"]
            ).astype(int)
            reference["SMA10/SMA20 State Signal"] = (
                reference["SMA10"] > reference["SMA20"]
            ).astype(int)

            execution_dates: list[pd.Timestamp | pd.NaT] = []
            end_dates: list[pd.Timestamp | pd.NaT] = []
            for executable, execution_row, end_row in zip(
                reference["Five-Session Executable"],
                reference["Execution Price Row"],
                reference["End Price Row"],
            ):
                if bool(executable):
                    execution_dates.append(
                        pd.Timestamp(
                            asset_prices.iloc[int(execution_row)]["Date"]
                        )
                    )
                    end_dates.append(
                        pd.Timestamp(asset_prices.iloc[int(end_row)]["Date"])
                    )
                else:
                    execution_dates.append(pd.NaT)
                    end_dates.append(pd.NaT)
            reference["Execution Date"] = execution_dates
            reference["Scheduled End Date"] = end_dates
            frames.append(reference)

    ledger = pd.concat(frames, ignore_index=True)
    for column in ["Date", "Target_Date", "Execution Date", "Scheduled End Date"]:
        ledger[column] = pd.to_datetime(ledger[column])
    return ledger


def independent_five_sleeve_path(
    asset: str,
    period: str,
    strategy: str,
    signal_ledger: pd.DataFrame,
    prices: pd.DataFrame,
    signal_column: str,
) -> IndependentPath:
    """Run an independent self-financing five-sleeve simulation."""

    current = signal_ledger.loc[
        signal_ledger["Asset"].eq(asset)
        & signal_ledger["Period"].eq(period)
        & signal_ledger["Five-Session Executable"]
    ].sort_values("Date")
    desired_values = current[signal_column].to_numpy(dtype=int)

    schedules: dict[int, dict[int, int]] = {
        phase: {} for phase in range(SLEEVE_COUNT)
    }
    final_rows: dict[int, int] = {}
    for (_, row), desired in zip(current.iterrows(), desired_values):
        phase = int(row["Phase_Offset"])
        execution_row = int(row["Execution Price Row"])
        end_row = int(row["End Price Row"])
        schedules[phase][execution_row] = int(desired)
        final_rows[phase] = end_row
    for phase, end_row in final_rows.items():
        schedules[phase][end_row] = 0

    start_row = int(current["Execution Price Row"].min())
    end_row = int(max(final_rows.values()))
    grid = np.arange(start_row, end_row + 1, dtype=int)
    dates = pd.to_datetime(prices["Date"]).reset_index(drop=True)
    opens = prices["Open"].to_numpy(dtype=float)

    net_sleeves = np.full(
        SLEEVE_COUNT, 1.0 / SLEEVE_COUNT, dtype=float
    )
    gross_sleeves = np.full(
        SLEEVE_COUNT, 1.0 / SLEEVE_COUNT, dtype=float
    )
    positions = np.zeros(SLEEVE_COUNT, dtype=int)
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for grid_offset, price_row in enumerate(grid):
        for phase in range(SLEEVE_COUNT):
            if price_row not in schedules[phase]:
                continue
            old_position = int(positions[phase])
            new_position = int(schedules[phase][price_row])
            if old_position == new_position:
                continue
            before = float(net_sleeves[phase])
            factor = (
                ENTRY_FACTOR
                if (old_position, new_position) == (0, 1)
                else EXIT_FACTOR
            )
            net_sleeves[phase] *= factor
            positions[phase] = new_position
            trade_rows.append(
                {
                    "Asset": asset,
                    "Period": period,
                    "Strategy": strategy,
                    "Open Date": dates.iloc[price_row],
                    "Phase": phase,
                    "Old Position": old_position,
                    "New Position": new_position,
                    "Transition": f"{old_position}->{new_position}",
                    "Net Sleeve NAV Before": before,
                    "Applied Factor": factor,
                    "Net Sleeve NAV After": float(net_sleeves[phase]),
                    "Cost Amount": float(before - net_sleeves[phase]),
                }
            )

        net_total = float(net_sleeves.sum())
        gross_total = float(gross_sleeves.sum())
        net_exposure = float(
            np.dot(net_sleeves, positions) / net_total
        )
        gross_exposure = float(
            np.dot(gross_sleeves, positions) / gross_total
        )
        row: dict[str, Any] = {
            "Asset": asset,
            "Period": period,
            "Strategy": strategy,
            "Open Date": dates.iloc[price_row],
            "Price Row": int(price_row),
            "Open Price": float(opens[price_row]),
            "Net NAV": net_total,
            "Gross NAV": gross_total,
            "Net Exposure": net_exposure,
            "Gross Exposure": gross_exposure,
            "Active Sleeves": int(positions.sum()),
        }
        for phase in range(SLEEVE_COUNT):
            row[f"Phase {phase} Net NAV"] = float(net_sleeves[phase])
            row[f"Phase {phase} Position"] = int(positions[phase])
        equity_rows.append(row)

        if grid_offset == len(grid) - 1:
            continue
        market_factor = opens[price_row + 1] / opens[price_row]
        growth = np.where(positions == 1, market_factor, 1.0)
        net_sleeves *= growth
        gross_sleeves *= growth

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    net_path = np.r_[1.0, equity["Net NAV"].to_numpy(dtype=float)]
    gross_path = np.r_[1.0, equity["Gross NAV"].to_numpy(dtype=float)]
    # Volatility is based only on the genuine open-to-open market intervals.
    # The initial same-open entry cost is reflected in total return and NAV,
    # but is not treated as an additional daily observation.
    net_returns = (
        equity["Net NAV"].astype(float).pct_change().dropna().to_numpy()
    )
    intervals = end_row - start_row
    interval_exposure = equity.iloc[:-1]["Net Exposure"].to_numpy(
        dtype=float
    )
    metric = {
        "Asset": asset,
        "Period": period,
        "Strategy": strategy,
        "Family": FAMILY[strategy],
        "Signal Rows": len(current),
        "Path Start": dates.iloc[start_row],
        "Path End": dates.iloc[end_row],
        "Market Intervals": intervals,
        "Gross Total Return": float(gross_path[-1] - 1.0),
        "Net Total Return": float(net_path[-1] - 1.0),
        "Cost Drag": float(gross_path[-1] - net_path[-1]),
        "Annualised Net Return": annualised_return(
            float(net_path[-1]), intervals
        ),
        "Annualised Volatility": float(
            np.std(net_returns, ddof=1)
            * math.sqrt(SESSIONS_PER_YEAR)
        ),
        "Maximum Drawdown": maximum_drawdown(net_path),
        "Average Exposure": float(np.mean(interval_exposure)),
        "One-way Sleeve Changes": len(trades),
        "Terminal All Cash": bool(
            int(equity.iloc[-1]["Active Sleeves"]) == 0
        ),
    }
    return IndependentPath(metric=metric, equity=equity, trades=trades)


def independent_buy_hold(
    asset: str,
    period: str,
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
    alpha: float,
    label: str,
) -> IndependentPath:
    """Build an actual shares-plus-cash Buy-and-Hold path."""

    opens = prices["Open"].to_numpy(dtype=float)
    dates = pd.to_datetime(prices["Date"]).reset_index(drop=True)
    grid = np.arange(start_row, end_row + 1, dtype=int)
    price_ratio = opens[grid] / opens[start_row]
    net_stock_value = alpha * ENTRY_FACTOR * price_ratio
    gross_stock_value = alpha * price_ratio
    cash_value = 1.0 - alpha
    net_nav = cash_value + net_stock_value
    gross_nav = cash_value + gross_stock_value
    net_exposure_intervals = (
        net_stock_value[:-1] / net_nav[:-1]
    )
    gross_exposure_intervals = (
        gross_stock_value[:-1] / gross_nav[:-1]
    )
    net_nav = net_nav.copy()
    net_nav[-1] = cash_value + net_stock_value[-1] * EXIT_FACTOR
    net_path = np.r_[1.0, net_nav]
    gross_path = np.r_[1.0, gross_nav]
    net_returns = pd.Series(net_nav).pct_change().dropna().to_numpy()
    intervals = end_row - start_row

    equity = pd.DataFrame(
        {
            "Asset": asset,
            "Period": period,
            "Strategy": label,
            "Open Date": dates.iloc[grid].to_numpy(),
            "Price Row": grid,
            "Open Price": opens[grid],
            "Net NAV": net_nav,
            "Gross NAV": gross_nav,
            "Net Exposure": np.r_[net_exposure_intervals, 0.0],
            "Gross Exposure": np.r_[gross_exposure_intervals, 0.0],
            "Active Sleeves": np.r_[
                np.ones(intervals, dtype=int), 0
            ],
        }
    )
    metric = {
        "Asset": asset,
        "Period": period,
        "Strategy": label,
        "Family": FAMILY.get(label, "Exposure-matched baseline"),
        "Signal Rows": None,
        "Path Start": dates.iloc[start_row],
        "Path End": dates.iloc[end_row],
        "Market Intervals": intervals,
        "Gross Total Return": float(gross_path[-1] - 1.0),
        "Net Total Return": float(net_path[-1] - 1.0),
        "Cost Drag": float(gross_path[-1] - net_path[-1]),
        "Annualised Net Return": annualised_return(
            float(net_path[-1]), intervals
        ),
        "Annualised Volatility": float(
            np.std(net_returns, ddof=1)
            * math.sqrt(SESSIONS_PER_YEAR)
        ),
        "Maximum Drawdown": maximum_drawdown(net_path),
        "Average Exposure": float(
            np.mean(net_exposure_intervals)
        ),
        "Gross Average Exposure": float(
            np.mean(gross_exposure_intervals)
        ),
        "One-way Sleeve Changes": 2,
        "Terminal All Cash": True,
        "Initial Stock Allocation": float(alpha),
    }
    return IndependentPath(
        metric=metric,
        equity=equity,
        trades=pd.DataFrame(),
    )


def independent_matched_alpha(
    target_exposure: float,
    asset: str,
    period: str,
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
) -> float:
    if target_exposure <= 0.0:
        return 0.0
    if target_exposure >= 1.0:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        path = independent_buy_hold(
            asset,
            period,
            prices,
            start_row,
            end_row,
            midpoint,
            "Exposure-Matched Buy-and-Hold",
        )
        if path.metric["Average Exposure"] < target_exposure:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)


def independent_constant_exposure_passive(
    prices: pd.DataFrame,
    start_row: int,
    end_row: int,
    exposure: float,
) -> dict[str, Any]:
    """Rebuild the synthetic constant-exposure passive reference.

    Each genuine open-to-open interval compounds
    ``1 + exposure * market_return``.  Entry and terminal friction affect
    only the exposed fraction.  No intermediate rebalancing cost is charged;
    this intentionally favours the passive reference.
    """

    if not 0.0 <= exposure <= 1.0:
        raise ValueError("Constant exposure must fall in [0, 1].")
    opens = prices["Open"].to_numpy(dtype=float)
    market_returns = (
        opens[start_row + 1 : end_row + 1]
        / opens[start_row:end_row]
        - 1.0
    )
    interval_factors = 1.0 + exposure * market_returns
    gross_nav = np.r_[1.0, np.cumprod(interval_factors)]
    entry_portfolio_factor = (
        (1.0 - exposure) + exposure * ENTRY_FACTOR
    )
    exit_portfolio_factor = (
        (1.0 - exposure) + exposure * EXIT_FACTOR
    )
    net_nav = gross_nav * entry_portfolio_factor
    net_nav[-1] *= exit_portfolio_factor
    intervals = end_row - start_row
    return {
        "Exposure": float(exposure),
        "Gross Total Return": float(gross_nav[-1] - 1.0),
        "Net Total Return": float(net_nav[-1] - 1.0),
        "Maximum Drawdown": maximum_drawdown(np.r_[1.0, net_nav]),
        "Annualised Net Return": annualised_return(
            float(net_nav[-1]), intervals
        ),
        "Entry Portfolio Factor": float(entry_portfolio_factor),
        "Exit Portfolio Factor": float(exit_portfolio_factor),
        "Intermediate Rebalancing Costs Included": False,
    }


def compare_keyed_frames(
    saved: pd.DataFrame,
    rebuilt: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str],
    text_columns: list[str],
) -> tuple[bool, float, list[str]]:
    """Compare two tables after a one-to-one key merge."""

    left = saved.copy()
    right = rebuilt.copy()
    merged = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=("_saved", "_rebuilt"),
        indicator=True,
        validate="one_to_one",
    )
    problems: list[str] = []
    if not merged["_merge"].eq("both").all():
        problems.append(
            f"unmatched_rows={int((merged['_merge'] != 'both').sum())}"
        )
    maximum = 0.0
    for column in numeric_columns:
        saved_name = f"{column}_saved"
        rebuilt_name = f"{column}_rebuilt"
        if saved_name not in merged or rebuilt_name not in merged:
            problems.append(f"missing_numeric_column={column}")
            continue
        passed, difference = numeric_difference(
            merged[saved_name], merged[rebuilt_name]
        )
        maximum = max(maximum, difference)
        if not passed:
            problems.append(
                f"{column}:max_abs_difference={difference:.6g}"
            )
    for column in text_columns:
        saved_name = f"{column}_saved"
        rebuilt_name = f"{column}_rebuilt"
        if saved_name not in merged or rebuilt_name not in merged:
            problems.append(f"missing_text_column={column}")
            continue
        left_values = merged[saved_name]
        right_values = merged[rebuilt_name]
        both_missing = left_values.isna() & right_values.isna()
        text_equal = (
            left_values.astype(str).to_numpy()
            == right_values.astype(str).to_numpy()
        )
        if not np.all(both_missing.to_numpy() | text_equal):
            problems.append(f"text_mismatch={column}")
    return not problems, maximum, problems


def validate_png(path: Path) -> tuple[bool, dict[str, int]]:
    if not path.exists() or path.stat().st_size < 10_000:
        return False, {
            "bytes": path.stat().st_size if path.exists() else 0,
            "width": 0,
            "height": 0,
        }
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return False, {
            "bytes": path.stat().st_size,
            "width": 0,
            "height": 0,
        }
    width, height = struct.unpack(">II", header[16:24])
    return bool(width >= 600 and height >= 300), {
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
    }


def main(*, write_artifacts: bool = False) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    recorder = ValidationRecorder()
    protocol = load_protocol()
    prediction_manifest = json.loads(
        PREDICTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    prices = load_prices(protocol)
    predictions = load_predictions()

    # ------------------------------------------------------------------
    # Frozen inputs and data quality
    # ------------------------------------------------------------------
    frozen_expectations: dict[Path, str] = {}
    for asset in ASSETS:
        snapshot = protocol["data_snapshots"][asset]
        frozen_expectations[ROOT / snapshot["file"]] = snapshot["sha256"]
    upstream = protocol["upstream_prediction_protocol"]
    frozen_expectations[
        ROOT / upstream["rq3_protocol_file"]
    ] = upstream["rq3_protocol_sha256"]
    frozen_expectations[
        ROOT / upstream["rq3_engine_file"]
    ] = upstream["rq3_engine_sha256"]
    frozen_expectations[
        ROOT / upstream["tsla_frozen_predictions_file"]
    ] = upstream["tsla_frozen_predictions_sha256"]
    frozen_observed = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in frozen_expectations
    }
    recorder.add(
        "INP-01",
        "Inputs",
        "Every predeclared price and RQ3 upstream input matches protocol SHA-256.",
        all(
            sha256(path) == expected
            for path, expected in frozen_expectations.items()
        ),
        {
            str(path.relative_to(ROOT)): expected
            for path, expected in frozen_expectations.items()
        },
        frozen_observed,
    )
    combined_hash_ok = (
        prediction_manifest["combined_prediction_sha256"]
        == sha256(PREDICTIONS_PATH)
        and prediction_manifest[
            "return_metrics_used_for_training_or_selection"
        ]
        is False
    )
    recorder.add(
        "INP-02",
        "Inputs",
        "The combined three-asset prediction ledger matches its manifest and was not return-selected.",
        combined_hash_ok,
        {
            "sha256": prediction_manifest["combined_prediction_sha256"],
            "return_selected": False,
        },
        {
            "sha256": sha256(PREDICTIONS_PATH),
            "return_selected": prediction_manifest[
                "return_metrics_used_for_training_or_selection"
            ],
        },
    )
    metadata_trace_ok = (
        metadata["protocol_sha256"] == sha256(PROTOCOL_PATH)
        and metadata["run_script_sha256"] == sha256(RUN_SCRIPT_PATH)
        and metadata["prediction_manifest_sha256"]
        == sha256(PREDICTION_MANIFEST_PATH)
        and metadata["combined_predictions_sha256"]
        == sha256(PREDICTIONS_PATH)
    )
    recorder.add(
        "INP-03",
        "Inputs",
        "Experiment metadata traces the exact protocol, run script, manifest, and predictions.",
        metadata_trace_ok,
        "all four saved SHA-256 values match current formal files",
        {
            "protocol": metadata["protocol_sha256"],
            "run_script": metadata["run_script_sha256"],
            "prediction_manifest": metadata[
                "prediction_manifest_sha256"
            ],
            "predictions": metadata["combined_predictions_sha256"],
        },
    )
    production_constants = read_production_numeric_constants()
    mapping = protocol["prediction_to_decision_mapping"]
    frictions = protocol["frictions"]
    protocol_constants = {
        "THRESHOLD": float(
            protocol["upstream_prediction_protocol"][
                "probability_threshold"
            ]
        ),
        "HORIZON": int(mapping["holding_open_to_open_intervals"]),
        "SLEEVES": int(mapping["sleeves"]),
        "TRANSACTION_COST": float(
            frictions["transaction_cost_bps"]
        )
        / 10_000.0,
        "SLIPPAGE": float(frictions["adverse_slippage_bps"])
        / 10_000.0,
    }
    validator_constants = {
        "THRESHOLD": THRESHOLD,
        "HORIZON": HORIZON,
        "SLEEVES": SLEEVE_COUNT,
        "TRANSACTION_COST": TRANSACTION_COST,
        "SLIPPAGE": SLIPPAGE,
        "TRADING_SESSIONS_PER_YEAR": SESSIONS_PER_YEAR,
    }
    constant_names = set(validator_constants)
    constant_alignment_ok = (
        set(production_constants) == constant_names
        and all(
            np.isclose(
                float(production_constants[name]),
                float(validator_constants[name]),
            )
            for name in constant_names
        )
        and all(
            np.isclose(
                float(production_constants[name]),
                float(protocol_constants[name]),
            )
            for name in protocol_constants
        )
        and np.isclose(float(metadata["entry_factor"]), ENTRY_FACTOR)
        and np.isclose(float(metadata["exit_factor"]), EXIT_FACTOR)
    )
    recorder.add(
        "INP-04",
        "Protocol and code",
        "Protocol, production source constants, validator constants, and saved friction factors are aligned.",
        constant_alignment_ok,
        validator_constants,
        {
            "production_source": production_constants,
            "protocol": protocol_constants,
            "metadata_entry_factor": metadata["entry_factor"],
            "metadata_exit_factor": metadata["exit_factor"],
        },
    )
    total_one_way_bps = float(frictions["total_one_way_bps"])
    cost_bps = float(frictions["transaction_cost_bps"])
    slippage_bps = float(frictions["adverse_slippage_bps"])
    range_ok = (
        0.0 <= protocol_constants["THRESHOLD"] <= 1.0
        and protocol_constants["HORIZON"] >= 1
        and protocol_constants["SLEEVES"] >= 1
        and 0.0 < float(mapping["initial_weight_per_sleeve"]) <= 1.0
        and np.isclose(
            protocol_constants["SLEEVES"]
            * float(mapping["initial_weight_per_sleeve"]),
            1.0,
        )
        and cost_bps >= 0.0
        and slippage_bps >= 0.0
        and total_one_way_bps < 10_000.0
        and np.isclose(total_one_way_bps, cost_bps + slippage_bps)
        and 0.0 < ENTRY_FACTOR <= 1.0
        and 0.0 < EXIT_FACTOR <= 1.0
        and SESSIONS_PER_YEAR > 0
    )
    recorder.add(
        "INP-05",
        "Protocol and code",
        "Threshold, horizon, sleeves, initial weights, costs, and annualisation constants are within valid ranges and reconcile.",
        range_ok,
        {
            "threshold": "[0, 1]",
            "horizon_and_sleeves": "positive integers",
            "sleeve_weights_sum": 1.0,
            "total_one_way_bps": cost_bps + slippage_bps,
            "entry_and_exit_factors": "(0, 1]",
        },
        {
            "threshold": protocol_constants["THRESHOLD"],
            "horizon": protocol_constants["HORIZON"],
            "sleeves": protocol_constants["SLEEVES"],
            "initial_weight": mapping["initial_weight_per_sleeve"],
            "total_one_way_bps": total_one_way_bps,
            "entry_factor": ENTRY_FACTOR,
            "exit_factor": EXIT_FACTOR,
            "sessions_per_year": SESSIONS_PER_YEAR,
        },
    )

    required_price_fields = {
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
    data_issues: list[str] = []
    calendars: list[pd.DatetimeIndex] = []
    for asset, frame in prices.items():
        missing = required_price_fields.difference(frame.columns)
        if missing:
            data_issues.append(f"{asset}:missing={sorted(missing)}")
            continue
        calendars.append(pd.DatetimeIndex(frame["Date"]))
        if len(frame) != 2513:
            data_issues.append(f"{asset}:rows={len(frame)}")
        if frame["Date"].duplicated().any():
            data_issues.append(f"{asset}:duplicate_dates")
        if not frame["Date"].is_monotonic_increasing:
            data_issues.append(f"{asset}:unsorted_dates")
        # Rolling SMA fields legitimately contain warm-up NaNs near the start
        # of the ten-year snapshot.  Source completeness is therefore checked
        # on OHLCV here; the ledger comparison below separately verifies that
        # every evaluated signal origin has the required MA values.
        if frame[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
            data_issues.append(f"{asset}:missing_ohlcv_values")
        if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
            data_issues.append(f"{asset}:nonpositive_ohlc")
        if (frame["Volume"] < 0).any():
            data_issues.append(f"{asset}:negative_volume")
        if (
            frame["High"]
            < frame[["Open", "Close", "Low"]].max(axis=1)
        ).any():
            data_issues.append(f"{asset}:invalid_high")
        if (
            frame["Low"]
            > frame[["Open", "Close", "High"]].min(axis=1)
        ).any():
            data_issues.append(f"{asset}:invalid_low")
    common_calendar = (
        len(calendars) == len(ASSETS)
        and all(calendars[0].equals(item) for item in calendars[1:])
    )
    if not common_calendar:
        data_issues.append("asset_calendars_differ")
    recorder.add(
        "DATA-01",
        "Raw data",
        "The three price snapshots have complete OHLCV, valid ranges, ordered unique dates, and one common 2,513-session calendar.",
        not data_issues,
        "3 assets × 2,513 rows; common calendar; valid non-missing OHLCV and MA fields",
        data_issues or "all raw-data checks passed",
    )

    # ------------------------------------------------------------------
    # Prediction population, target labels, and temporal grid
    # ------------------------------------------------------------------
    expected_counts = {
        "Development OOF": 1250,
        "Retrospective audit": 407,
    }
    group_counts = (
        predictions.groupby(["Asset", "Period", "Comparator"])
        .size()
        .to_dict()
    )
    population_ok = (
        len(predictions) == 19_884
        and not predictions.duplicated(
            ["Asset", "Period", "Comparator", "Date"]
        ).any()
        and all(
            group_counts.get((asset, period, model))
            == expected_counts[period]
            for asset in ASSETS
            for period in PERIODS
            for model in MODELS
        )
    )
    recorder.add(
        "PRED-01",
        "Predictions",
        "All four model ledgers are complete and unique for all assets and periods.",
        population_ok,
        "19,884 rows; 24 complete asset-period-model groups",
        {
            "rows": len(predictions),
            "groups": len(group_counts),
            "duplicates": int(
                predictions.duplicated(
                    ["Asset", "Period", "Comparator", "Date"]
                ).sum()
            ),
        },
    )
    threshold_ok = (
        predictions["Probability"].between(0.0, 1.0).all()
        and np.isfinite(
            predictions["Probability"].to_numpy(dtype=float)
        ).all()
        and np.array_equal(
            predictions["Prediction"].to_numpy(dtype=int),
            (
                predictions["Probability"].to_numpy(dtype=float)
                >= THRESHOLD
            ).astype(int),
        )
    )
    recorder.add(
        "PRED-02",
        "Predictions",
        "Every saved model signal reproduces from the frozen 0.50 probability threshold.",
        threshold_ok,
        "Prediction = 1[Probability >= 0.50] for 19,884 rows",
        f"checked_rows={len(predictions)}",
    )

    label_mismatches = 0
    target_date_mismatches = 0
    absent_origins = 0
    unique_origins = predictions.loc[
        predictions["Comparator"].eq(MODELS[0])
    ].copy()
    for asset in ASSETS:
        frame = prices[asset]
        calendar = pd.DatetimeIndex(frame["Date"])
        closes = frame["Close"].to_numpy(dtype=float)
        current = unique_origins.loc[
            unique_origins["Asset"].eq(asset)
        ]
        locations = calendar.get_indexer(pd.DatetimeIndex(current["Date"]))
        for (_, row), location in zip(current.iterrows(), locations):
            if location < 4 or location + 5 >= len(frame):
                absent_origins += 1
                continue
            past_mean = float(np.mean(closes[location - 4 : location + 1]))
            future_mean = float(np.mean(closes[location + 1 : location + 6]))
            expected_label = int(future_mean > past_mean)
            if int(row["Actual"]) != expected_label:
                label_mismatches += 1
            expected_target_date = pd.Timestamp(
                frame.iloc[location + 5]["Date"]
            )
            if pd.Timestamp(row["Target_Date"]) != expected_target_date:
                target_date_mismatches += 1
    recorder.add(
        "PRED-03",
        "Target",
        "Actual labels independently reproduce the RQ3 five-future-close-mean target.",
        label_mismatches == 0 and absent_origins == 0,
        "0 mismatches across 4,971 unique asset-period origins",
        {
            "mismatches": label_mismatches,
            "unreconstructable": absent_origins,
            "origins": len(unique_origins),
        },
    )
    recorder.add(
        "PRED-04",
        "Target",
        "Target_Date is the fifth future trading session (t+5), distinct from the t+6 trade endpoint.",
        target_date_mismatches == 0,
        "0 Target_Date mismatches",
        {
            "mismatches": target_date_mismatches,
            "origins": len(unique_origins),
        },
    )

    grid_issues: list[str] = []
    for asset in ASSETS:
        for period in PERIODS:
            current = predictions.loc[
                predictions["Asset"].eq(asset)
                & predictions["Period"].eq(period)
            ]
            reference = (
                current.loc[current["Comparator"].eq(MODELS[0])]
                .sort_values("Date")
                .reset_index(drop=True)
            )
            eligible = reference["Eligible_Row_ID"].to_numpy(dtype=int)
            if not np.all(np.diff(eligible) == 1):
                grid_issues.append(f"{asset}/{period}:nonconsecutive")
            if not np.array_equal(
                reference["Phase_Offset"].to_numpy(dtype=int),
                eligible % HORIZON,
            ):
                grid_issues.append(f"{asset}/{period}:phase")
            for model in MODELS[1:]:
                comparison = (
                    current.loc[current["Comparator"].eq(model)]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                for column in [
                    "Date",
                    "Target_Date",
                    "Eligible_Row_ID",
                    "Phase_Offset",
                    "Actual",
                    "Period",
                    "Fold",
                ]:
                    if not np.array_equal(
                        reference[column].astype(str).to_numpy(),
                        comparison[column].astype(str).to_numpy(),
                    ):
                        grid_issues.append(
                            f"{asset}/{period}/{model}:{column}"
                        )
    recorder.add(
        "PRED-05",
        "Predictions",
        "Models share an identical consecutive origin grid and Eligible_Row_ID modulo five phase assignment.",
        not grid_issues,
        "common metadata grid for 6 asset-period panels",
        grid_issues or "all grids and phases agree",
    )

    frozen_tsla = pd.read_csv(TSLA_FROZEN_PREDICTIONS_PATH)
    frozen_tsla["Date"] = pd.to_datetime(frozen_tsla["Date"])
    frozen_tsla["Target_Date"] = pd.to_datetime(
        frozen_tsla["Target_Date"]
    )
    frozen_tsla = frozen_tsla.loc[
        frozen_tsla["Role"].eq("Main model")
        & frozen_tsla["Comparator"].isin(MODELS)
    ].copy()
    combined_tsla = predictions.loc[predictions["Asset"].eq("TSLA")].copy()
    keys = ["Period", "Comparator", "Date"]
    tsla_merged = combined_tsla.merge(
        frozen_tsla,
        on=keys,
        how="outer",
        suffixes=("_combined", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    tsla_columns = [
        "Target_Date",
        "Eligible_Row_ID",
        "Phase_Offset",
        "Actual",
        "Role",
        "Fold",
        "Probability",
        "Prediction",
    ]
    tsla_exact = tsla_merged["_merge"].eq("both").all()
    for column in tsla_columns:
        if column == "Probability":
            same, _ = numeric_difference(
                tsla_merged[f"{column}_combined"],
                tsla_merged[f"{column}_frozen"],
            )
        else:
            same = np.array_equal(
                tsla_merged[f"{column}_combined"].astype(str).to_numpy(),
                tsla_merged[f"{column}_frozen"].astype(str).to_numpy(),
            )
        tsla_exact = tsla_exact and same
    recorder.add(
        "PRED-06",
        "Predictions",
        "The TSLA portion of the combined file exactly preserves frozen RQ3 model predictions.",
        tsla_exact,
        "6,628 frozen TSLA rows copied without mutation",
        {
            "combined_rows": len(combined_tsla),
            "frozen_rows": len(frozen_tsla),
            "unmatched": int((tsla_merged["_merge"] != "both").sum()),
        },
    )

    # ------------------------------------------------------------------
    # Reconstruct and compare the signal ledger, including t+6 boundary
    # ------------------------------------------------------------------
    independent_ledger = reconstruct_signal_ledger(predictions, prices)
    saved_ledger = pd.read_csv(SIGNAL_LEDGER_PATH, keep_default_na=False)
    for column in [
        "Date",
        "Target_Date",
        "Execution Date",
        "Scheduled End Date",
    ]:
        saved_ledger[column] = pd.to_datetime(
            saved_ledger[column], errors="coerce"
        )
    saved_ledger["Five-Session Executable"] = normalise_bool(
        saved_ledger["Five-Session Executable"]
    )
    ledger_structure_ok = (
        len(saved_ledger) == 4_971
        and not saved_ledger.duplicated(
            ["Asset", "Period", "Date"]
        ).any()
        and set(saved_ledger["Asset"]) == set(ASSETS)
        and set(saved_ledger["Period"]) == set(PERIODS)
    )
    recorder.add(
        "LEDGER-01",
        "Signal ledger",
        "The saved wide signal ledger has one unique row per asset-period-origin.",
        ledger_structure_ok,
        "4,971 unique rows",
        {
            "rows": len(saved_ledger),
            "duplicates": int(
                saved_ledger.duplicated(
                    ["Asset", "Period", "Date"]
                ).sum()
            ),
        },
    )
    ledger_numeric = [
        "Eligible_Row_ID",
        "Phase_Offset",
        "Actual",
        "Signal Price Row",
        "Execution Price Row",
        "End Price Row",
        "Signal Close",
        "EMA8",
        "EMA13",
        "SMA10",
        "SMA20",
        "EMA8/EMA13 State Signal",
        "SMA10/SMA20 State Signal",
        *[
            item
            for model in MODELS
            for item in [
                f"{model} Probability",
                f"{model} Signal",
            ]
        ],
    ]
    ledger_text = [
        "Target_Date",
        "Fold",
        "Execution Date",
        "Scheduled End Date",
        "Five-Session Executable",
        "Exclusion Reason",
    ]
    ledger_match, ledger_maximum, ledger_problems = compare_keyed_frames(
        saved_ledger,
        independent_ledger,
        ["Asset", "Period", "Date"],
        ledger_numeric,
        ledger_text,
    )
    recorder.diagnostics["signal_ledger_max_abs_difference"] = ledger_maximum
    recorder.add(
        "LEDGER-02",
        "Signal ledger",
        "Saved probabilities, signals, indicators, row indices, and dates match an independent rebuild.",
        ledger_match,
        f"maximum numeric difference <= {ABSOLUTE_TOLERANCE}",
        {
            "max_abs_difference": ledger_maximum,
            "problems": ledger_problems,
        },
    )

    executable = independent_ledger.loc[
        independent_ledger["Five-Session Executable"]
    ]
    date_mapping_ok = True
    mapping_mismatches = 0
    for asset in ASSETS:
        calendar = pd.DatetimeIndex(prices[asset]["Date"])
        current = executable.loc[executable["Asset"].eq(asset)]
        origins = calendar.get_indexer(pd.DatetimeIndex(current["Date"]))
        expected_execution = calendar[origins + 1]
        expected_end = calendar[origins + 6]
        current_ok = (
            pd.DatetimeIndex(current["Execution Date"]).equals(
                expected_execution
            )
            and pd.DatetimeIndex(current["Scheduled End Date"]).equals(
                expected_end
            )
        )
        if not current_ok:
            date_mapping_ok = False
            mapping_mismatches += 1
    recorder.add(
        "LEDGER-03",
        "Temporal mapping",
        "Every executable signal maps Close[t] to Open[t+1] execution and Open[t+6] scheduled end.",
        date_mapping_ok,
        "t+1 execution and t+6 endpoint for 4,968 signals",
        {
            "executable_rows": len(executable),
            "asset_panels_with_mismatch": mapping_mismatches,
        },
    )
    boundary_counts = (
        independent_ledger.groupby(["Asset", "Period"])[
            "Five-Session Executable"
        ]
        .sum()
        .astype(int)
        .to_dict()
    )
    excluded = independent_ledger.loc[
        ~independent_ledger["Five-Session Executable"]
    ]
    boundary_ok = (
        all(
            boundary_counts[(asset, "Development OOF")] == 1250
            and boundary_counts[(asset, "Retrospective audit")] == 406
            for asset in ASSETS
        )
        and len(excluded) == 3
        and set(excluded["Date"]) == {pd.Timestamp("2026-07-15")}
        and excluded["Exclusion Reason"].eq("missing Open[t+6]").all()
    )
    recorder.add(
        "LEDGER-04",
        "Temporal boundary",
        "The later period retains 406 executable origins per asset and excludes only the terminal t+6 boundary.",
        boundary_ok,
        "1,250 Development and 406/407 audit origins executable per asset; only 2026-07-15 excluded",
        {
            "counts": {
                f"{asset}/{period}": count
                for (asset, period), count in boundary_counts.items()
            },
            "excluded": excluded[
                ["Asset", "Period", "Date", "Exclusion Reason"]
            ]
            .astype(str)
            .to_dict(orient="records"),
        },
    )

    # ------------------------------------------------------------------
    # Independent five-sleeve, Buy-and-Hold, and metric calculations
    # ------------------------------------------------------------------
    signal_columns = {
        **{model: f"{model} Signal" for model in MODELS},
        "EMA8/EMA13 State": "EMA8/EMA13 State Signal",
        "SMA10/SMA20 State": "SMA10/SMA20 State Signal",
    }
    independent_paths: dict[
        tuple[str, str, str], IndependentPath
    ] = {}
    independent_metric_rows: list[dict[str, Any]] = []
    independent_equity_frames: list[pd.DataFrame] = []
    independent_trade_frames: list[pd.DataFrame] = []

    for asset in ASSETS:
        for period in PERIODS:
            current = independent_ledger.loc[
                independent_ledger["Asset"].eq(asset)
                & independent_ledger["Period"].eq(period)
                & independent_ledger["Five-Session Executable"]
            ]
            start_row = int(current["Execution Price Row"].min())
            end_row = int(current["End Price Row"].max())
            for strategy in SLEEVE_STRATEGIES:
                path = independent_five_sleeve_path(
                    asset,
                    period,
                    strategy,
                    independent_ledger,
                    prices[asset],
                    signal_columns[strategy],
                )
                independent_paths[(asset, period, strategy)] = path
                independent_metric_rows.append(path.metric)
                independent_equity_frames.append(path.equity)
                independent_trade_frames.append(path.trades)
            buy_hold = independent_buy_hold(
                asset,
                period,
                prices[asset],
                start_row,
                end_row,
                1.0,
                "Buy-and-Hold",
            )
            buy_hold.metric["Signal Rows"] = len(current)
            independent_paths[
                (asset, period, "Buy-and-Hold")
            ] = buy_hold
            independent_metric_rows.append(buy_hold.metric)
            independent_equity_frames.append(buy_hold.equity)

    independent_metrics = pd.DataFrame(independent_metric_rows)
    independent_equity = pd.concat(
        independent_equity_frames, ignore_index=True
    )
    independent_trades = pd.concat(
        independent_trade_frames, ignore_index=True
    )
    saved_equity = pd.read_csv(EQUITY_PATH)
    saved_equity["Open Date"] = pd.to_datetime(
        saved_equity["Open Date"], errors="raise"
    )
    saved_trades = pd.read_csv(TRADES_PATH)
    saved_trades["Open Date"] = pd.to_datetime(
        saved_trades["Open Date"], errors="raise"
    )

    equity_numeric = [
        "Price Row",
        "Open Price",
        "Net NAV",
        "Gross NAV",
        "Net Exposure",
        "Gross Exposure",
        "Active Sleeves",
    ]
    equity_match, equity_maximum, equity_problems = compare_keyed_frames(
        saved_equity,
        independent_equity,
        ["Asset", "Period", "Strategy", "Open Date"],
        equity_numeric,
        [],
    )
    recorder.diagnostics["equity_max_abs_difference"] = equity_maximum
    recorder.add(
        "PATH-01",
        "Portfolio paths",
        "Every saved strategy NAV, gross path, exposure, open price, and active-sleeve count independently reproduces.",
        equity_match,
        f"maximum numeric difference <= {ABSOLUTE_TOLERANCE}",
        {
            "saved_rows": len(saved_equity),
            "rebuilt_rows": len(independent_equity),
            "max_abs_difference": equity_maximum,
            "problems": equity_problems,
        },
    )

    saved_sleeves = saved_equity.loc[
        saved_equity["Strategy"].isin(SLEEVE_STRATEGIES)
    ].copy()
    phase_nav_columns = [
        f"Phase {phase} Net NAV" for phase in range(SLEEVE_COUNT)
    ]
    phase_position_columns = [
        f"Phase {phase} Position" for phase in range(SLEEVE_COUNT)
    ]
    phase_nav_sum = saved_sleeves[phase_nav_columns].sum(axis=1)
    phase_position_sum = saved_sleeves[
        phase_position_columns
    ].sum(axis=1)
    self_financing_ok = (
        np.allclose(
            phase_nav_sum,
            saved_sleeves["Net NAV"],
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
        and np.array_equal(
            phase_position_sum.to_numpy(dtype=int),
            saved_sleeves["Active Sleeves"].to_numpy(dtype=int),
        )
        and saved_sleeves[phase_position_columns]
        .isin([0, 1])
        .all()
        .all()
    )
    recorder.add(
        "PATH-02",
        "Five sleeves",
        "Each portfolio is the sum of five independently evolving sleeve NAVs without cross-sleeve capital rebalancing.",
        self_financing_ok,
        "phase NAVs sum to portfolio NAV and phase positions sum to Active Sleeves on every row",
        {
            "rows": len(saved_sleeves),
            "max_nav_reconciliation_error": float(
                np.max(
                    np.abs(
                        phase_nav_sum.to_numpy(dtype=float)
                        - saved_sleeves["Net NAV"].to_numpy(dtype=float)
                    )
                )
            ),
        },
    )

    reconstructed_exposure = (
        saved_sleeves[phase_nav_columns].to_numpy(dtype=float)
        * saved_sleeves[phase_position_columns].to_numpy(dtype=float)
    ).sum(axis=1) / saved_sleeves["Net NAV"].to_numpy(dtype=float)
    exposure_ok, exposure_difference = numeric_difference(
        saved_sleeves["Net Exposure"], reconstructed_exposure
    )
    naive_exposure = (
        saved_sleeves["Active Sleeves"].to_numpy(dtype=float)
        / SLEEVE_COUNT
    )
    value_weighting_is_material = bool(
        np.max(np.abs(reconstructed_exposure - naive_exposure)) > 1e-6
    )
    recorder.add(
        "PATH-03",
        "Exposure",
        "Portfolio exposure is value-weighted from live sleeve NAVs rather than reported as Active Sleeves divided by five.",
        exposure_ok and value_weighting_is_material,
        "Net Exposure = sum(phase NAV × phase position) / total NAV; differs from naive count when sleeve values diverge",
        {
            "max_formula_difference": exposure_difference,
            "max_value_weighted_vs_naive_difference": float(
                np.max(
                    np.abs(
                        reconstructed_exposure - naive_exposure
                    )
                )
            ),
        },
    )

    last_rows = (
        saved_equity.sort_values("Open Date")
        .groupby(["Asset", "Period", "Strategy"], as_index=False)
        .tail(1)
    )
    terminal_ok = (
        last_rows["Active Sleeves"].eq(0).all()
        and last_rows["Net Exposure"].abs().le(ABSOLUTE_TOLERANCE).all()
    )
    recorder.add(
        "PATH-04",
        "Terminal rule",
        "Every asset-period-strategy path closes to cash at its scheduled terminal open.",
        terminal_ok,
        "42 paths end with zero active sleeves and zero exposure",
        {
            "paths": len(last_rows),
            "noncash": int(
                (
                    ~last_rows["Active Sleeves"].eq(0)
                    | ~last_rows["Net Exposure"]
                    .abs()
                    .le(ABSOLUTE_TOLERANCE)
                ).sum()
            ),
        },
    )

    trade_numeric = [
        "Phase",
        "Old Position",
        "New Position",
        "Net Sleeve NAV Before",
        "Applied Factor",
        "Net Sleeve NAV After",
        "Cost Amount",
    ]
    trade_match, trade_maximum, trade_problems = compare_keyed_frames(
        saved_trades,
        independent_trades,
        [
            "Asset",
            "Period",
            "Strategy",
            "Open Date",
            "Phase",
            "Transition",
        ],
        [
            item
            for item in trade_numeric
            if item not in {"Phase"}
        ],
        [],
    )
    recorder.diagnostics["trade_ledger_max_abs_difference"] = trade_maximum
    recorder.add(
        "COST-01",
        "Trades and costs",
        "The complete saved transition ledger independently reproduces from sleeve state changes.",
        trade_match,
        f"same keyed transitions and maximum numeric difference <= {ABSOLUTE_TOLERANCE}",
        {
            "saved_rows": len(saved_trades),
            "rebuilt_rows": len(independent_trades),
            "max_abs_difference": trade_maximum,
            "problems": trade_problems,
        },
    )

    entry_rows = saved_trades["Transition"].eq("0->1")
    exit_rows = saved_trades["Transition"].eq("1->0")
    factors_ok = (
        (entry_rows | exit_rows).all()
        and np.allclose(
            saved_trades.loc[entry_rows, "Applied Factor"],
            ENTRY_FACTOR,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
        and np.allclose(
            saved_trades.loc[exit_rows, "Applied Factor"],
            EXIT_FACTOR,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
    )
    expected_after = (
        saved_trades["Net Sleeve NAV Before"].to_numpy(dtype=float)
        * saved_trades["Applied Factor"].to_numpy(dtype=float)
    )
    expected_cost = (
        saved_trades["Net Sleeve NAV Before"].to_numpy(dtype=float)
        - expected_after
    )
    cost_arithmetic_ok = (
        np.allclose(
            saved_trades["Net Sleeve NAV After"],
            expected_after,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
        and np.allclose(
            saved_trades["Cost Amount"],
            expected_cost,
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        )
    )
    recorder.add(
        "COST-02",
        "Trades and costs",
        "Exact 10 bps transaction cost plus 5 bps adverse slippage factors apply only to 0↔1 transitions.",
        factors_ok and cost_arithmetic_ok,
        {
            "entry_factor": ENTRY_FACTOR,
            "exit_factor": EXIT_FACTOR,
            "unchanged_state_cost": 0,
        },
        {
            "entry_rows": int(entry_rows.sum()),
            "exit_rows": int(exit_rows.sum()),
            "other_transition_rows": int(
                (~(entry_rows | exit_rows)).sum()
            ),
            "max_after_error": float(
                np.max(
                    np.abs(
                        saved_trades[
                            "Net Sleeve NAV After"
                        ].to_numpy(dtype=float)
                        - expected_after
                    )
                )
            ),
            "max_cost_error": float(
                np.max(
                    np.abs(
                        saved_trades["Cost Amount"].to_numpy(
                            dtype=float
                        )
                        - expected_cost
                    )
                )
            ),
        },
    )

    saved_metrics = pd.read_csv(HEADLINE_METRICS_PATH)
    saved_metrics["Path Start"] = pd.to_datetime(
        saved_metrics["Path Start"], errors="raise"
    )
    saved_metrics["Path End"] = pd.to_datetime(
        saved_metrics["Path End"], errors="raise"
    )
    independent_metrics["Path Start"] = pd.to_datetime(
        independent_metrics["Path Start"]
    )
    independent_metrics["Path End"] = pd.to_datetime(
        independent_metrics["Path End"]
    )
    metric_structure_ok = (
        len(saved_metrics) == 42
        and not saved_metrics.duplicated(
            ["Asset", "Period", "Strategy"]
        ).any()
        and set(saved_metrics["Strategy"]) == set(HEADLINE_STRATEGIES)
    )
    recorder.add(
        "METRIC-01",
        "Headline metrics",
        "The headline table contains all seven strategies for all six asset-period panels.",
        metric_structure_ok,
        "42 unique asset-period-strategy rows",
        {
            "rows": len(saved_metrics),
            "duplicates": int(
                saved_metrics.duplicated(
                    ["Asset", "Period", "Strategy"]
                ).sum()
            ),
        },
    )
    metric_numeric = [
        "Signal Rows",
        "Market Intervals",
        "Gross Total Return",
        "Net Total Return",
        "Cost Drag",
        "Annualised Net Return",
        "Annualised Volatility",
        "Maximum Drawdown",
        "Average Exposure",
        "One-way Sleeve Changes",
    ]
    metric_text = [
        "Family",
        "Path Start",
        "Path End",
        "Terminal All Cash",
    ]
    metric_match, metric_maximum, metric_problems = compare_keyed_frames(
        saved_metrics,
        independent_metrics,
        ["Asset", "Period", "Strategy"],
        metric_numeric,
        metric_text,
    )
    recorder.diagnostics["headline_metric_max_abs_difference"] = (
        metric_maximum
    )
    recorder.add(
        "METRIC-02",
        "Headline metrics",
        "Returns, annualisation, volatility, drawdown, exposure, cost drag, dates, and change counts independently reproduce.",
        metric_match,
        f"maximum numeric difference <= {ABSOLUTE_TOLERANCE}",
        {
            "max_abs_difference": metric_maximum,
            "problems": metric_problems,
        },
    )

    drawdown_mismatches = 0
    exposure_mismatches = 0
    for _, metric in saved_metrics.iterrows():
        subset = saved_equity.loc[
            saved_equity["Asset"].eq(metric["Asset"])
            & saved_equity["Period"].eq(metric["Period"])
            & saved_equity["Strategy"].eq(metric["Strategy"])
        ].sort_values("Open Date")
        path = np.r_[1.0, subset["Net NAV"].to_numpy(dtype=float)]
        recalculated_drawdown = maximum_drawdown(path)
        recalculated_exposure = float(
            subset.iloc[:-1]["Net Exposure"].mean()
        )
        if not np.isclose(
            recalculated_drawdown,
            float(metric["Maximum Drawdown"]),
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        ):
            drawdown_mismatches += 1
        if not np.isclose(
            recalculated_exposure,
            float(metric["Average Exposure"]),
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        ):
            exposure_mismatches += 1
    recorder.add(
        "METRIC-03",
        "Risk metrics",
        "Maximum drawdown uses the complete NAV path including initial capital, and Average Exposure uses interval-level value-weighted exposure.",
        drawdown_mismatches == 0 and exposure_mismatches == 0,
        "0 mismatches across 42 paths",
        {
            "drawdown_mismatches": drawdown_mismatches,
            "exposure_mismatches": exposure_mismatches,
        },
    )
    volatility_mismatches = 0
    volatility_interval_mismatches = 0
    for _, metric in saved_metrics.iterrows():
        subset = saved_equity.loc[
            saved_equity["Asset"].eq(metric["Asset"])
            & saved_equity["Period"].eq(metric["Period"])
            & saved_equity["Strategy"].eq(metric["Strategy"])
        ].sort_values("Open Date")
        market_interval_returns = (
            subset["Net NAV"]
            .astype(float)
            .pct_change()
            .dropna()
            .to_numpy()
        )
        if len(market_interval_returns) != int(
            metric["Market Intervals"]
        ):
            volatility_interval_mismatches += 1
            continue
        recalculated_volatility = float(
            np.std(market_interval_returns, ddof=1)
            * math.sqrt(SESSIONS_PER_YEAR)
        )
        if not np.isclose(
            recalculated_volatility,
            float(metric["Annualised Volatility"]),
            rtol=0.0,
            atol=ABSOLUTE_TOLERANCE,
        ):
            volatility_mismatches += 1
    recorder.add(
        "METRIC-04",
        "Risk metrics",
        "Annualised volatility uses only genuine equity open-to-open market intervals and excludes the initial same-open entry-cost observation.",
        volatility_mismatches == 0
        and volatility_interval_mismatches == 0,
        "pct_change(equity Net NAV) has exactly Market Intervals observations for every path",
        {
            "volatility_mismatches": volatility_mismatches,
            "interval_count_mismatches": volatility_interval_mismatches,
            "paths_checked": len(saved_metrics),
        },
    )

    buy_hold_rows = saved_metrics.loc[
        saved_metrics["Strategy"].eq("Buy-and-Hold")
    ]
    buy_hold_equity = saved_equity.loc[
        saved_equity["Strategy"].eq("Buy-and-Hold")
    ]
    buy_hold_ok = (
        len(buy_hold_rows) == 6
        and buy_hold_rows["One-way Sleeve Changes"].eq(2).all()
        and buy_hold_equity.groupby(["Asset", "Period"])[
            "Net Exposure"
        ]
        .apply(
            lambda values: bool(
                np.allclose(
                    values.iloc[:-1],
                    1.0,
                    rtol=0.0,
                    atol=ABSOLUTE_TOLERANCE,
                )
                and np.isclose(
                    values.iloc[-1],
                    0.0,
                    rtol=0.0,
                    atol=ABSOLUTE_TOLERANCE,
                )
            )
        )
        .all()
    )
    recorder.add(
        "BASE-01",
        "Buy-and-Hold",
        "Full Buy-and-Hold is an actual 100% stock path with entry and terminal exit friction.",
        buy_hold_ok,
        "6 paths; exposure 1 during all market intervals and 0 after terminal exit; two one-way changes",
        {
            "metric_rows": len(buy_hold_rows),
            "equity_rows": len(buy_hold_equity),
        },
    )

    # ------------------------------------------------------------------
    # Actual exposure-matched Buy-and-Hold, not alpha = target exposure
    # ------------------------------------------------------------------
    independent_matched_rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        for period in PERIODS:
            panel = independent_ledger.loc[
                independent_ledger["Asset"].eq(asset)
                & independent_ledger["Period"].eq(period)
                & independent_ledger["Five-Session Executable"]
            ]
            start_row = int(panel["Execution Price Row"].min())
            end_row = int(panel["End Price Row"].max())
            for model in MODELS:
                model_metric = independent_paths[
                    (asset, period, model)
                ].metric
                target_exposure = float(
                    model_metric["Average Exposure"]
                )
                alpha = independent_matched_alpha(
                    target_exposure,
                    asset,
                    period,
                    prices[asset],
                    start_row,
                    end_row,
                )
                matched_path = independent_buy_hold(
                    asset,
                    period,
                    prices[asset],
                    start_row,
                    end_row,
                    alpha,
                    f"Exposure-Matched Buy-and-Hold ({model})",
                )
                constant_exposure_path = (
                    independent_constant_exposure_passive(
                        prices[asset],
                        start_row,
                        end_row,
                        target_exposure,
                    )
                )
                independent_matched_rows.append(
                    {
                        "Asset": asset,
                        "Period": period,
                        "Model": model,
                        "Model Average Exposure": target_exposure,
                        "Matched Initial Stock Allocation": alpha,
                        "Matched Average Exposure": matched_path.metric[
                            "Average Exposure"
                        ],
                        "Model Gross Total Return": model_metric[
                            "Gross Total Return"
                        ],
                        "Matched Gross Total Return": matched_path.metric[
                            "Gross Total Return"
                        ],
                        "Gross Return Difference vs Static Matched B&H": model_metric[
                            "Gross Total Return"
                        ]
                        - matched_path.metric["Gross Total Return"],
                        "Model Net Total Return": model_metric[
                            "Net Total Return"
                        ],
                        "Matched Net Total Return": matched_path.metric[
                            "Net Total Return"
                        ],
                        "Net Return Difference vs Static Matched B&H": model_metric[
                            "Net Total Return"
                        ]
                        - matched_path.metric["Net Total Return"],
                        "Matched Maximum Drawdown": matched_path.metric[
                            "Maximum Drawdown"
                        ],
                        "Constant-Exposure Passive Gross Total Return": constant_exposure_path[
                            "Gross Total Return"
                        ],
                        "Constant-Exposure Passive Net Total Return": constant_exposure_path[
                            "Net Total Return"
                        ],
                        "Gross Timing Value vs Constant-Exposure Passive": model_metric[
                            "Gross Total Return"
                        ]
                        - constant_exposure_path["Gross Total Return"],
                        "Net Timing Value vs Constant-Exposure Passive": model_metric[
                            "Net Total Return"
                        ]
                        - constant_exposure_path["Net Total Return"],
                        "Constant-Exposure Passive Maximum Drawdown": constant_exposure_path[
                            "Maximum Drawdown"
                        ],
                        "Constant-Exposure Intermediate Rebalancing Costs Included": constant_exposure_path[
                            "Intermediate Rebalancing Costs Included"
                        ],
                        "Path Start": matched_path.metric["Path Start"],
                        "Path End": matched_path.metric["Path End"],
                        "Market Intervals": matched_path.metric[
                            "Market Intervals"
                        ],
                        "Interpretation": "Ex-post descriptive exposure match; not used for model tuning or selection",
                    }
                )
    independent_matched = pd.DataFrame(independent_matched_rows)
    saved_matched = pd.read_csv(MATCHED_PATH)
    for frame in [saved_matched, independent_matched]:
        frame["Path Start"] = pd.to_datetime(frame["Path Start"])
        frame["Path End"] = pd.to_datetime(frame["Path End"])
    matched_structure_ok = (
        len(saved_matched) == 24
        and not saved_matched.duplicated(
            ["Asset", "Period", "Model"]
        ).any()
    )
    recorder.add(
        "MATCH-01",
        "Exposure matching",
        "There is one unique dual exposure-reference result per model, asset, and period.",
        matched_structure_ok,
        "24 unique rows",
        {
            "rows": len(saved_matched),
            "duplicates": int(
                saved_matched.duplicated(
                    ["Asset", "Period", "Model"]
                ).sum()
            ),
        },
    )
    matched_numeric = [
        "Model Average Exposure",
        "Matched Initial Stock Allocation",
        "Matched Average Exposure",
        "Model Gross Total Return",
        "Matched Gross Total Return",
        "Gross Return Difference vs Static Matched B&H",
        "Model Net Total Return",
        "Matched Net Total Return",
        "Net Return Difference vs Static Matched B&H",
        "Matched Maximum Drawdown",
        "Constant-Exposure Passive Gross Total Return",
        "Constant-Exposure Passive Net Total Return",
        "Gross Timing Value vs Constant-Exposure Passive",
        "Net Timing Value vs Constant-Exposure Passive",
        "Constant-Exposure Passive Maximum Drawdown",
        "Market Intervals",
    ]
    matched_text = [
        "Constant-Exposure Intermediate Rebalancing Costs Included",
        "Path Start",
        "Path End",
        "Interpretation",
    ]
    matched_match, matched_maximum, matched_problems = compare_keyed_frames(
        saved_matched,
        independent_matched,
        ["Asset", "Period", "Model"],
        matched_numeric,
        matched_text,
    )
    recorder.diagnostics[
        "exposure_matched_max_abs_difference"
    ] = matched_maximum
    recorder.add(
        "MATCH-02",
        "Exposure matching",
        "Static and constant-exposure reference returns, drawdowns, differences, and timing values independently reproduce.",
        matched_match,
        f"maximum numeric difference <= {ABSOLUTE_TOLERANCE}",
        {
            "max_abs_difference": matched_maximum,
            "problems": matched_problems,
        },
    )
    match_error = np.abs(
        saved_matched["Model Average Exposure"].to_numpy(dtype=float)
        - saved_matched["Matched Average Exposure"].to_numpy(dtype=float)
    )
    alpha_gap = np.abs(
        saved_matched["Matched Initial Stock Allocation"].to_numpy(
            dtype=float
        )
        - saved_matched["Model Average Exposure"].to_numpy(dtype=float)
    )
    actual_matching_ok = (
        float(match_error.max()) <= 1e-10
        and float(alpha_gap.max()) > 1e-6
    )
    recorder.add(
        "MATCH-03",
        "Exposure matching",
        "The benchmark matches actual value-weighted path exposure; it does not simply set alpha equal to the target mean exposure.",
        actual_matching_ok,
        "maximum exposure error <= 1e-10 and at least one non-trivial alpha adjustment",
        {
            "max_exposure_error": float(match_error.max()),
            "max_alpha_minus_target": float(alpha_gap.max()),
        },
    )
    static_net_identity = (
        saved_matched["Model Net Total Return"].to_numpy(dtype=float)
        - saved_matched["Matched Net Total Return"].to_numpy(dtype=float)
    )
    static_gross_identity = (
        saved_matched["Model Gross Total Return"].to_numpy(dtype=float)
        - saved_matched["Matched Gross Total Return"].to_numpy(dtype=float)
    )
    static_net_ok, static_net_difference = numeric_difference(
        saved_matched[
            "Net Return Difference vs Static Matched B&H"
        ],
        static_net_identity,
    )
    static_gross_ok, static_gross_difference = numeric_difference(
        saved_matched[
            "Gross Return Difference vs Static Matched B&H"
        ],
        static_gross_identity,
    )
    recorder.add(
        "MATCH-04",
        "Exposure matching",
        "Static matched Buy-and-Hold fields are explicitly return differences, not labelled as pure timing value.",
        static_net_ok and static_gross_ok,
        f"identity holds within {ABSOLUTE_TOLERANCE}",
        {
            "net_max_abs_difference": static_net_difference,
            "gross_max_abs_difference": static_gross_difference,
        },
    )
    constant_cost_flags = normalise_bool(
        saved_matched[
            "Constant-Exposure Intermediate Rebalancing Costs Included"
        ]
    )
    constant_reference_ok = (
        not constant_cost_flags.any()
        and protocol["exposure_matching"][
            "constant_exposure_reference"
        ].find("omit intermediate rebalancing costs")
        >= 0
    )
    recorder.add(
        "MATCH-05",
        "Constant-exposure passive",
        "The constant-exposure passive compounds every open-to-open interval and charges only exposed-fraction entry and terminal costs.",
        constant_reference_ok and matched_match,
        "no intermediate rebalancing costs; all independently rebuilt fields match",
        {
            "rows_with_intermediate_costs": int(
                constant_cost_flags.sum()
            ),
            "entry_cost_scope": "exposed fraction",
            "terminal_cost_scope": "exposed fraction",
        },
    )
    constant_net_identity = (
        saved_matched["Model Net Total Return"].to_numpy(dtype=float)
        - saved_matched[
            "Constant-Exposure Passive Net Total Return"
        ].to_numpy(dtype=float)
    )
    constant_gross_identity = (
        saved_matched["Model Gross Total Return"].to_numpy(dtype=float)
        - saved_matched[
            "Constant-Exposure Passive Gross Total Return"
        ].to_numpy(dtype=float)
    )
    constant_net_ok, constant_net_difference = numeric_difference(
        saved_matched[
            "Net Timing Value vs Constant-Exposure Passive"
        ],
        constant_net_identity,
    )
    constant_gross_ok, constant_gross_difference = numeric_difference(
        saved_matched[
            "Gross Timing Value vs Constant-Exposure Passive"
        ],
        constant_gross_identity,
    )
    recorder.add(
        "MATCH-06",
        "Constant-exposure passive",
        "Timing value is exactly model return minus constant-exposure passive return in both gross and net terms.",
        constant_net_ok and constant_gross_ok,
        f"identities hold within {ABSOLUTE_TOLERANCE}",
        {
            "net_max_abs_difference": constant_net_difference,
            "gross_max_abs_difference": constant_gross_difference,
        },
    )

    # The convenient model-results table must be a faithful merge, not a
    # separately calculated or cherry-picked result.
    saved_model_results = pd.read_csv(MODEL_RESULTS_PATH)
    expected_model_results = saved_metrics.loc[
        saved_metrics["Strategy"].isin(MODELS)
    ].merge(
        saved_matched[
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
    expected_model_results["Return Rank within Asset-Period"] = (
        expected_model_results.groupby(["Asset", "Period"])[
            "Net Total Return"
        ]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    model_numeric = [
        "Net Total Return",
        "Maximum Drawdown",
        "Average Exposure",
        "Matched Initial Stock Allocation",
        "Matched Net Total Return",
        "Net Return Difference vs Static Matched B&H",
        "Constant-Exposure Passive Net Total Return",
        "Net Timing Value vs Constant-Exposure Passive",
        "Return Rank within Asset-Period",
    ]
    model_match, model_maximum, model_problems = compare_keyed_frames(
        saved_model_results,
        expected_model_results,
        ["Asset", "Period", "Strategy"],
        model_numeric,
        ["Model"],
    )
    recorder.add(
        "MATCH-07",
        "Model results",
        "The model-results table faithfully merges headline and matched results and ranks all four models within each panel.",
        model_match and len(saved_model_results) == 24,
        "24 rows; no recomputation or model omission",
        {
            "rows": len(saved_model_results),
            "max_abs_difference": model_maximum,
            "problems": model_problems,
        },
    )

    # ------------------------------------------------------------------
    # Presentation artifacts and explicitly excluded extensions
    # ------------------------------------------------------------------
    figure_results = {
        filename: validate_png(FIGURES / filename)
        for filename in EXPECTED_FIGURES
    }
    figures_ok = all(result[0] for result in figure_results.values())
    recorder.add(
        "PRES-01",
        "Figures",
        "All four predeclared RQ4 comparison charts are valid, non-placeholder PNG files.",
        figures_ok,
        EXPECTED_FIGURES,
        {
            filename: details
            for filename, (_, details) in figure_results.items()
        },
        severity="Major",
    )

    formal_csvs = [
        SIGNAL_LEDGER_PATH,
        HEADLINE_METRICS_PATH,
        MODEL_RESULTS_PATH,
        EQUITY_PATH,
        TRADES_PATH,
        MATCHED_PATH,
    ]
    prohibited_statistical_columns: list[str] = []
    for path in formal_csvs:
        columns = [str(column).lower() for column in pd.read_csv(
            path, nrows=0
        ).columns]
        for column in columns:
            if (
                "bootstrap" in column
                or "p-value" in column
                or "p value" in column
                or "holm" in column
                or "significance" in column
            ):
                prohibited_statistical_columns.append(
                    f"{path.name}:{column}"
                )
    bootstrap_files = [
        str(path.relative_to(ROOT))
        for path in [*OUTPUTS.glob("*"), *FIGURES.glob("*")]
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ["bootstrap", "p_value", "p-value", "holm"]
        )
    ]
    no_bootstrap_ok = (
        protocol["research_positioning"]["statistical_inference"].startswith(
            "No RQ4 bootstrap"
        )
        and metadata["bootstrap_used"] is False
        and not prohibited_statistical_columns
        and not bootstrap_files
    )
    recorder.add(
        "SCOPE-01",
        "Research scope",
        "RQ4 remains descriptive: no bootstrap, Holm adjustment, p-value, or significance output is present.",
        no_bootstrap_ok,
        "protocol and metadata explicitly false; no inferential result files or columns",
        {
            "metadata_bootstrap_used": metadata["bootstrap_used"],
            "prohibited_columns": prohibited_statistical_columns,
            "prohibited_files": bootstrap_files,
        },
    )

    retraining_files = [
        str(path.relative_to(ROOT))
        for path in [*OUTPUTS.glob("*"), *FIGURES.glob("*")]
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ["quarterly", "rolling_retrain", "rolling-retrain"]
        )
    ]
    no_retraining_ok = (
        protocol["research_positioning"]["excluded_extension"]
        == "No static-model versus quarterly rolling-retraining comparison is included."
        and metadata["quarterly_rolling_retraining_used"] is False
        and not retraining_files
    )
    recorder.add(
        "SCOPE-02",
        "Research scope",
        "No static-model versus quarterly rolling-retraining extension is included.",
        no_retraining_ok,
        "explicitly excluded in protocol and metadata; no retraining artifacts",
        {
            "metadata_quarterly_rolling_retraining_used": metadata[
                "quarterly_rolling_retraining_used"
            ],
            "retraining_files": retraining_files,
        },
    )
    no_return_selection_ok = (
        protocol["upstream_prediction_protocol"][
            "return_metrics_used_for_training_or_selection"
        ]
        is False
        and prediction_manifest[
            "return_metrics_used_for_training_or_selection"
        ]
        is False
        and metadata[
            "return_metrics_used_for_model_training_or_selection"
        ]
        is False
    )
    recorder.add(
        "SCOPE-03",
        "Research scope",
        "RQ4 returns did not train, tune, threshold, or select any predictive model.",
        no_return_selection_ok,
        "false in protocol, prediction manifest, and experiment metadata",
        {
            "protocol": protocol["upstream_prediction_protocol"][
                "return_metrics_used_for_training_or_selection"
            ],
            "prediction_manifest": prediction_manifest[
                "return_metrics_used_for_training_or_selection"
            ],
            "metadata": metadata[
                "return_metrics_used_for_model_training_or_selection"
            ],
        },
    )

    # ------------------------------------------------------------------
    # Persist the independent validation artifacts before returning status
    # ------------------------------------------------------------------
    checks = pd.DataFrame(recorder.rows)
    passed = int(checks["Status"].eq("Pass").sum())
    failed = int(checks["Status"].eq("Fail").sum())
    report = {
        "validator": "Independent five-session cross-asset RQ4 validator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "overall_status": "PASS" if failed == 0 else "FAIL",
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": failed,
        "artifacts_written": bool(write_artifacts),
        "independence_statement": (
            "validate_rq4.py does not import run_rq4_experiment. Target "
            "labels, signal dates, five sleeves, frictions, NAVs, risk "
            "metrics, Buy-and-Hold, and matched allocations were "
            "recomputed from frozen source files."
        ),
        "coverage": {
            "assets": ASSETS,
            "periods": PERIODS,
            "models": MODELS,
            "moving_average_baselines": MOVING_AVERAGES,
            "combined_prediction_rows": len(predictions),
            "unique_signal_origins": len(independent_ledger),
            "executable_signal_origins": int(
                independent_ledger[
                    "Five-Session Executable"
                ].sum()
            ),
            "headline_strategy_paths": len(saved_metrics),
            "equity_rows": len(saved_equity),
            "sleeve_transition_rows": len(saved_trades),
            "exposure_matched_rows": len(saved_matched),
        },
        "numerical_absolute_tolerance": ABSOLUTE_TOLERANCE,
        "diagnostics": recorder.diagnostics,
        "scope_assertions": {
            "rq4_bootstrap_used": False,
            "quarterly_rolling_retraining_used": False,
            "return_metrics_used_for_model_training_or_selection": False,
        },
        "failed_check_ids": checks.loc[
            checks["Status"].eq("Fail"), "Check ID"
        ].tolist(),
        "artifacts": {
            "checks_csv": str(CHECKS_PATH.relative_to(ROOT)),
            "report_json": str(REPORT_PATH.relative_to(ROOT)),
        },
    }
    if write_artifacts:
        checks.to_csv(CHECKS_PATH, index=False)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        refresh_result_manifest()
    result_manifest_verified, result_manifest_files = verify_result_manifest()
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "checks": f"{passed}/{len(checks)}",
                "artifacts_written": bool(write_artifacts),
                "result_manifest_verified": result_manifest_verified,
                "result_manifest_files": result_manifest_files,
                "checks_csv": str(CHECKS_PATH),
                "report_json": str(REPORT_PATH),
            },
            indent=2,
        ),
        flush=True,
    )
    if failed or not result_manifest_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate saved RQ4 evidence. The default mode is read-only."
        )
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Replace the saved validation CSV and JSON after recomputation.",
    )
    arguments = parser.parse_args()
    main(write_artifacts=arguments.write_artifacts)
