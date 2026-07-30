"""Tests for the v8 exploratory five-session downstream engine.

These tests intentionally do not exercise the retained v7 one-session
``src.backtesting`` compatibility module.  The v8 Sandbox converts information
available after Close[t] into five equal sleeves that first trade at Open[t+1]
and revisit the same sleeve at Open[t+6].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v8_sandbox import (
    BUY_HOLD_STRATEGY,
    EMA_STRATEGY,
    EXPLORATORY_SCOPE,
    FIVE_SESSION_TARGET_COLUMN,
    FIVE_SESSION_TARGET_DATE_COLUMN,
    FUTURE_MEAN_COLUMN,
    PAST_MEAN_COLUMN,
    SMA_STRATEGY,
    add_five_session_mean_target,
    run_five_session_sandbox,
)


PREDICTION_STRATEGY = "Prediction Strategy"


def make_prices(rows: int = 48, *, flat_open: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    if flat_open:
        opens = np.full(rows, 100.0)
    else:
        opens = 100.0 + np.arange(rows, dtype=float)
    closes = opens + 0.5 * np.sin(np.arange(rows, dtype=float) / 2.0)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": np.maximum(opens, closes) + 1.0,
            "Low": np.minimum(opens, closes) - 1.0,
            "Close": closes,
            "Volume": 1_000_000 + np.arange(rows),
        }
    )


def make_predictions(
    prices: pd.DataFrame,
    rows: list[int],
    signals: list[int] | None = None,
) -> pd.DataFrame:
    if signals is None:
        signals = [1] * len(rows)
    return pd.DataFrame(
        {
            "Date": prices.iloc[rows]["Date"].to_numpy(),
            "Signal": signals,
            "Predicted Probability": np.where(
                np.asarray(signals) == 1,
                0.80,
                0.20,
            ),
        }
    )


def strategy_metric(result, strategy: str) -> pd.Series:
    return result.metrics.set_index("Strategy").loc[strategy]


def strategy_equity(result, strategy: str) -> pd.DataFrame:
    return (
        result.equity_rows.loc[result.equity_rows["Strategy"].eq(strategy)]
        .sort_values("Price Row")
        .reset_index(drop=True)
    )


def strategy_trades(result, strategy: str) -> pd.DataFrame:
    return (
        result.sleeve_trade_ledger.loc[
            result.sleeve_trade_ledger["Strategy"].eq(strategy)
        ]
        .sort_values("Price Row")
        .reset_index(drop=True)
    )


def test_five_session_mean_target_matches_close_windows_and_t_plus_five_date():
    prices = make_prices(35)
    targeted = add_five_session_mean_target(prices)
    row = 12

    expected_past = prices.loc[row - 4 : row, "Close"].mean()
    expected_future = prices.loc[row + 1 : row + 5, "Close"].mean()
    expected_target = int(expected_future > expected_past)

    assert targeted.loc[row, PAST_MEAN_COLUMN] == pytest.approx(expected_past)
    assert targeted.loc[row, FUTURE_MEAN_COLUMN] == pytest.approx(expected_future)
    assert targeted.loc[row, FIVE_SESSION_TARGET_COLUMN] == expected_target
    assert (
        targeted.loc[row, FIVE_SESSION_TARGET_DATE_COLUMN]
        == prices.loc[row + 5, "Date"]
    )

    eligible_indices = targeted.index[
        targeted[FIVE_SESSION_TARGET_COLUMN].notna()
    ]
    assert eligible_indices.min() == 4
    assert eligible_indices.max() == len(prices) - 6

    # The classification label is defined from Close only; changing Open must
    # not change it.
    altered_open = prices.copy()
    altered_open["Open"] = altered_open["Open"] * 7.0
    altered_target = add_five_session_mean_target(altered_open)
    pd.testing.assert_series_equal(
        targeted[FIVE_SESSION_TARGET_COLUMN],
        altered_target[FIVE_SESSION_TARGET_COLUMN],
    )


def test_close_t_signal_executes_at_open_t_plus_one_and_ends_open_t_plus_six():
    prices = make_prices(40, flat_open=True)
    signal_row = 20
    execution_row = signal_row + 1
    end_row = signal_row + 6

    # A dramatic same-close value must not be treated as an executable fill.
    prices.loc[signal_row, "Close"] = 10_000.0
    prices.loc[execution_row, "Open"] = 120.0
    prices.loc[end_row, "Open"] = 108.0
    predictions = make_predictions(prices, [signal_row], [1])

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    signal = result.signal_ledger.iloc[0]

    assert signal["Signal Date"] == prices.loc[signal_row, "Date"]
    assert signal["Execution Date"] == prices.loc[execution_row, "Date"]
    assert signal["Scheduled End Date"] == prices.loc[end_row, "Date"]
    assert signal["Signal Date"] < signal["Execution Date"]
    assert signal["Execution Date"] < signal["Scheduled End Date"]
    assert signal["Execution Price Row"] == execution_row
    assert signal["End Price Row"] == end_row

    trades = strategy_trades(result, PREDICTION_STRATEGY)
    assert trades["Price Row"].tolist() == [execution_row, end_row]
    assert trades["Transition"].tolist() == ["0->1", "1->0"]
    assert (trades["Origin Signal Date"] < trades["Open Date"]).all()

    # Only one 20% sleeve is invested; the other four remain cash.
    expected_final_nav = 0.8 + 0.2 * (108.0 / 120.0)
    observed_return = strategy_metric(
        result,
        PREDICTION_STRATEGY,
    )["Net Total Return"]
    assert 1.0 + observed_return == pytest.approx(expected_final_nav)


def test_five_daily_signals_form_five_independent_twenty_percent_sleeves():
    prices = make_prices(flat_open=True)
    predictions = make_predictions(prices, [19, 20, 21, 22, 23])

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    trades = strategy_trades(result, PREDICTION_STRATEGY)
    entries = trades.loc[trades["Transition"].eq("0->1")]
    exits = trades.loc[trades["Transition"].eq("1->0")]

    assert set(entries["Sleeve"]) == {0, 1, 2, 3, 4}
    assert len(entries) == len(exits) == 5
    for _, entry in entries.iterrows():
        exit_row = exits.loc[exits["Sleeve"].eq(entry["Sleeve"])].iloc[0]
        assert exit_row["Price Row"] - entry["Price Row"] == 5
        assert entry["Net Sleeve NAV Before"] == pytest.approx(0.2)

    equity = strategy_equity(result, PREDICTION_STRATEGY).set_index("Price Row")
    assert equity.loc[[20, 21, 22, 23, 24], "Active Sleeves"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert equity.loc[[25, 26, 27, 28, 29], "Active Sleeves"].tolist() == [
        4,
        3,
        2,
        1,
        0,
    ]
    assert equity.loc[20, "Net Exposure"] == pytest.approx(0.2)
    assert equity.loc[24, "Net Exposure"] == pytest.approx(1.0)
    assert strategy_metric(
        result,
        PREDICTION_STRATEGY,
    )["Average Exposure"] == pytest.approx(5.0 / 9.0)


def test_same_sleeve_one_to_one_revisit_is_not_charged_twice():
    prices = make_prices()
    # Executions at rows 20 and 25 revisit the same modulo-five sleeve.
    predictions = make_predictions(prices, [19, 24], [1, 1])
    transaction_cost_bps = 10.0
    slippage_bps = 5.0

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    trades = strategy_trades(result, PREDICTION_STRATEGY)

    assert trades["Transition"].tolist() == ["0->1", "1->0"]
    assert trades["Price Row"].tolist() == [20, 30]
    assert 25 not in set(trades["Price Row"])

    transaction_rate = transaction_cost_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    entry_factor = 1.0 / (
        (1.0 + transaction_rate) * (1.0 + slippage_rate)
    )
    exit_factor = (1.0 - transaction_rate) * (1.0 - slippage_rate)
    price_ratio = prices.loc[30, "Open"] / prices.loc[20, "Open"]
    expected_gross_nav = 0.8 + 0.2 * price_ratio
    expected_net_nav = 0.8 + 0.2 * entry_factor * price_ratio * exit_factor

    metric = strategy_metric(result, PREDICTION_STRATEGY)
    assert 1.0 + metric["Gross Total Return"] == pytest.approx(
        expected_gross_nav
    )
    assert 1.0 + metric["Net Total Return"] == pytest.approx(expected_net_nav)
    assert metric["Cost Drag (pp)"] == pytest.approx(
        (expected_gross_nav - expected_net_nav) * 100.0
    )


def test_noncontiguous_prediction_subset_closes_at_each_unreplaced_t_plus_six():
    prices = make_prices()
    # These executions (20 and 30) share a sleeve but have a five-session gap.
    # The first exposure must close at row 25, not remain open until row 30.
    predictions = make_predictions(prices, [19, 29], [1, 1])

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    trades = strategy_trades(result, PREDICTION_STRATEGY)

    assert trades["Price Row"].tolist() == [20, 25, 30, 35]
    assert trades["Transition"].tolist() == ["0->1", "1->0", "0->1", "1->0"]
    equity = strategy_equity(result, PREDICTION_STRATEGY).set_index("Price Row")
    assert equity.loc[25, "Active Sleeves"] == 0
    assert equity.loc[29, "Active Sleeves"] == 0
    assert equity.loc[30, "Active Sleeves"] == 1
    assert equity.loc[35, "Active Sleeves"] == 0


def test_baseline_states_use_signal_close_and_share_one_evaluation_grid():
    prices = make_prices()
    signal_rows = [19, 20, 21, 22, 23]
    ema_states = np.array([1, 0, 1, 0, 1])
    sma_states = np.array([0, 1, 0, 1, 0])

    # Explicit values make the intended state independent of price direction.
    prices["EMA8"] = 90.0
    prices["EMA13"] = 100.0
    prices["SMA10"] = 90.0
    prices["SMA20"] = 100.0
    prices.loc[signal_rows, "EMA8"] = np.where(ema_states == 1, 110.0, 90.0)
    prices.loc[signal_rows, "EMA13"] = 100.0
    prices.loc[signal_rows, "SMA10"] = np.where(sma_states == 1, 110.0, 90.0)
    prices.loc[signal_rows, "SMA20"] = 100.0
    predictions = make_predictions(prices, signal_rows, [1, 0, 1, 0, 1])

    result = run_five_session_sandbox(prices, predictions)
    ledger = result.signal_ledger

    assert ledger["EMA8/EMA13 State Signal"].tolist() == ema_states.tolist()
    assert ledger["SMA10/SMA20 State Signal"].tolist() == sma_states.tolist()
    assert set(result.metrics["Strategy"]) == {
        PREDICTION_STRATEGY,
        EMA_STRATEGY,
        SMA_STRATEGY,
        BUY_HOLD_STRATEGY,
    }
    assert result.metrics["Scope"].eq(EXPLORATORY_SCOPE).all()

    path_ranges = result.metrics.set_index("Strategy")[["Path Start", "Path End"]]
    assert path_ranges["Path Start"].nunique() == 1
    assert path_ranges["Path End"].nunique() == 1

    buy_hold_metric = strategy_metric(result, BUY_HOLD_STRATEGY)
    assert buy_hold_metric["Average Exposure"] == pytest.approx(1.0)
    assert buy_hold_metric["One-way Position Changes"] == 2
    assert strategy_trades(
        result,
        BUY_HOLD_STRATEGY,
    )["Transition"].tolist() == ["0->1", "1->0"]


def test_cost_metrics_match_independent_buy_and_hold_calculation():
    prices = make_prices()
    predictions = make_predictions(prices, [19, 20, 21], [1, 0, 1])
    transaction_cost_bps = 20.0
    slippage_bps = 10.0

    zero_cost = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    with_cost = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    assert np.allclose(zero_cost.metrics["Cost Drag (pp)"], 0.0)

    start_row = 20
    end_row = 27
    gross_nav = prices.loc[end_row, "Open"] / prices.loc[start_row, "Open"]
    transaction_rate = transaction_cost_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    entry_factor = 1.0 / (
        (1.0 + transaction_rate) * (1.0 + slippage_rate)
    )
    exit_factor = (1.0 - transaction_rate) * (1.0 - slippage_rate)
    expected_net_nav = entry_factor * gross_nav * exit_factor

    metric = strategy_metric(with_cost, BUY_HOLD_STRATEGY)
    assert 1.0 + metric["Gross Total Return"] == pytest.approx(gross_nav)
    assert 1.0 + metric["Net Total Return"] == pytest.approx(expected_net_nav)
    assert metric["Cost Drag (pp)"] == pytest.approx(
        (gross_nav - expected_net_nav) * 100.0
    )
    assert metric["Net Total Return"] < strategy_metric(
        zero_cost,
        BUY_HOLD_STRATEGY,
    )["Net Total Return"]


def test_terminal_signals_without_t_plus_six_are_excluded_not_truncated():
    prices = make_prices(42)
    # Row 35 is executable (end row 41); row 36 would require missing row 42.
    predictions = make_predictions(prices, [19, 35, 36], [1, 1, 1])

    result = run_five_session_sandbox(prices, predictions)
    ledger = result.signal_ledger.set_index("Signal Date")

    included = ledger.loc[prices.loc[35, "Date"]]
    excluded = ledger.loc[prices.loc[36, "Date"]]
    assert bool(included["Five-Session Executable"])
    assert included["Scheduled End Date"] == prices.loc[41, "Date"]
    assert not bool(excluded["Five-Session Executable"])
    assert excluded["Exclusion Reason"] == "missing Open[t+6]"
    assert pd.isna(excluded["Execution Date"])
    assert pd.isna(excluded["Scheduled End Date"])
    assert len(result.excluded_signals) == 1
    assert result.metrics["Executable Prediction Dates"].eq(2).all()
    assert result.metrics["Excluded Prediction Dates"].eq(1).all()


def test_all_strategies_finish_in_cash_and_reconcile_sleeve_nav():
    prices = make_prices()
    predictions = make_predictions(
        prices,
        [19, 20, 24, 25, 29],
        [1, 0, 1, 1, 0],
    )

    result = run_five_session_sandbox(prices, predictions)
    assert result.metrics["Terminal All Cash"].all()

    terminal_rows = (
        result.equity_rows.sort_values(["Strategy", "Price Row"])
        .groupby("Strategy", as_index=False)
        .tail(1)
    )
    assert terminal_rows["Active Sleeves"].eq(0).all()
    assert terminal_rows["Net Exposure"].eq(0.0).all()

    for strategy in [PREDICTION_STRATEGY, EMA_STRATEGY, SMA_STRATEGY]:
        terminal = terminal_rows.loc[
            terminal_rows["Strategy"].eq(strategy)
        ].iloc[0]
        sleeve_nav = sum(
            float(terminal[f"Sleeve {sleeve} Net NAV"])
            for sleeve in range(5)
        )
        assert sleeve_nav == pytest.approx(terminal["Net NAV"])
        assert all(
            int(terminal[f"Sleeve {sleeve} Position"]) == 0
            for sleeve in range(5)
        )


def test_future_close_tampering_cannot_change_an_earlier_signal_or_return():
    prices = make_prices()
    predictions = make_predictions(prices, [22], [1])
    baseline = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    future_tampered = prices.copy()
    future_tampered.loc[23:, "Close"] *= 100.0
    rerun = run_five_session_sandbox(
        future_tampered,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )

    comparison_columns = [
        "Prediction Signal",
        "EMA8/EMA13 State Signal",
        "SMA10/SMA20 State Signal",
        "Execution Date",
        "Scheduled End Date",
    ]
    pd.testing.assert_frame_equal(
        baseline.signal_ledger[comparison_columns],
        rerun.signal_ledger[comparison_columns],
    )
    pd.testing.assert_series_equal(
        baseline.metrics.set_index("Strategy")["Net Total Return"].sort_index(),
        rerun.metrics.set_index("Strategy")["Net Total Return"].sort_index(),
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("absent_prediction_date", "absent from the price calendar"),
        ("duplicate_price_date", "duplicate trading dates"),
        ("nonbinary_signal", "binary"),
        ("invalid_probability", r"finite in \[0, 1\]"),
    ],
)
def test_tampered_or_noncausal_inputs_are_rejected(tamper: str, message: str):
    prices = make_prices()
    predictions = make_predictions(prices, [19, 20], [1, 0])

    if tamper == "absent_prediction_date":
        predictions.loc[0, "Date"] = pd.Timestamp("2030-01-01")
    elif tamper == "duplicate_price_date":
        prices.loc[20, "Date"] = prices.loc[19, "Date"]
    elif tamper == "nonbinary_signal":
        predictions.loc[0, "Signal"] = 2
    elif tamper == "invalid_probability":
        predictions.loc[0, "Predicted Probability"] = 1.2

    with pytest.raises(ValueError, match=message):
        run_five_session_sandbox(prices, predictions)


@pytest.mark.parametrize(
    ("transaction_cost_bps", "slippage_bps"),
    [
        (-1, 0),
        (0, -1),
        (10_000, 0),
        (0, 10_000),
    ],
)
def test_invalid_friction_assumptions_are_rejected(
    transaction_cost_bps: float,
    slippage_bps: float,
):
    prices = make_prices()
    predictions = make_predictions(prices, [19])

    with pytest.raises(ValueError, match=r"\[0, 10,000\) bps"):
        run_five_session_sandbox(
            prices,
            predictions,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
