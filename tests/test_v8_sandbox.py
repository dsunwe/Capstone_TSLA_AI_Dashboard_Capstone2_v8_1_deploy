from __future__ import annotations

import numpy as np
import pandas as pd

from src.v8_sandbox import (
    BUY_HOLD_STRATEGY,
    EMA_STRATEGY,
    SMA_STRATEGY,
    add_five_session_mean_target,
    run_five_session_sandbox,
)


def price_frame(rows: int = 42) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    opens = 100.0 + np.arange(rows, dtype=float)
    closes = opens + np.sin(np.arange(rows, dtype=float) / 2.0)
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


def prediction_frame(
    prices: pd.DataFrame,
    row_numbers: list[int],
    signals: list[int] | None = None,
) -> pd.DataFrame:
    if signals is None:
        signals = [1] * len(row_numbers)
    return pd.DataFrame(
        {
            "Date": prices.iloc[row_numbers]["Date"].to_numpy(),
            "Signal": signals,
            "Predicted Probability": np.where(
                np.asarray(signals) == 1,
                0.8,
                0.2,
            ),
        }
    )


def test_date_causality_and_missing_final_endpoint_are_auditable():
    prices = price_frame(42)
    # Row 36 has t+1 but no t+6 (42 is beyond the final row 41).
    predictions = prediction_frame(prices, [19, 22, 36])

    result = run_five_session_sandbox(prices, predictions)
    ledger = result.signal_ledger.set_index("Signal Date")

    for signal_row in [19, 22]:
        row = ledger.loc[prices.loc[signal_row, "Date"]]
        assert row["Execution Date"] == prices.loc[signal_row + 1, "Date"]
        assert row["Scheduled End Date"] == prices.loc[signal_row + 6, "Date"]
        assert row["Signal Price Row"] == signal_row
        assert row["Execution Price Row"] == signal_row + 1
        assert row["End Price Row"] == signal_row + 6
        assert row["Sleeve"] == (signal_row + 1) % 5
        assert prices.loc[signal_row, "Date"] < row["Execution Date"]
        assert row["Execution Date"] < row["Scheduled End Date"]

    excluded = ledger.loc[prices.loc[36, "Date"]]
    assert not bool(excluded["Five-Session Executable"])
    assert excluded["Exclusion Reason"] == "missing Open[t+6]"
    assert len(result.excluded_signals) == 1

    trade_rows = result.sleeve_trade_ledger
    causal_trades = trade_rows.loc[
        trade_rows["Strategy"].ne(BUY_HOLD_STRATEGY)
    ]
    assert (
        causal_trades["Origin Signal Date"] < causal_trades["Open Date"]
    ).all()


def test_five_daily_signals_use_five_independent_twenty_percent_sleeves():
    prices = price_frame()
    predictions = prediction_frame(prices, [19, 20, 21, 22, 23])

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=0,
        slippage_bps=0,
    )
    prediction_trades = result.sleeve_trade_ledger.loc[
        result.sleeve_trade_ledger["Strategy"].eq("Prediction Strategy")
    ]
    entries = prediction_trades.loc[prediction_trades["Transition"].eq("0->1")]
    exits = prediction_trades.loc[prediction_trades["Transition"].eq("1->0")]

    assert set(entries["Sleeve"]) == {0, 1, 2, 3, 4}
    assert len(entries) == len(exits) == 5
    for _, entry in entries.iterrows():
        matching_exit = exits.loc[exits["Sleeve"].eq(entry["Sleeve"])].iloc[0]
        assert matching_exit["Price Row"] - entry["Price Row"] == 5

    equity = result.equity_rows.loc[
        result.equity_rows["Strategy"].eq("Prediction Strategy")
    ]
    first_entry_row = equity.loc[equity["Price Row"].eq(20)].iloc[0]
    assert first_entry_row["Active Sleeves"] == 1
    assert np.isclose(first_entry_row["Net Exposure"], 0.2)


def test_continuous_one_to_one_revisit_has_no_duplicate_fee():
    prices = price_frame()
    # Both signals belong to the same sleeve: executions are rows 20 and 25.
    predictions = prediction_frame(prices, [19, 24], [1, 1])
    transaction_cost_bps = 10.0
    slippage_bps = 5.0

    result = run_five_session_sandbox(
        prices,
        predictions,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    trades = result.sleeve_trade_ledger.loc[
        result.sleeve_trade_ledger["Strategy"].eq("Prediction Strategy")
    ]

    assert trades["Transition"].tolist() == ["0->1", "1->0"]
    assert trades["Price Row"].tolist() == [20, 30]
    assert 25 not in set(trades["Price Row"])

    transaction_rate = transaction_cost_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    entry_factor = 1.0 / (
        (1.0 + transaction_rate) * (1.0 + slippage_rate)
    )
    exit_factor = (1.0 - transaction_rate) * (1.0 - slippage_rate)
    expected_final_nav = (
        0.8
        + 0.2
        * entry_factor
        * (prices.loc[30, "Open"] / prices.loc[20, "Open"])
        * exit_factor
    )
    observed = result.metrics.set_index("Strategy").loc[
        "Prediction Strategy",
        "Net Total Return",
    ]
    assert np.isclose(1.0 + observed, expected_final_nav, atol=1e-12)


def test_every_strategy_is_terminal_cash_and_comparators_are_returned():
    prices = price_frame()
    predictions = prediction_frame(
        prices,
        [19, 20, 24, 25, 29],
        [1, 0, 1, 1, 0],
    )
    result = run_five_session_sandbox(prices, predictions)

    assert set(result.metrics["Strategy"]) == {
        "Prediction Strategy",
        EMA_STRATEGY,
        SMA_STRATEGY,
        BUY_HOLD_STRATEGY,
    }
    assert result.metrics["Terminal All Cash"].all()
    assert {
        "Net Total Return",
        "Annualized Net Return",
        "Maximum Drawdown",
        "Average Exposure",
        "Cost Drag (pp)",
    }.issubset(result.metrics.columns)

    terminal = (
        result.equity_rows.sort_values(["Strategy", "Open Date"])
        .groupby("Strategy", as_index=False)
        .tail(1)
    )
    assert terminal["Active Sleeves"].eq(0).all()
    assert terminal["Net Exposure"].eq(0.0).all()


def test_target_helper_uses_future_five_close_mean_and_t_plus_five_date():
    prices = price_frame(30)
    targeted = add_five_session_mean_target(prices)
    row = 10

    expected_past = prices.loc[row - 4 : row, "Close"].mean()
    expected_future = prices.loc[row + 1 : row + 5, "Close"].mean()
    assert np.isclose(targeted.loc[row, "Past 5-Session Mean Close"], expected_past)
    assert np.isclose(
        targeted.loc[row, "Future 5-Session Mean Close"],
        expected_future,
    )
    assert targeted.loc[row, "Five-Session Mean Target"] == int(
        expected_future > expected_past
    )
    assert (
        targeted.loc[row, "Five-Session Target Date"]
        == prices.loc[row + 5, "Date"]
    )
