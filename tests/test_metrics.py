import pandas as pd

from src.metrics import (
    calculate_backtest_metrics,
    calculate_drawdown_episodes,
    equity_drawdown_frame,
    format_metrics_for_display,
)
from src.visualization import price_moving_average_chart


def test_metrics_total_return_and_trade_count():
    returns = pd.Series([0.10, 0.0, -0.05, 0.02])
    position = pd.Series([1, 0, 1, 1])
    metrics = calculate_backtest_metrics(returns, position)
    expected_total = (1.10 * 1.00 * 0.95 * 1.02) - 1
    assert abs(metrics["Total Return"] - expected_total) < 1e-12
    assert metrics["Number of Trades"] == 2
    assert metrics["Exposure"] == 0.75
    assert metrics["Positive Active-Day Rate"] == 2 / 3


def test_metric_display_uses_precise_rank_and_entry_labels():
    display = format_metrics_for_display(
        pd.DataFrame(
            [
                {
                    "Average Rank": 1,
                    "Number of Trades": 2,
                }
            ]
        )
    )

    assert "Rank by Mean Macro-F1" in display.columns
    assert "Entries (0→1)" in display.columns
    assert "Average Rank" not in display.columns
    assert "Number of Trades" not in display.columns


def test_metrics_drawdown_is_non_positive():
    metrics = calculate_backtest_metrics(pd.Series([0.1, -0.2, 0.05]))
    assert metrics["Maximum Drawdown"] <= 0


def test_first_period_loss_is_measured_from_initial_capital():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    returns = pd.Series([-0.10, 0.02])
    metrics = calculate_backtest_metrics(returns, dates=pd.Series(dates))
    equity = equity_drawdown_frame(pd.Series(dates), returns, "Test")

    assert abs(equity.loc[0, "Drawdown"] + 0.10) < 1e-12
    assert abs(metrics["Maximum Drawdown"] + 0.10) < 1e-12


def test_drawdown_episodes_capture_recovered_and_unrecovered_events():
    dates = pd.Series(pd.bdate_range("2026-01-01", periods=7))
    returns = pd.Series([0.10, -0.10, -0.10, 0.25, 0.05, -0.20, 0.10])

    episodes = calculate_drawdown_episodes(
        dates,
        returns,
        minimum_drawdown=0.15,
    )

    assert len(episodes) == 2
    first = episodes.iloc[0]
    assert first["Peak Date"] == dates.iloc[0]
    assert first["Trough Date"] == dates.iloc[2]
    assert first["Recovery Date"] == dates.iloc[3]
    assert first["Recovery Status"] == "Recovered"
    assert abs(first["Decline"] + 0.19) < 1e-12
    assert first["Drawdown Duration (Trading Days)"] == 2
    assert first["Recovery Duration (Trading Days)"] == 1
    assert first["Underwater Duration (Calendar Days)"] == 5

    second = episodes.iloc[1]
    assert second["Peak Date"] == dates.iloc[4]
    assert second["Trough Date"] == dates.iloc[5]
    assert pd.isna(second["Recovery Date"])
    assert second["Recovery Status"] == "Unrecovered"
    assert abs(second["Decline"] + 0.20) < 1e-12
    assert pd.isna(second["Recovery Duration (Trading Days)"])
    assert second["Underwater Duration (Trading Days)"] == 2
    assert second["Underwater Duration (Calendar Days)"] == 2


def test_drawdown_episodes_top_n_keeps_deepest_event():
    dates = pd.Series(pd.bdate_range("2026-01-01", periods=7))
    returns = pd.Series([0.10, -0.10, -0.10, 0.25, 0.05, -0.20, 0.10])

    episodes = calculate_drawdown_episodes(
        dates,
        returns,
        minimum_drawdown=0,
        top_n=1,
    )

    assert len(episodes) == 1
    assert episodes.iloc[0]["Recovery Status"] == "Unrecovered"
    assert abs(episodes.iloc[0]["Decline"] + 0.20) < 1e-12


def test_price_moving_average_chart_computes_lines_without_mutating_input():
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2026-01-01", periods=25),
            "Price": range(100, 125),
        }
    )
    original_columns = frame.columns.tolist()

    figure = price_moving_average_chart(frame)

    assert figure is not None
    assert [line.get_label() for line in figure.axes[0].lines] == [
        "Price",
        "MA10",
        "MA20",
    ]
    assert frame.columns.tolist() == original_columns
