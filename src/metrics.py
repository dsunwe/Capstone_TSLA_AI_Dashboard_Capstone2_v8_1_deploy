from __future__ import annotations

import numpy as np
import pandas as pd


_DRAWDOWN_EPISODE_COLUMNS = [
    "Episode",
    "Peak Date",
    "Episode Start Date",
    "Trough Date",
    "Recovery Date",
    "Recovery Status",
    "Peak Equity",
    "Trough Equity",
    "Decline",
    "Drawdown Duration (Trading Days)",
    "Drawdown Duration (Calendar Days)",
    "Recovery Duration (Trading Days)",
    "Recovery Duration (Calendar Days)",
    "Underwater Duration (Trading Days)",
    "Underwater Duration (Calendar Days)",
]


def calculate_backtest_metrics(
    returns: pd.Series,
    position: pd.Series | None = None,
    dates: pd.Series | None = None,
) -> dict[str, float | int]:
    """Calculate consistent performance, risk, and operation metrics."""
    returns = pd.Series(returns).fillna(0).astype(float).reset_index(drop=True)
    if position is None:
        position = pd.Series(1, index=returns.index, dtype=float)
    else:
        position = pd.Series(position).fillna(0).astype(float).reset_index(drop=True)

    if dates is not None:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")

    if returns.empty:
        return {
            "Total Return": 0.0,
            "Annualised Return": 0.0,
            "Annualised Volatility": 0.0,
            "Maximum Drawdown": 0.0,
            "Calmar Ratio": 0.0,
            "Number of Trades": 0,
            "Exposure": 0.0,
            "Positive Active-Day Rate": 0.0,
        }

    equity = (1 + returns).cumprod()
    # Include the initial capital (1.0) as the first high-water mark. Without
    # this floor, a loss on the first evaluation day would incorrectly show a
    # drawdown of zero because the reduced equity becomes its own peak.
    running_peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / running_peak - 1
    total_return = float(equity.iloc[-1] - 1)

    if dates is not None and dates.notna().sum() >= 2:
        calendar_years = max((dates.max() - dates.min()).days / 365.25, 1 / 365.25)
        trading_years = max(len(returns) / 252, 1 / 252)
        years = max(calendar_years, trading_years)
    else:
        years = max(len(returns) / 252, 1 / 252)

    annualised_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    daily_std = float(returns.std()) if len(returns) > 1 else 0.0
    annualised_volatility = daily_std * np.sqrt(252)
    maximum_drawdown = float(drawdown.min())
    calmar_ratio = annualised_return / abs(maximum_drawdown) if maximum_drawdown < 0 else 0.0

    previous_position = position.shift(1).fillna(0)
    number_of_trades = int(((position == 1) & (previous_position == 0)).sum())
    exposure = float((position > 0).mean())
    invested_returns = returns[position > 0]
    positive_active_day_rate = (
        float((invested_returns > 0).mean()) if not invested_returns.empty else 0.0
    )

    return {
        "Total Return": total_return,
        "Annualised Return": float(annualised_return),
        "Annualised Volatility": float(annualised_volatility),
        "Maximum Drawdown": maximum_drawdown,
        "Calmar Ratio": float(calmar_ratio),
        "Number of Trades": number_of_trades,
        "Exposure": exposure,
        "Positive Active-Day Rate": positive_active_day_rate,
    }


def equity_drawdown_frame(
    dates: pd.Series,
    returns: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    returns = pd.Series(returns).fillna(0).astype(float).reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    return pd.DataFrame(
        {
            "Date": dates,
            "Model": model_name,
            "Daily_Return": returns,
            "Equity": equity,
            "Drawdown": drawdown,
        }
    )


def calculate_drawdown_episodes(
    dates: pd.Series,
    returns: pd.Series,
    *,
    minimum_drawdown: float = 0.10,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Identify material peak-to-trough drawdown and recovery episodes.

    Parameters
    ----------
    dates:
        Chronologically ordered observation dates.
    returns:
        Period returns aligned one-to-one with ``dates``.
    minimum_drawdown:
        Minimum absolute peak-to-trough decline to retain. For example, ``0.10``
        retains episodes with a decline of at least 10%. Set to ``0`` to retain
        every episode.
    top_n:
        Optionally retain only the ``n`` deepest qualifying episodes. Returned
        rows remain in chronological order.

    Notes
    -----
    ``Decline`` is a negative decimal return. Durations use observation counts
    (trading days) as well as calendar days where both endpoint dates exist.
    An episode is recovered on the first observation whose equity meets or
    exceeds the previous high-water mark. If the sample starts below the
    initial capital of 1.0, ``Peak Date`` is missing because that high-water
    mark predates the first observed return.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    returns = pd.Series(returns).reset_index(drop=True)
    if len(dates) != len(returns):
        raise ValueError("dates and returns must have the same length")
    if dates.isna().any():
        raise ValueError("dates must contain valid date-like values")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be in chronological order")
    if minimum_drawdown < 0 or not np.isfinite(minimum_drawdown):
        raise ValueError("minimum_drawdown must be a finite non-negative value")
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None")
    if dates.empty:
        return pd.DataFrame(columns=_DRAWDOWN_EPISODE_COLUMNS)

    numeric_returns = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    if not np.isfinite(numeric_returns).all():
        raise ValueError("returns must contain finite numeric values")
    if (numeric_returns <= -1).any():
        raise ValueError("returns must be greater than -1.0")

    equity = (1.0 + numeric_returns).cumprod()
    running_peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / running_peak - 1.0
    tolerance = 1e-12

    episode_bounds: list[tuple[int, int | None, int]] = []
    episode_start: int | None = None
    peak_index = -1
    for index, value in enumerate(drawdown):
        if episode_start is None and value < -tolerance:
            episode_start = index
            peak_index = index - 1
        elif episode_start is not None and value >= -tolerance:
            episode_bounds.append((episode_start, index, peak_index))
            episode_start = None
    if episode_start is not None:
        episode_bounds.append((episode_start, None, peak_index))

    rows: list[dict[str, object]] = []
    final_index = len(equity) - 1
    for episode_number, (start_index, recovery_index, peak_index) in enumerate(
        episode_bounds,
        start=1,
    ):
        episode_end = recovery_index if recovery_index is not None else final_index + 1
        trough_index = int(drawdown.iloc[start_index:episode_end].idxmin())
        peak_date = dates.iloc[peak_index] if peak_index >= 0 else pd.NaT
        trough_date = dates.iloc[trough_index]
        recovery_date = (
            dates.iloc[recovery_index] if recovery_index is not None else pd.NaT
        )
        peak_equity = float(running_peak.iloc[start_index])
        trough_equity = float(equity.iloc[trough_index])
        decline = trough_equity / peak_equity - 1.0
        if abs(decline) + tolerance < minimum_drawdown:
            continue

        drawdown_trading_days = trough_index - peak_index
        recovery_trading_days = (
            recovery_index - trough_index if recovery_index is not None else pd.NA
        )
        underwater_trading_days = (
            recovery_index - peak_index
            if recovery_index is not None
            else final_index - peak_index
        )
        drawdown_calendar_days = (
            int((trough_date - peak_date).days) if pd.notna(peak_date) else pd.NA
        )
        recovery_calendar_days = (
            int((recovery_date - trough_date).days)
            if pd.notna(recovery_date)
            else pd.NA
        )
        underwater_end_date = (
            recovery_date if pd.notna(recovery_date) else dates.iloc[final_index]
        )
        underwater_calendar_days = (
            int((underwater_end_date - peak_date).days)
            if pd.notna(peak_date)
            else pd.NA
        )

        rows.append(
            {
                "Episode": episode_number,
                "Peak Date": peak_date,
                "Episode Start Date": dates.iloc[start_index],
                "Trough Date": trough_date,
                "Recovery Date": recovery_date,
                "Recovery Status": (
                    "Recovered" if recovery_index is not None else "Unrecovered"
                ),
                "Peak Equity": peak_equity,
                "Trough Equity": trough_equity,
                "Decline": float(decline),
                "Drawdown Duration (Trading Days)": drawdown_trading_days,
                "Drawdown Duration (Calendar Days)": drawdown_calendar_days,
                "Recovery Duration (Trading Days)": recovery_trading_days,
                "Recovery Duration (Calendar Days)": recovery_calendar_days,
                "Underwater Duration (Trading Days)": underwater_trading_days,
                "Underwater Duration (Calendar Days)": underwater_calendar_days,
            }
        )

    episodes = pd.DataFrame(rows, columns=_DRAWDOWN_EPISODE_COLUMNS)
    if episodes.empty:
        return episodes
    if top_n is not None and len(episodes) > top_n:
        episodes = episodes.nsmallest(top_n, "Decline").sort_values(
            "Episode Start Date",
            kind="stable",
        )
    for column in [
        "Episode",
        "Drawdown Duration (Trading Days)",
        "Drawdown Duration (Calendar Days)",
        "Recovery Duration (Trading Days)",
        "Recovery Duration (Calendar Days)",
        "Underwater Duration (Trading Days)",
        "Underwater Duration (Calendar Days)",
    ]:
        episodes[column] = episodes[column].astype("Int64")
    return episodes.reset_index(drop=True)


def format_metrics_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    main_uncertainty = {
        "Point Estimate",
        "Requested Replicates",
        "Block Length",
    }.issubset(display.columns)
    percent_columns = [
        "Total Return",
        "Annualised Return",
        "Annualised Volatility",
        "Maximum Drawdown",
        "Exposure",
        "Positive Active-Day Rate",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "ROC-AUC",
        "PR-AUC",
        "Brier Skill",
        "Positive Prevalence",
        "Mean_Accuracy",
        "Std_Accuracy",
        "Mean_Balanced_Accuracy",
        "Mean_Macro_F1",
        "Std_Macro_F1",
        "Mean_ROC_AUC",
        "Mean CV Balanced Accuracy",
        "Mean CV Macro F1",
        "CV Standard Deviation",
        "Selection Rate",
        "Majority Baseline Accuracy",
        "Accuracy Improvement",
        "Majority Baseline F1-score",
        "F1-score Improvement",
        "Bootstrap P(Delta > 0)",
        "Empirical p-value",
        "Decline",
        "Gross Return",
        "Net Return",
        "Trade-Level Win Rate",
    ]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value * 100:,.2f}%"
            )
    if main_uncertainty:
        for column in [
            "Point Estimate",
            "Reference Value",
            "Difference",
            "CI Lower",
            "CI Upper",
            "Long-Signal Mean Gross Return",
            "Cash-Signal Market Mean Gross Return",
            "Model Mean Daily Net Return",
            "Benchmark Mean Daily Net Return",
            "Bootstrap Share Above Reference",
            "Bootstrap Share Above 0",
        ]:
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: (
                        "N/A" if pd.isna(value) else f"{value * 100:,.4f}%"
                    )
                )
    if "Calmar Ratio" in display.columns:
        display["Calmar Ratio"] = display["Calmar Ratio"].map(lambda value: f"{value:,.2f}")
    for column in [
        "MCC",
        "Brier Score",
        "Difference",
        "CI Lower",
        "CI Upper",
        "Observed Delta",
        "Null Mean Delta",
        "Null 95th Percentile",
    ]:
        if column in display.columns and not (
            main_uncertainty and column in {"Difference", "CI Lower", "CI Upper"}
        ):
            display[column] = display[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:,.4f}"
            )
    if "Number of Trades" in display.columns:
        display["Number of Trades"] = display["Number of Trades"].astype(int)
    for column in ["Closed Trades", "Open Trades"]:
        if column in display.columns:
            display[column] = display[column].astype(int)
    for column in ["Training Time (s)", "Inference Time (s)", "Total_Training_Time"]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:,.4f}"
            )
    return display.rename(
        columns={
            # Frozen v7.5 CSV schemas remain unchanged. These clearer names are
            # presentation-only labels for packaged and exploratory tables.
            "Average Rank": "Rank by Mean Macro-F1",
            "Number of Trades": "Entries (0→1)",
        }
    )
