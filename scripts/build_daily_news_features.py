"""Map cached article probabilities to XNYS sessions and build v7 daily features."""
from __future__ import annotations

import argparse
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd


ASSETS = ("AAPL", "NVDA", "TSLA")
ATTENTION_COLUMNS = [
    "Log_News_Count",
    "Unique_Story_Count",
    "Unique_Source_Count",
    "Abnormal_News_Volume_20",
    "No_News",
]
SENTIMENT_COLUMNS = [
    "FinBERT_Sentiment_Balance",
    "FinBERT_Positive_Share",
    "FinBERT_Negative_Share",
    "FinBERT_Neutral_Share",
    "FinBERT_Disagreement",
    "FinBERT_Max_Negative",
    "FinBERT_Sentiment_3D",
    "FinBERT_Sentiment_Surprise_20",
]


def xnys_schedule(start: str, end: str) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[start:end, ["open", "close"]].reset_index()
    schedule = schedule.rename(
        columns={"index": "Date", "open": "Open UTC", "close": "Close UTC"}
    )
    schedule["Date"] = pd.to_datetime(schedule["Date"]).dt.tz_localize(None)
    schedule["Open UTC"] = pd.to_datetime(schedule["Open UTC"], utc=True)
    schedule["Close UTC"] = pd.to_datetime(schedule["Close UTC"], utc=True)
    schedule["Calendar"] = "XNYS"
    schedule["Calendar Package"] = f"exchange_calendars {xcals.__version__}"
    return schedule


def map_articles_to_sessions(
    articles: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    output = articles.copy()
    output["source_timestamp_utc"] = pd.to_datetime(
        output["source_timestamp_utc"], utc=True, errors="raise"
    )
    session_dates = schedule["Date"].to_numpy(dtype="datetime64[ns]")
    close_values = schedule["Close UTC"].astype("int64").to_numpy()
    timestamps = output["source_timestamp_utc"].astype("int64").to_numpy()
    date_only = output["timestamp_precision"].eq("date_only_placeholder").to_numpy()

    mapped_positions = np.searchsorted(close_values, timestamps, side="right")
    source_dates = (
        output["source_timestamp_utc"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
        .to_numpy(dtype="datetime64[ns]")
    )
    # A date-only record has no auditable within-day publication time. It is
    # therefore deferred to the first exchange session strictly after its date.
    mapped_positions[date_only] = np.searchsorted(
        session_dates, source_dates[date_only], side="right"
    )
    valid = mapped_positions < len(schedule)
    output["mapping_status"] = np.where(valid, "mapped", "outside_calendar")
    output["Date"] = pd.NaT
    output.loc[valid, "Date"] = schedule.iloc[
        mapped_positions[valid]
    ]["Date"].to_numpy()
    output["Date"] = pd.to_datetime(output["Date"])

    eastern_dates = (
        output["source_timestamp_utc"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    output["date_only_deferred"] = date_only
    output["after_close_or_non_session"] = (
        ~date_only & output["Date"].gt(eastern_dates)
    )
    output["mapped_before_source_violation"] = False
    mapped = output.loc[valid].copy()
    close_lookup = schedule.set_index("Date")["Close UTC"]
    mapped_closes = mapped["Date"].map(close_lookup)
    intraday_mask = mapped["timestamp_precision"].eq("intraday")
    mapped.loc[intraday_mask, "mapped_before_source_violation"] = (
        mapped_closes.loc[intraday_mask]
        <= mapped.loc[intraday_mask, "source_timestamp_utc"]
    ).to_numpy()
    return mapped.reset_index(drop=True)


def aggregate_daily_features(
    mapped: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    study_start: str,
    study_end: str,
) -> pd.DataFrame:
    mapped = mapped.copy()
    mapped["balance"] = mapped["prob_positive"] - mapped["prob_negative"]
    grouped = mapped.groupby(["Asset", "Date"], sort=True)
    daily = grouped.agg(
        Unique_Story_Count=("story_id", "nunique"),
        Unique_Source_Count=("source_domain", "nunique"),
        FinBERT_Sentiment_Balance=("balance", "mean"),
        FinBERT_Disagreement=("balance", lambda values: float(values.std(ddof=0))),
        FinBERT_Max_Negative=("prob_negative", "max"),
        Scored_Story_Count=("story_id", "size"),
        Source_Max_Timestamp_UTC=("source_timestamp_utc", "max"),
    ).reset_index()
    label_counts = (
        mapped.assign(value=1)
        .pivot_table(
            index=["Asset", "Date"],
            columns="predicted_label",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["positive", "negative", "neutral"], fill_value=0)
        .reset_index()
    )
    daily = daily.merge(label_counts, on=["Asset", "Date"], validate="one_to_one")
    denominator = daily["Scored_Story_Count"].replace(0, np.nan)
    daily["FinBERT_Positive_Share"] = daily["positive"] / denominator
    daily["FinBERT_Negative_Share"] = daily["negative"] / denominator
    daily["FinBERT_Neutral_Share"] = daily["neutral"] / denominator
    daily = daily.drop(columns=["positive", "negative", "neutral"])

    eligible_dates = schedule.loc[
        schedule["Date"].ge(pd.Timestamp(study_start))
        & schedule["Date"].le(pd.Timestamp(study_end)),
        ["Date"],
    ]
    grid = pd.MultiIndex.from_product(
        [ASSETS, eligible_dates["Date"]],
        names=["Asset", "Date"],
    ).to_frame(index=False)
    panel = grid.merge(daily, on=["Asset", "Date"], how="left", validate="one_to_one")
    panel["Unique_Story_Count"] = panel["Unique_Story_Count"].fillna(0).astype(int)
    panel["Unique_Source_Count"] = panel["Unique_Source_Count"].fillna(0).astype(int)
    panel["Scored_Story_Count"] = panel["Scored_Story_Count"].fillna(0).astype(int)
    panel["No_News"] = panel["Unique_Story_Count"].eq(0).astype(int)
    panel["Log_News_Count"] = np.log1p(panel["Unique_Story_Count"])
    zero_on_no_news = [
        "FinBERT_Sentiment_Balance",
        "FinBERT_Positive_Share",
        "FinBERT_Negative_Share",
        "FinBERT_Neutral_Share",
        "FinBERT_Disagreement",
        "FinBERT_Max_Negative",
    ]
    panel[zero_on_no_news] = panel[zero_on_no_news].fillna(0.0)

    completed: list[pd.DataFrame] = []
    for asset, asset_frame in panel.groupby("Asset", sort=True):
        asset_frame = asset_frame.sort_values("Date").copy()
        count = asset_frame["Unique_Story_Count"].astype(float)
        count_history = count.shift(1).rolling(20, min_periods=5)
        asset_frame["Abnormal_News_Volume_20"] = (
            (count - count_history.mean())
            / count_history.std(ddof=0).replace(0, np.nan)
        ).fillna(0.0)
        balance = asset_frame["FinBERT_Sentiment_Balance"].astype(float)
        asset_frame["FinBERT_Sentiment_3D"] = balance.ewm(
            span=3, adjust=False, min_periods=1
        ).mean()
        asset_frame["FinBERT_Sentiment_Surprise_20"] = (
            balance
            - balance.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(0.0)
        asset_frame["feature_as_of"] = asset_frame["Date"]
        asset_frame["attention_history_end"] = asset_frame["Date"].shift(1)
        asset_frame["sentiment_history_end"] = asset_frame["Date"].shift(1)
        completed.append(asset_frame)
    output = pd.concat(completed, ignore_index=True)
    columns = [
        "Asset",
        "Date",
        *ATTENTION_COLUMNS,
        *SENTIMENT_COLUMNS,
        "Scored_Story_Count",
        "Source_Max_Timestamp_UTC",
        "feature_as_of",
        "attention_history_end",
        "sentiment_history_end",
    ]
    return output[columns].sort_values(["Asset", "Date"]).reset_index(drop=True)


def coverage_audit(
    articles: pd.DataFrame,
    daily: pd.DataFrame,
    dedup_audit: pd.DataFrame,
) -> pd.DataFrame:
    formal_articles = articles.loc[
        articles["Date"].between(daily["Date"].min(), daily["Date"].max())
    ].copy()
    rows: list[dict[str, object]] = []
    for asset in ASSETS:
        asset_articles = formal_articles.loc[formal_articles["Asset"].eq(asset)]
        asset_daily = daily.loc[daily["Asset"].eq(asset)]
        raw_row = dedup_audit.loc[dedup_audit["Asset"].eq(asset)]
        deduplicated_source_stories = (
            int(raw_row["Unique Stories"].iloc[0])
            if not raw_row.empty and "Unique Stories" in raw_row
            else len(asset_articles)
        )
        rows.append(
            {
                "Asset": asset,
                "Raw Articles": (
                    int(raw_row["Raw Articles"].iloc[0])
                    if not raw_row.empty
                    else len(asset_articles)
                ),
                "Unique Stories": int(len(asset_articles)),
                "Deduplicated Source Stories": deduplicated_source_stories,
                "Excluded Outside Session Window": (
                    deduplicated_source_stories - len(asset_articles)
                ),
                "Duplicate Rows Removed": (
                    int(raw_row["Duplicate Rows Removed"].iloc[0])
                    if not raw_row.empty
                    else np.nan
                ),
                "Earliest Source Timestamp UTC": (
                    asset_articles["source_timestamp_utc"].min()
                    if not asset_articles.empty
                    else pd.NaT
                ),
                "Latest Source Timestamp UTC": (
                    asset_articles["source_timestamp_utc"].max()
                    if not asset_articles.empty
                    else pd.NaT
                ),
                "Date-Only Deferred": int(
                    asset_articles["date_only_deferred"].sum()
                ),
                "Intraday Timestamped": int(
                    asset_articles["timestamp_precision"].eq("intraday").sum()
                ),
                "After-Close/Non-Session Mapped": int(
                    asset_articles["after_close_or_non_session"].sum()
                ),
                "Trading Sessions": int(len(asset_daily)),
                "Sessions With News": int(asset_daily["No_News"].eq(0).sum()),
                "Sessions Without News": int(asset_daily["No_News"].eq(1).sum()),
                "Unique Source Domains": int(
                    asset_articles["source_domain"].nunique()
                ),
                "Future Mapping Violations": int(
                    asset_articles["mapped_before_source_violation"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-scores", type=Path, required=True)
    parser.add_argument("--dedup-audit", type=Path, required=True)
    parser.add_argument("--calendar-output", type=Path, required=True)
    parser.add_argument("--daily-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--calendar-start", default="2021-01-01")
    parser.add_argument("--calendar-end", default="2024-01-10")
    parser.add_argument("--study-start", default="2022-06-03")
    parser.add_argument("--study-end", default="2023-12-16")
    args = parser.parse_args()

    articles = pd.read_csv(args.article_scores)
    dedup = pd.read_csv(args.dedup_audit)
    schedule = xnys_schedule(args.calendar_start, args.calendar_end)
    mapped = map_articles_to_sessions(articles, schedule)
    source_start = pd.Timestamp(args.study_start, tz="UTC")
    source_end = pd.Timestamp(args.study_end, tz="UTC") + pd.Timedelta(days=1)
    mapped = mapped.loc[
        mapped["source_timestamp_utc"].ge(source_start)
        & mapped["source_timestamp_utc"].lt(source_end)
    ].reset_index(drop=True)
    daily = aggregate_daily_features(
        mapped,
        schedule,
        study_start=args.study_start,
        study_end=args.study_end,
    )
    coverage = coverage_audit(mapped, daily, dedup)
    for path in [args.calendar_output, args.daily_output, args.coverage_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(args.calendar_output, index=False)
    daily.to_csv(args.daily_output, index=False)
    coverage.to_csv(args.coverage_output, index=False)
    print(
        f"Wrote {len(daily):,} daily rows from {len(mapped):,} mapped unique stories."
    )


if __name__ == "__main__":
    main()
