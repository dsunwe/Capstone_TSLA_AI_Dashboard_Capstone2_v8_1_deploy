import numpy as np
import pandas as pd

from src.config import NEWS_ATTENTION_FEATURES, NEWS_SENTIMENT_FEATURES
from src.validation import (
    validate_sentiment_daily_news,
    validate_sentiment_oof,
    validate_sentiment_study,
)


def _daily_news_frame() -> pd.DataFrame:
    frames = []
    for asset_number, asset in enumerate(("TSLA", "AAPL"), start=1):
        dates = pd.bdate_range("2022-01-03", periods=60)
        count = pd.Series(
            np.where(np.arange(60) % 11 == 0, 0, (np.arange(60) % 7) + asset_number),
            dtype=float,
        )
        no_news = count.eq(0).astype(int)
        positive = pd.Series(np.where(no_news.eq(1), 0.0, 0.50))
        negative = pd.Series(np.where(no_news.eq(1), 0.0, 0.25))
        neutral = pd.Series(np.where(no_news.eq(1), 0.0, 0.25))
        balance = positive - negative

        preceding_count = count.shift(1)
        count_history = preceding_count.rolling(20, min_periods=5)
        abnormal = (
            (count - count_history.mean())
            / count_history.std(ddof=0).replace(0, np.nan)
        ).fillna(0)
        sentiment_3d = balance.ewm(span=3, adjust=False, min_periods=1).mean()
        surprise = (
            balance
            - balance.shift(1).rolling(20, min_periods=5).mean()
        ).fillna(0)

        frame = pd.DataFrame(
            {
                "Asset": asset,
                "Date": dates,
                "Log_News_Count": np.log1p(count),
                "Unique_Story_Count": count,
                "Unique_Source_Count": np.minimum(count, 3),
                "Abnormal_News_Volume_20": abnormal,
                "No_News": no_news,
                "FinBERT_Sentiment_Balance": balance,
                "FinBERT_Positive_Share": positive,
                "FinBERT_Negative_Share": negative,
                "FinBERT_Neutral_Share": neutral,
                "FinBERT_Disagreement": np.where(no_news.eq(1), 0.0, 0.10),
                "FinBERT_Max_Negative": negative,
                "FinBERT_Sentiment_3D": sentiment_3d,
                "FinBERT_Sentiment_Surprise_20": surprise,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _oof_and_thresholds() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_rows = []
    threshold_rows = []
    for asset in ("TSLA", "AAPL"):
        for target in ("direction", "large_move"):
            for fold, test_start in (
                (1, pd.Timestamp("2023-01-09")),
                (2, pd.Timestamp("2023-02-13")),
            ):
                train_end = test_start - pd.offsets.BDay(2)
                purge_date = test_start - pd.offsets.BDay(1)
                threshold = 0.02 + fold / 1000 if target == "large_move" else np.nan
                test_dates = pd.bdate_range(test_start, periods=3)
                threshold_rows.append(
                    {
                        "Asset": asset,
                        "Target": target,
                        "Fold": fold,
                        "Train End": train_end,
                        "Purged Anchor Date": purge_date,
                        "Test Start": test_dates.min(),
                        "Test End": test_dates.max(),
                        "Large-Move Threshold": threshold,
                    }
                )
                for feature_set in ("F0", "F1", "F2", "F3", "F4", "F5"):
                    for date_number, date in enumerate(test_dates):
                        oof_rows.append(
                            {
                                "Asset": asset,
                                "Target": target,
                                "Feature Set": feature_set,
                                "Model": "Random Forest Classifier",
                                "Fold": fold,
                                "Date": date,
                                "Train End": train_end,
                                "Actual": date_number % 2,
                                "Predicted": date_number % 2,
                                "Probability": 0.25 + date_number * 0.25,
                                "Large-Move Threshold": threshold,
                            }
                        )
    return pd.DataFrame(oof_rows), pd.DataFrame(threshold_rows)


def _status(report: pd.DataFrame, test_id: str) -> str:
    return report.loc[report["Test ID"].eq(test_id), "Status"].iloc[0]


def test_daily_news_validation_accepts_complete_causal_panel():
    report = validate_sentiment_daily_news(_daily_news_frame())

    assert set(report["Test ID"]) == {"N01", "N02", "N03", "N04", "N05"}
    assert report["Status"].eq("Pass").all()
    assert set(NEWS_ATTENTION_FEATURES + NEWS_SENTIMENT_FEATURES).issubset(
        _daily_news_frame().columns
    )


def test_daily_news_validation_detects_probability_and_no_news_encoding_errors():
    frame = _daily_news_frame()
    no_news_index = frame.index[frame["No_News"].eq(1)][0]
    frame.loc[no_news_index, "FinBERT_Neutral_Share"] = 1.0
    news_index = frame.index[frame["No_News"].eq(0)][0]
    frame.loc[news_index, "FinBERT_Positive_Share"] = 1.25

    report = validate_sentiment_daily_news(frame)

    assert _status(report, "N03") == "Fail"
    assert _status(report, "N04") == "Fail"


def test_daily_news_validation_detects_duplicates_missing_columns_and_future_rolling():
    frame = _daily_news_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame = frame.drop(columns=["Unique_Source_Count"])
    frame["FinBERT_Sentiment_Surprise_20"] = (
        frame.groupby("Asset")["FinBERT_Sentiment_Balance"].shift(-1).fillna(0)
    )

    report = validate_sentiment_daily_news(frame)

    assert _status(report, "N01") == "Fail"
    assert _status(report, "N02") == "Fail"
    assert _status(report, "N05") == "Fail"


def test_sentiment_oof_validation_accepts_paired_chronological_ledger():
    oof, thresholds = _oof_and_thresholds()

    report = validate_sentiment_oof(oof, thresholds)

    assert set(report["Test ID"]) == {"O01", "O02", "O03", "O04"}
    assert report["Status"].eq("Pass").all()


def test_sentiment_oof_validation_detects_all_required_failure_modes():
    oof, thresholds = _oof_and_thresholds()
    duplicate = oof.iloc[[0]].copy()
    oof = pd.concat([oof, duplicate], ignore_index=True)
    oof.loc[0, "Train End"] = oof.loc[0, "Date"]
    oof = oof.loc[~oof["Feature Set"].eq("F5")].copy()

    large_row = thresholds.index[thresholds["Target"].eq("large_move")][0]
    thresholds.loc[large_row, "Large-Move Threshold"] = np.nan
    thresholds.loc[large_row, "Purged Anchor Date"] = thresholds.loc[
        large_row, "Train End"
    ]

    report = validate_sentiment_oof(oof, thresholds)

    assert _status(report, "O01") == "Fail"
    assert _status(report, "O02") == "Fail"
    assert _status(report, "O03") == "Fail"
    assert _status(report, "O04") == "Fail"


def test_combined_sentiment_validation_has_one_nine_check_ledger():
    daily = _daily_news_frame()
    oof, thresholds = _oof_and_thresholds()

    report = validate_sentiment_study(daily, oof, thresholds)

    assert len(report) == 9
    assert report["Status"].eq("Pass").all()
