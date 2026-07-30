import numpy as np
import pandas as pd
import pytest

from src.news import (
    ATTENTION_FEATURE_COLUMNS,
    CANONICAL_NEWS_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    aggregate_daily_news,
    deduplicate_news_articles,
    map_news_to_sessions,
    news_coverage_audit,
    normalize_alpha_vantage,
    normalize_gdelt,
    normalize_news_frame,
)
from src.config import NEWS_ATTENTION_FEATURES, NEWS_SENTIMENT_FEATURES


def _sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2024-07-05", "2024-07-08", "2024-07-09"]
            ),
            "market_close_utc": pd.to_datetime(
                [
                    "2024-07-05 20:00:00+00:00",
                    "2024-07-08 20:00:00+00:00",
                    "2024-07-09 20:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )


def test_provider_normalizers_emit_canonical_schema_and_ticker_relevance():
    alpha = normalize_alpha_vantage(
        {
            "feed": [
                {
                    "time_published": "20240705T150000",
                    "title": "Tesla delivery update",
                    "summary": "Quarterly deliveries were reported.",
                    "url": "https://example.com/tesla",
                    "source": "Example",
                    "ticker_sentiment": [
                        {"ticker": "TSLA", "relevance_score": "0.87"},
                        {"ticker": "AAPL", "relevance_score": "0.01"},
                    ],
                }
            ]
        },
        asset="TSLA",
        retrieved_at_utc="2024-07-10T00:00:00Z",
    )
    gdelt = normalize_gdelt(
        {
            "articles": [
                {
                    "seendate": "20240705T160000Z",
                    "title": "Tesla delivery update",
                    "url": "https://example.org/tesla",
                    "domain": "example.org",
                }
            ]
        },
        asset="TSLA",
    )

    assert alpha.columns.tolist() == CANONICAL_NEWS_COLUMNS
    assert gdelt.columns.tolist() == CANONICAL_NEWS_COLUMNS
    assert alpha.loc[0, "relevance_score"] == pytest.approx(0.87)
    assert alpha.loc[0, "published_at_utc"] == pd.Timestamp(
        "2024-07-05 15:00:00+00:00"
    )
    assert gdelt.loc[0, "provider"] == "GDELT"
    assert alpha.loc[0, "article_id"]


def test_url_or_title_exact_dedup_keeps_earliest_connected_article():
    raw = normalize_news_frame(
        pd.DataFrame(
            {
                "published": [
                    "2024-07-05T12:00:00Z",
                    "2024-07-05T13:00:00Z",
                    "2024-07-05T14:00:00Z",
                    "2024-07-05T15:00:00Z",
                ],
                "title": ["Original", "Bridge", "Bridge", "Independent"],
                "url": [
                    "https://example.com/story?utm_source=x",
                    "https://example.com/story",
                    "https://other.example/story",
                    "https://independent.example/story",
                ],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={"published_at_utc": "published"},
    )
    deduplicated = deduplicate_news_articles(raw)

    assert deduplicated["title"].tolist() == ["Original", "Independent"]
    assert deduplicated.loc[0, "published_at_utc"] == pd.Timestamp(
        "2024-07-05 12:00:00+00:00"
    )


def test_exact_title_dedup_defaults_to_a_48_hour_boundary():
    raw = normalize_news_frame(
        pd.DataFrame(
            {
                "published": [
                    "2024-01-01T12:00:00Z",
                    "2024-01-03T11:59:00Z",
                    "2024-02-01T12:00:00Z",
                ],
                "title": ["Recurring headline"] * 3,
                "url": [
                    "https://one.example/story",
                    "https://two.example/story",
                    "https://three.example/story",
                ],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={"published_at_utc": "published"},
    )

    deduplicated = deduplicate_news_articles(raw)
    assert deduplicated["published_at_utc"].tolist() == [
        pd.Timestamp("2024-01-01 12:00:00+00:00"),
        pd.Timestamp("2024-02-01 12:00:00+00:00"),
    ]


def test_session_mapping_uses_strict_close_and_conservative_date_only_rule():
    raw = normalize_news_frame(
        pd.DataFrame(
            {
                "published": [
                    "2024-07-05T19:59:00Z",
                    "2024-07-05T20:00:00Z",
                    "2024-07-06T10:00:00Z",
                ],
                "title": ["Before close", "At close", "Weekend"],
                "url": ["https://a", "https://b", "https://c"],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={"published_at_utc": "published"},
    )
    date_only = normalize_news_frame(
        pd.DataFrame(
            {
                "date": ["2024-07-05"],
                "title": ["Unknown intraday time"],
                "url": ["https://d"],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={"published_date": "date"},
    )
    mapped = map_news_to_sessions(pd.concat([raw, date_only]), _sessions())

    dates = dict(zip(mapped["title"], mapped["feature_date"]))
    assert dates["Before close"] == pd.Timestamp("2024-07-05")
    assert dates["At close"] == pd.Timestamp("2024-07-08")
    assert dates["Weekend"] == pd.Timestamp("2024-07-08")
    assert dates["Unknown intraday time"] == pd.Timestamp("2024-07-08")
    timestamped = mapped.loc[mapped["timestamp_precision"].eq("timestamp")]
    assert bool(
        (
            timestamped["published_at_utc"]
            < timestamped["feature_close_utc"]
        ).all()
    )


def test_daily_aggregation_separates_attention_and_sentiment_and_keeps_no_news():
    articles = normalize_news_frame(
        pd.DataFrame(
            {
                "published": [
                    "2024-07-05T12:00:00Z",
                    "2024-07-05T13:00:00Z",
                ],
                "title": ["Positive story", "Negative story"],
                "url": ["https://example.com/positive", "https://example.com/negative"],
                "source": ["Source A", "Source B"],
                "relevance": [2.0, 1.0],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={
            "published_at_utc": "published",
            "relevance_score": "relevance",
        },
    )
    mapped = map_news_to_sessions(articles, _sessions())
    mapped["positive_probability"] = [0.8, 0.1]
    mapped["negative_probability"] = [0.1, 0.7]
    mapped["neutral_probability"] = [0.1, 0.2]
    daily = aggregate_daily_news(
        mapped,
        session_dates=_sessions()["session_date"].iloc[:2],
        assets=["TSLA"],
    )

    first = daily.loc[daily["Date"].eq(pd.Timestamp("2024-07-05"))].iloc[0]
    second = daily.loc[daily["Date"].eq(pd.Timestamp("2024-07-08"))].iloc[0]
    expected_positive = (2 * 0.8 + 1 * 0.1) / 3
    expected_negative = (2 * 0.1 + 1 * 0.7) / 3
    assert first["Unique_Story_Count"] == 2
    assert first["FinBERT_Sentiment_Balance"] == pytest.approx(
        expected_positive - expected_negative
    )
    assert second["No_News"] == 1
    assert second["Unique_Story_Count"] == 0
    assert second["FinBERT_Neutral_Share"] == 0
    assert set(ATTENTION_FEATURE_COLUMNS).isdisjoint(SENTIMENT_FEATURE_COLUMNS)
    assert ATTENTION_FEATURE_COLUMNS == NEWS_ATTENTION_FEATURES
    assert SENTIMENT_FEATURE_COLUMNS == NEWS_SENTIMENT_FEATURES


def test_coverage_audit_reports_exact_duplicates_and_mapped_sessions():
    raw = normalize_news_frame(
        pd.DataFrame(
            {
                "published": [
                    "2024-07-05T12:00:00Z",
                    "2024-07-05T13:00:00Z",
                ],
                "title": ["Same title", "Same title"],
                "url": ["https://one.example", "https://two.example"],
            }
        ),
        asset="TSLA",
        provider="Fixture",
        column_map={"published_at_utc": "published"},
    )
    audit = news_coverage_audit(raw, sessions=_sessions())

    assert audit.loc[0, "Raw Articles"] == 2
    assert audit.loc[0, "Unique Articles"] == 1
    assert audit.loc[0, "Exact Duplicate Rows"] == 1
    assert audit.loc[0, "Mapped Articles"] == 1
    assert 0 < audit.loc[0, "Session Coverage"] <= 1
