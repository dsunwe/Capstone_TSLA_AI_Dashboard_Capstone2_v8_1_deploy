from __future__ import annotations

from datetime import date, datetime
import hashlib
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import unicodedata

import numpy as np
import pandas as pd


CANONICAL_NEWS_COLUMNS = [
    "asset",
    "provider",
    "article_id",
    "published_at_utc",
    "published_date",
    "timestamp_precision",
    "title",
    "summary",
    "url",
    "source",
    "relevance_score",
    "retrieved_at_utc",
]

ATTENTION_FEATURE_COLUMNS = [
    "Log_News_Count",
    "Unique_Story_Count",
    "Unique_Source_Count",
    "Abnormal_News_Volume_20",
    "No_News",
]

SENTIMENT_FEATURE_COLUMNS = [
    "FinBERT_Sentiment_Balance",
    "FinBERT_Positive_Share",
    "FinBERT_Negative_Share",
    "FinBERT_Neutral_Share",
    "FinBERT_Disagreement",
    "FinBERT_Max_Negative",
    "FinBERT_Sentiment_3D",
    "FinBERT_Sentiment_Surprise_20",
]

NEWS_AUDIT_FEATURE_COLUMNS = [
    "News_Relevance_Sum",
    "FinBERT_Scored_Article_Count",
    "FinBERT_Sentiment_Missing",
]

_DATE_ONLY_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),
    re.compile(r"^\d{8}$"),
]
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


def _empty_canonical_frame() -> pd.DataFrame:
    result = pd.DataFrame(columns=CANONICAL_NEWS_COLUMNS)
    result["published_at_utc"] = pd.to_datetime(
        result["published_at_utc"], utc=True
    )
    result["published_date"] = pd.to_datetime(result["published_date"])
    result["retrieved_at_utc"] = pd.to_datetime(
        result["retrieved_at_utc"], utc=True
    )
    return result


def _clean_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _is_date_only(value: Any) -> bool:
    if isinstance(value, date) and not isinstance(value, datetime):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return any(pattern.fullmatch(stripped) for pattern in _DATE_ONLY_PATTERNS)
    return False


def _parse_publication(
    value: Any,
    *,
    force_date_only: bool = False,
    default_timezone: str = "UTC",
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, str]:
    if value is None or (not isinstance(value, (str, date, datetime, pd.Timestamp)) and pd.isna(value)):
        return pd.NaT, pd.NaT, "missing"
    if isinstance(value, str) and not value.strip():
        return pd.NaT, pd.NaT, "missing"

    if force_date_only or _is_date_only(value):
        parsed_date = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed_date):
            return pd.NaT, pd.NaT, "missing"
        if isinstance(parsed_date, pd.DatetimeIndex):
            parsed_date = parsed_date[0]
        timestamp = pd.Timestamp(parsed_date)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(default_timezone).tz_localize(None)
        return pd.NaT, timestamp.normalize(), "date"

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT, pd.NaT, "missing"
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(default_timezone)
    timestamp = timestamp.tz_convert("UTC")
    return timestamp, timestamp.tz_convert(default_timezone).tz_localize(None).normalize(), "timestamp"


def _canonical_article_id(
    *,
    asset: str,
    provider: str,
    title: str,
    url: str,
    published_value: Any,
) -> str:
    payload = "\x1f".join(
        [
            asset.upper(),
            provider.upper(),
            _clean_text(title),
            _clean_text(url),
            _clean_text(published_value),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_source_column(
    frame: pd.DataFrame,
    canonical_name: str,
    column_map: Mapping[str, str],
    aliases: Iterable[str],
) -> str | None:
    if canonical_name in column_map and column_map[canonical_name] in frame.columns:
        return column_map[canonical_name]
    for source_name, target_name in column_map.items():
        if target_name == canonical_name and source_name in frame.columns:
            return source_name
    for candidate in [canonical_name, *aliases]:
        if candidate in frame.columns:
            return candidate
    return None


def normalize_news_frame(
    frame: pd.DataFrame,
    *,
    asset: str,
    provider: str,
    column_map: Mapping[str, str] | None = None,
    default_timezone: str = "UTC",
) -> pd.DataFrame:
    """Normalize a provider table to one auditable article-level schema.

    ``column_map`` may map canonical names to provider columns or provider
    columns to canonical names. Date-only publication values deliberately
    remain date-only so they can later be mapped conservatively to the next
    trading session.
    """
    if frame is None or frame.empty:
        return _empty_canonical_frame()

    source = frame.copy().reset_index(drop=True)
    mapping = dict(column_map or {})
    aliases = {
        "article_id": ["id", "news_id"],
        "published_at_utc": [
            "published_at",
            "time_published",
            "seendate",
            "published",
            "datetime",
        ],
        "published_date": ["date", "publication_date"],
        "title": ["headline", "name"],
        "summary": ["description", "snippet"],
        "url": ["link", "article_url"],
        "source": ["domain", "publisher", "source_name"],
        "relevance_score": ["relevance", "ticker_relevance"],
        "retrieved_at_utc": ["retrieved_at", "ingested_at_utc"],
    }
    resolved = {
        name: _resolve_source_column(source, name, mapping, aliases.get(name, []))
        for name in aliases
    }

    publication_column = resolved["published_at_utc"] or resolved["published_date"]
    force_date_only = (
        publication_column is not None
        and publication_column == resolved["published_date"]
        and resolved["published_at_utc"] is None
    )
    publication_values = (
        source[publication_column]
        if publication_column is not None
        else pd.Series([None] * len(source))
    )
    parsed = [
        _parse_publication(
            value,
            force_date_only=force_date_only,
            default_timezone=default_timezone,
        )
        for value in publication_values
    ]

    def values(name: str, default: Any = "") -> pd.Series:
        column = resolved.get(name)
        if column is None:
            return pd.Series([default] * len(source))
        return source[column]

    titles = values("title").map(_clean_text)
    summaries = values("summary").map(_clean_text)
    urls = values("url").map(_clean_text)
    publication_text = publication_values.map(_clean_text)
    raw_ids = values("article_id").map(_clean_text)
    article_ids = [
        raw_id
        or _canonical_article_id(
            asset=asset,
            provider=provider,
            title=title,
            url=url,
            published_value=published,
        )
        for raw_id, title, url, published in zip(
            raw_ids, titles, urls, publication_text
        )
    ]
    relevance = pd.to_numeric(values("relevance_score", np.nan), errors="coerce")

    retrieved_column = resolved.get("retrieved_at_utc")
    if retrieved_column is None:
        retrieved = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns, UTC]")
    else:
        retrieved = pd.to_datetime(source[retrieved_column], errors="coerce", utc=True)

    result = pd.DataFrame(
        {
            "asset": asset.upper(),
            "provider": provider,
            "article_id": article_ids,
            "published_at_utc": pd.to_datetime(
                [item[0] for item in parsed], errors="coerce", utc=True
            ),
            "published_date": pd.to_datetime(
                [item[1] for item in parsed], errors="coerce"
            ).normalize(),
            "timestamp_precision": [item[2] for item in parsed],
            "title": titles,
            "summary": summaries,
            "url": urls,
            "source": values("source").map(_clean_text),
            "relevance_score": relevance,
            "retrieved_at_utc": retrieved,
        }
    )
    return result[CANONICAL_NEWS_COLUMNS]


def normalize_alpha_vantage(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    asset: str,
    retrieved_at_utc: Any = None,
) -> pd.DataFrame:
    """Normalize an Alpha Vantage ``NEWS_SENTIMENT`` response.

    Alpha Vantage's own sentiment score is intentionally ignored. Only the
    ticker-specific relevance score is retained for later FinBERT aggregation.
    """
    feed = payload.get("feed", []) if isinstance(payload, Mapping) else payload
    rows: list[dict[str, Any]] = []
    for article in feed or []:
        ticker_rows = article.get("ticker_sentiment") or []
        match = next(
            (
                ticker
                for ticker in ticker_rows
                if str(ticker.get("ticker", "")).upper() == asset.upper()
            ),
            None,
        )
        rows.append(
            {
                "article_id": article.get("id", ""),
                "time_published": article.get("time_published"),
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "relevance_score": (
                    match.get("relevance_score") if match is not None else np.nan
                ),
                "retrieved_at_utc": retrieved_at_utc,
            }
        )
    return normalize_news_frame(
        pd.DataFrame(rows),
        asset=asset,
        provider="Alpha Vantage",
        column_map={
            "published_at_utc": "time_published",
            "relevance_score": "relevance_score",
        },
        default_timezone="UTC",
    )


def normalize_gdelt(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    asset: str,
    retrieved_at_utc: Any = None,
) -> pd.DataFrame:
    """Normalize a GDELT DOC response without treating GDELT tone as FinBERT."""
    articles = (
        payload.get("articles", [])
        if isinstance(payload, Mapping)
        else payload
    )
    rows = [
        {
            "article_id": article.get("id", ""),
            "seendate": article.get("seendate") or article.get("date"),
            "title": article.get("title", ""),
            "summary": article.get("summary") or article.get("description", ""),
            "url": article.get("url", ""),
            "source": article.get("domain") or article.get("source", ""),
            "relevance_score": article.get("relevance_score", np.nan),
            "retrieved_at_utc": retrieved_at_utc,
        }
        for article in articles or []
    ]
    return normalize_news_frame(
        pd.DataFrame(rows),
        asset=asset,
        provider="GDELT",
        column_map={"published_at_utc": "seendate"},
        default_timezone="UTC",
    )


def _normalise_url(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.netloc:
        return text
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            urlencode(query),
            "",
        )
    )


def _normalise_title(value: Any) -> str:
    return _clean_text(value).casefold()


def deduplicate_news_articles(
    frame: pd.DataFrame,
    *,
    title_window_hours: float | None = 48.0,
) -> pd.DataFrame:
    """Remove exact normalized-URL or normalized-title duplicates.

    Deduplication is performed independently per asset and across providers.
    URL matches are global because a canonical URL identifies the same story.
    Exact-title matches are merged only inside a 48-hour window by default, so
    recurring headlines in different months or years remain distinct. Connected
    duplicate groups are resolved with union-find and retain the earliest row.
    """
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else _empty_canonical_frame()
    required = {"asset", "url", "title"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing news deduplication columns: {sorted(missing)}")

    data = frame.copy().reset_index(drop=False).rename(columns={"index": "_original_index"})
    data["_url_key"] = data["url"].map(_normalise_url)
    data["_title_key"] = data["title"].map(_normalise_title)
    timestamp = pd.to_datetime(
        data.get("published_at_utc"), errors="coerce", utc=True
    )
    published_date = pd.to_datetime(
        data.get("published_date"), errors="coerce", utc=True
    )
    data["_publication_sort"] = timestamp.fillna(published_date)
    if title_window_hours is not None and float(title_window_hours) < 0:
        raise ValueError("title_window_hours must be non-negative or None.")

    parent = list(range(len(data)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, indices in data.groupby("asset", sort=False).groups.items():
        url_owner: dict[str, int] = {}
        for index in list(indices):
            url_key = data.at[index, "_url_key"]
            if not url_key:
                continue
            if url_key in url_owner:
                union(url_owner[url_key], index)
            else:
                url_owner[url_key] = index

        asset_rows = data.loc[list(indices)]
        for title_key, title_rows in asset_rows.loc[
            asset_rows["_title_key"].ne("")
        ].groupby("_title_key", sort=False):
            del title_key
            ordered = title_rows.dropna(subset=["_publication_sort"]).sort_values(
                ["_publication_sort", "_original_index"], kind="mergesort"
            )
            if ordered.empty:
                continue
            if title_window_hours is None:
                owner = int(ordered.index[0])
                for index in ordered.index[1:]:
                    union(owner, int(index))
                continue

            window = pd.Timedelta(hours=float(title_window_hours))
            cluster_owner = int(ordered.index[0])
            cluster_start = pd.Timestamp(
                ordered.loc[cluster_owner, "_publication_sort"]
            )
            for index in ordered.index[1:]:
                publication_time = pd.Timestamp(
                    ordered.loc[index, "_publication_sort"]
                )
                if publication_time - cluster_start <= window:
                    union(cluster_owner, int(index))
                else:
                    cluster_owner = int(index)
                    cluster_start = publication_time

    data["_duplicate_root"] = [find(index) for index in range(len(data))]
    keep_indices: list[int] = []
    far_future = pd.Timestamp.max.tz_localize("UTC")
    for _, group in data.groupby("_duplicate_root", sort=False):
        ordered = group.assign(
            _resolved_sort=group["_publication_sort"].fillna(far_future)
        ).sort_values(["_resolved_sort", "_original_index"], kind="mergesort")
        keep_indices.append(int(ordered.index[0]))

    result = (
        data.loc[keep_indices]
        .assign(_resolved_sort=lambda x: x["_publication_sort"].fillna(far_future))
        .sort_values(["_resolved_sort", "_original_index"], kind="mergesort")
        .drop(
            columns=[
                "_original_index",
                "_url_key",
                "_title_key",
                "_publication_sort",
                "_duplicate_root",
                "_resolved_sort",
            ]
        )
        .reset_index(drop=True)
    )
    return result


def _prepare_sessions(
    sessions: pd.DataFrame,
    *,
    session_date_col: str,
    market_close_col: str,
) -> pd.DataFrame:
    if not isinstance(sessions, pd.DataFrame) or sessions.empty:
        raise ValueError("A non-empty session calendar DataFrame is required.")
    calendar = sessions.copy()
    if session_date_col not in calendar.columns:
        if isinstance(calendar.index, pd.DatetimeIndex):
            calendar[session_date_col] = calendar.index
        else:
            raise ValueError(f"Session calendar lacks {session_date_col!r}.")
    if market_close_col not in calendar.columns:
        raise ValueError(f"Session calendar lacks {market_close_col!r}.")
    calendar[session_date_col] = pd.to_datetime(
        calendar[session_date_col], errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    calendar[market_close_col] = pd.to_datetime(
        calendar[market_close_col], errors="coerce", utc=True
    )
    calendar = (
        calendar.dropna(subset=[session_date_col, market_close_col])
        .sort_values(market_close_col)
        .drop_duplicates(session_date_col, keep="last")
        .reset_index(drop=True)
    )
    if calendar.empty:
        raise ValueError("The supplied session calendar has no valid sessions.")
    if not calendar[market_close_col].is_monotonic_increasing:
        raise ValueError("Market close timestamps must be chronological.")
    return calendar


def map_news_to_sessions(
    frame: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    session_date_col: str = "session_date",
    market_close_col: str = "market_close_utc",
) -> pd.DataFrame:
    """Map news to the first usable market close without look-ahead.

    A known timestamp maps to the first close strictly after publication.
    Date-only news maps to the first session date strictly after its stated
    date, because its intraday availability is unknown.
    """
    calendar = _prepare_sessions(
        sessions,
        session_date_col=session_date_col,
        market_close_col=market_close_col,
    )
    if frame is None or frame.empty:
        result = (
            frame.copy()
            if isinstance(frame, pd.DataFrame)
            else _empty_canonical_frame()
        )
        result["feature_date"] = pd.Series(dtype="datetime64[ns]")
        result["feature_close_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        result["mapping_status"] = pd.Series(dtype="object")
        result["mapping_rule"] = pd.Series(dtype="object")
        return result

    data = frame.copy().reset_index(drop=True)
    published_at = pd.to_datetime(
        data.get("published_at_utc"), errors="coerce", utc=True
    )
    published_date = pd.to_datetime(
        data.get("published_date"), errors="coerce"
    ).dt.tz_localize(None).dt.normalize()
    precision = data.get(
        "timestamp_precision",
        pd.Series(
            np.where(published_at.notna(), "timestamp", "date"),
            index=data.index,
        ),
    ).astype(str).str.lower()

    # Force nanoseconds explicitly: pandas 3 may otherwise expose microsecond
    # integer storage while ``Timestamp.value`` remains nanoseconds.
    close_values = calendar[market_close_col].to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    session_values = calendar[session_date_col].to_numpy(dtype="datetime64[ns]")
    mapped_dates: list[pd.Timestamp | pd.NaT] = []
    mapped_closes: list[pd.Timestamp | pd.NaT] = []
    statuses: list[str] = []
    rules: list[str] = []

    for index in data.index:
        is_date_only = precision.iloc[index] == "date" or pd.isna(
            published_at.iloc[index]
        )
        if is_date_only and pd.notna(published_date.iloc[index]):
            publication_day = np.datetime64(
                pd.Timestamp(published_date.iloc[index]).to_datetime64(), "ns"
            )
            location = int(np.searchsorted(session_values, publication_day, side="right"))
            rule = "date-only -> next session"
        elif pd.notna(published_at.iloc[index]):
            publication_ns = int(pd.Timestamp(published_at.iloc[index]).value)
            location = int(np.searchsorted(close_values, publication_ns, side="right"))
            rule = "timestamp -> first close strictly after publication"
        else:
            mapped_dates.append(pd.NaT)
            mapped_closes.append(pd.NaT)
            statuses.append("missing_publication_time")
            rules.append("unmapped")
            continue

        if location >= len(calendar):
            mapped_dates.append(pd.NaT)
            mapped_closes.append(pd.NaT)
            statuses.append("after_calendar")
            rules.append(rule)
            continue
        mapped_dates.append(pd.Timestamp(calendar.iloc[location][session_date_col]))
        mapped_closes.append(pd.Timestamp(calendar.iloc[location][market_close_col]))
        statuses.append("mapped")
        rules.append(rule)

    data["feature_date"] = pd.to_datetime(mapped_dates, errors="coerce")
    data["feature_close_utc"] = pd.to_datetime(
        mapped_closes, errors="coerce", utc=True
    )
    data["mapping_status"] = statuses
    data["mapping_rule"] = rules
    return data


def _session_date_values(session_dates: Any) -> pd.DatetimeIndex:
    if session_dates is None:
        return pd.DatetimeIndex([])
    if isinstance(session_dates, pd.DataFrame):
        for candidate in ["session_date", "Date", "date", "feature_date"]:
            if candidate in session_dates.columns:
                values = session_dates[candidate]
                break
        else:
            if isinstance(session_dates.index, pd.DatetimeIndex):
                values = session_dates.index
            else:
                raise ValueError("Could not identify session dates.")
    else:
        values = session_dates
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).dropna().tz_localize(
        None
    ).normalize().drop_duplicates().sort_values()


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan
    selected_values = values.loc[valid].astype(float)
    selected_weights = weights.loc[valid].astype(float).clip(lower=0)
    if float(selected_weights.sum()) <= 0:
        selected_weights = pd.Series(1.0, index=selected_values.index)
    return float(np.average(selected_values, weights=selected_weights))


def aggregate_daily_news(
    frame: pd.DataFrame,
    *,
    session_dates: Any = None,
    assets: Iterable[str] | None = None,
    abnormal_window: int = 20,
) -> pd.DataFrame:
    """Aggregate mapped articles while keeping attention and tone distinct."""
    if abnormal_window < 2:
        raise ValueError("abnormal_window must be at least 2.")
    required = {"asset", "feature_date", "title", "url", "source", "relevance_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing daily-news columns: {sorted(missing)}")

    mapped = frame.loc[
        frame["feature_date"].notna()
        & frame.get("mapping_status", pd.Series("mapped", index=frame.index)).eq("mapped")
    ].copy()
    mapped["feature_date"] = pd.to_datetime(mapped["feature_date"]).dt.tz_localize(
        None
    ).dt.normalize()
    mapped = deduplicate_news_articles(mapped)

    probability_columns = [
        "positive_probability",
        "negative_probability",
        "neutral_probability",
    ]
    for column in probability_columns:
        if column not in mapped.columns:
            mapped[column] = np.nan
        mapped[column] = pd.to_numeric(mapped[column], errors="coerce")
    mapped["relevance_score"] = pd.to_numeric(
        mapped["relevance_score"], errors="coerce"
    )

    rows: list[dict[str, Any]] = []
    for (asset, feature_date), group in mapped.groupby(
        ["asset", "feature_date"], sort=True
    ):
        relevance = group["relevance_score"].fillna(1.0).clip(lower=0)
        scored_mask = group[probability_columns].notna().all(axis=1)
        if scored_mask.any():
            probability_sums = group.loc[scored_mask, probability_columns].sum(axis=1)
            if not np.allclose(probability_sums, 1.0, atol=1e-5):
                raise ValueError("FinBERT probabilities must sum to one per article.")
            scored = group.loc[scored_mask]
            scored_weights = relevance.loc[scored.index]
            positive = _weighted_average(
                scored["positive_probability"], scored_weights
            )
            negative = _weighted_average(
                scored["negative_probability"], scored_weights
            )
            neutral = _weighted_average(
                scored["neutral_probability"], scored_weights
            )
            article_scores = (
                scored["positive_probability"] - scored["negative_probability"]
            )
            labels = scored[probability_columns].to_numpy().argmax(axis=1)
            balance = positive - negative
            dispersion = float(article_scores.std(ddof=0))
            positive_share = float(np.mean(labels == 0))
            negative_share = float(np.mean(labels == 1))
            neutral_share = float(np.mean(labels == 2))
        else:
            positive = negative = neutral = np.nan
            balance = dispersion = np.nan
            positive_share = negative_share = neutral_share = np.nan

        nonempty_sources = group["source"].map(_clean_text)
        rows.append(
            {
                "asset": asset,
                "Date": pd.Timestamp(feature_date),
                "Unique_Story_Count": int(len(group)),
                "Log_News_Count": float(np.log1p(len(group))),
                "Unique_Source_Count": int(
                    nonempty_sources[nonempty_sources.ne("")].nunique()
                ),
                "News_Relevance_Sum": float(relevance.sum()),
                "FinBERT_Scored_Article_Count": int(scored_mask.sum()),
                "FinBERT_Sentiment_Missing": int(not scored_mask.any()),
                "FinBERT_Positive_Share": positive_share,
                "FinBERT_Negative_Share": negative_share,
                "FinBERT_Neutral_Share": neutral_share,
                "FinBERT_Sentiment_Balance": balance,
                "FinBERT_Disagreement": dispersion,
                "FinBERT_Max_Negative": (
                    float(scored["negative_probability"].max())
                    if scored_mask.any()
                    else np.nan
                ),
            }
        )
    daily = pd.DataFrame(rows)

    source_assets = (
        list(assets)
        if assets is not None
        else mapped.get("asset", pd.Series(dtype=str)).tolist()
    )
    asset_values = sorted(
        {str(value).upper() for value in source_assets if str(value)}
    )
    calendar_dates = _session_date_values(session_dates)
    if len(calendar_dates) and asset_values:
        grid = pd.MultiIndex.from_product(
            [asset_values, calendar_dates], names=["asset", "Date"]
        ).to_frame(index=False)
        if daily.empty:
            daily = pd.DataFrame(
                columns=[
                    "asset",
                    "Date",
                    "Unique_Story_Count",
                    "Log_News_Count",
                    "Unique_Source_Count",
                    "News_Relevance_Sum",
                    "FinBERT_Scored_Article_Count",
                    "FinBERT_Sentiment_Missing",
                    "FinBERT_Positive_Share",
                    "FinBERT_Negative_Share",
                    "FinBERT_Neutral_Share",
                    "FinBERT_Sentiment_Balance",
                    "FinBERT_Disagreement",
                    "FinBERT_Max_Negative",
                ]
            )
        daily = grid.merge(daily, on=["asset", "Date"], how="left")
    elif daily.empty:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Date",
                *ATTENTION_FEATURE_COLUMNS,
                *SENTIMENT_FEATURE_COLUMNS,
                *NEWS_AUDIT_FEATURE_COLUMNS,
            ]
        )

    count_columns = [
        "Unique_Story_Count",
        "Log_News_Count",
        "Unique_Source_Count",
        "News_Relevance_Sum",
        "FinBERT_Scored_Article_Count",
    ]
    for column in count_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
    daily["No_News"] = daily["Unique_Story_Count"].eq(0).astype(int)
    missing_sentiment = pd.to_numeric(
        daily["FinBERT_Sentiment_Missing"], errors="coerce"
    )
    daily["FinBERT_Sentiment_Missing"] = np.where(
        daily["No_News"].eq(1),
        0,
        missing_sentiment.fillna(1),
    ).astype(int)

    base_sentiment_columns = [
        "FinBERT_Positive_Share",
        "FinBERT_Negative_Share",
        "FinBERT_Neutral_Share",
        "FinBERT_Sentiment_Balance",
        "FinBERT_Disagreement",
        "FinBERT_Max_Negative",
    ]
    for column in base_sentiment_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    no_news_mask = daily["No_News"].eq(1)
    daily.loc[no_news_mask, base_sentiment_columns] = 0.0

    outputs: list[pd.DataFrame] = []
    minimum_history = min(5, abnormal_window)
    for _, group in daily.sort_values(["asset", "Date"]).groupby(
        "asset", sort=False
    ):
        result = group.copy()
        prior_count = result["Unique_Story_Count"].shift(1)
        count_mean = prior_count.rolling(
            abnormal_window, min_periods=minimum_history
        ).mean()
        count_std = prior_count.rolling(
            abnormal_window, min_periods=minimum_history
        ).std(ddof=0)
        abnormal = (
            (result["Unique_Story_Count"] - count_mean)
            / count_std.replace(0, np.nan)
        )
        result["Abnormal_News_Volume_20"] = abnormal.fillna(0.0)

        balance_for_history = result["FinBERT_Sentiment_Balance"].fillna(0.0)
        result["FinBERT_Sentiment_3D"] = balance_for_history.ewm(
            span=3, adjust=False
        ).mean()
        prior_balance_mean = balance_for_history.shift(1).rolling(
            abnormal_window, min_periods=minimum_history
        ).mean()
        result["FinBERT_Sentiment_Surprise_20"] = (
            balance_for_history - prior_balance_mean
        ).fillna(0.0)
        result.loc[
            result["FinBERT_Sentiment_Missing"].eq(1),
            ["FinBERT_Sentiment_3D", "FinBERT_Sentiment_Surprise_20"],
        ] = np.nan
        outputs.append(result)

    final = pd.concat(outputs, ignore_index=True).sort_values(
        ["asset", "Date"]
    ).reset_index(drop=True)
    final = final.rename(columns={"asset": "Asset"})
    ordered = [
        "Asset",
        "Date",
        *ATTENTION_FEATURE_COLUMNS,
        *SENTIMENT_FEATURE_COLUMNS,
        *NEWS_AUDIT_FEATURE_COLUMNS,
    ]
    return final[ordered]


def news_coverage_audit(
    frame: pd.DataFrame,
    *,
    sessions: pd.DataFrame | None = None,
    session_date_col: str = "session_date",
    market_close_col: str = "market_close_utc",
) -> pd.DataFrame:
    """Summarize usable coverage, timestamp quality, and exact duplicates."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Providers",
                "Raw Articles",
                "Unique Articles",
                "Exact Duplicate Rows",
                "Duplicate Rate",
                "Timestamped Articles",
                "Date-only Articles",
                "Missing Publication Time",
                "Mapped Articles",
                "First Feature Date",
                "Last Feature Date",
                "News Sessions",
                "Eligible Sessions",
                "Session Coverage",
                "Missing Title Rate",
                "Missing URL Rate",
            ]
        )

    raw = frame.copy()
    mapped = raw
    calendar: pd.DataFrame | None = None
    if sessions is not None:
        calendar = _prepare_sessions(
            sessions,
            session_date_col=session_date_col,
            market_close_col=market_close_col,
        )
        if "feature_date" not in mapped.columns:
            mapped = map_news_to_sessions(
                mapped,
                calendar,
                session_date_col=session_date_col,
                market_close_col=market_close_col,
            )
    unique = deduplicate_news_articles(mapped)
    rows: list[dict[str, Any]] = []
    for asset, raw_group in raw.groupby("asset", sort=True):
        unique_group = unique.loc[unique["asset"].eq(asset)].copy()
        if "feature_date" in unique_group:
            feature_dates = pd.to_datetime(
                unique_group["feature_date"], errors="coerce"
            ).dropna()
        else:
            publication_dates = pd.to_datetime(
                unique_group.get(
                    "published_date",
                    pd.Series(pd.NaT, index=unique_group.index),
                ),
                errors="coerce",
            )
            timestamp_dates = pd.to_datetime(
                unique_group.get(
                    "published_at_utc",
                    pd.Series(pd.NaT, index=unique_group.index),
                ),
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None).dt.normalize()
            feature_dates = publication_dates.fillna(timestamp_dates).dropna()
        first_date = feature_dates.min() if not feature_dates.empty else pd.NaT
        last_date = feature_dates.max() if not feature_dates.empty else pd.NaT
        eligible_sessions = np.nan
        if calendar is not None and pd.notna(first_date) and pd.notna(last_date):
            eligible_sessions = int(
                calendar[session_date_col].between(first_date, last_date).sum()
            )
        news_sessions = int(feature_dates.nunique())
        session_coverage = (
            news_sessions / eligible_sessions
            if pd.notna(eligible_sessions) and eligible_sessions > 0
            else np.nan
        )
        precision = raw_group.get(
            "timestamp_precision", pd.Series("missing", index=raw_group.index)
        )
        mapped_count = (
            int(unique_group["feature_date"].notna().sum())
            if "feature_date" in unique_group
            else 0
        )
        rows.append(
            {
                "Asset": asset,
                "Providers": ", ".join(
                    sorted(set(raw_group.get("provider", pd.Series(dtype=str)).dropna()))
                ),
                "Raw Articles": int(len(raw_group)),
                "Unique Articles": int(len(unique_group)),
                "Exact Duplicate Rows": int(len(raw_group) - len(unique_group)),
                "Duplicate Rate": float(
                    (len(raw_group) - len(unique_group)) / max(len(raw_group), 1)
                ),
                "Timestamped Articles": int(precision.eq("timestamp").sum()),
                "Date-only Articles": int(precision.eq("date").sum()),
                "Missing Publication Time": int(precision.eq("missing").sum()),
                "Mapped Articles": mapped_count,
                "First Feature Date": first_date,
                "Last Feature Date": last_date,
                "News Sessions": news_sessions,
                "Eligible Sessions": eligible_sessions,
                "Session Coverage": session_coverage,
                "Missing Title Rate": float(
                    raw_group["title"].map(_clean_text).eq("").mean()
                ),
                "Missing URL Rate": float(
                    raw_group["url"].map(_clean_text).eq("").mean()
                ),
            }
        )
    return pd.DataFrame(rows)
