"""Download reproducible stock, market, macroeconomic, and sentiment snapshots.

The generated CSV files are committed with the project so the dashboard and
experiments do not require an internet connection. Running this script refreshes
the snapshots and writes a source manifest with the retrieval time and
availability rule used by the feature pipeline.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONTEXT_DIR = PROJECT_ROOT / "data" / "context"
START_DATE = date(2016, 7, 21)
END_DATE = date.today()
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148 Safari/537.36"
)

TARGET_ASSETS = {
    "TSLA": "stocks",
    "AAPL": "stocks",
    "NVDA": "stocks",
}
CONTEXT_ASSETS = {"QQQ": "etf"}

FRED_SERIES = {
    "SP500": {
        "frequency": "Daily close",
        "group": "Market index",
        "availability_rule": "Same trading date; used after the market close.",
    },
    "NASDAQCOM": {
        "frequency": "Daily close",
        "group": "Market index",
        "availability_rule": "Same trading date; used after the market close.",
    },
    "DFF": {
        "frequency": "Daily",
        "group": "Macroeconomic",
        "availability_rule": "Shifted by one calendar day before as-of merging.",
    },
    "DGS10": {
        "frequency": "Daily",
        "group": "Macroeconomic",
        "availability_rule": "Same trading date; market yield known after the close.",
    },
    "DGS2": {
        "frequency": "Daily",
        "group": "Macroeconomic",
        "availability_rule": (
            "Same trading date; combined with DGS10 locally to derive the 10Y-2Y spread."
        ),
    },
    "CPIAUCSL": {
        "frequency": "Monthly",
        "group": "Macroeconomic",
        "availability_rule": "Available from the 15th day of the following month.",
    },
    "UNRATE": {
        "frequency": "Monthly",
        "group": "Macroeconomic",
        "availability_rule": "Available from the 8th day of the following month.",
    },
    "VIXCLS": {
        "frequency": "Daily close",
        "group": "Sentiment proxy",
        "availability_rule": "Same trading date; used after the market close.",
    },
    "UMCSENT": {
        "frequency": "Monthly",
        "group": "Sentiment proxy",
        "availability_rule": "Available at the end of the following month.",
    },
}


def _get_text(url: str, *, headers: dict[str, str] | None = None) -> str:
    request = Request(url, headers=headers or {"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def download_nasdaq_history(symbol: str, asset_class: str) -> pd.DataFrame:
    """Download the full Nasdaq historical table using ISO date parameters."""
    query = urlencode(
        {
            "assetclass": asset_class,
            "fromdate": START_DATE.isoformat(),
            "todate": END_DATE.isoformat(),
            "limit": 5000,
        }
    )
    url = f"https://api.nasdaq.com/api/quote/{symbol}/historical?{query}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": (
            f"https://www.nasdaq.com/market-activity/"
            f"{'etf' if asset_class == 'etf' else 'stocks'}/{symbol.lower()}/historical"
        ),
    }
    payload = json.loads(_get_text(url, headers=headers))
    data = payload.get("data")
    if not data or not data.get("tradesTable", {}).get("rows"):
        raise RuntimeError(f"Nasdaq did not return historical rows for {symbol}.")
    rows = pd.DataFrame(data["tradesTable"]["rows"])
    result = rows.rename(
        columns={
            "date": "Date",
            "close": "Close/Last",
            "volume": "Volume",
            "open": "Open",
            "high": "High",
            "low": "Low",
        }
    )
    return result[["Date", "Close/Last", "Volume", "Open", "High", "Low"]]


def download_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(StringIO(_get_text(url)))
    frame.columns = ["Date", series_id]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    earliest = pd.Timestamp(START_DATE) - pd.DateOffset(years=2)
    return frame.loc[frame["Date"].ge(earliest)].reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest_rows: list[dict[str, object]] = []

    for symbol, asset_class in {**TARGET_ASSETS, **CONTEXT_ASSETS}.items():
        frame = download_nasdaq_history(symbol, asset_class)
        destination = (
            RAW_DIR / f"{symbol}_raw_2016_2026.csv"
            if symbol in TARGET_ASSETS
            else CONTEXT_DIR / f"{symbol}_raw_2016_2026.csv"
        )
        frame.to_csv(destination, index=False)
        page_kind = "etf" if asset_class == "etf" else "stocks"
        manifest_rows.append(
            {
                "Dataset": symbol,
                "Group": "Target asset" if symbol in TARGET_ASSETS else "Market proxy",
                "Frequency": "Daily OHLCV",
                "File": destination.relative_to(PROJECT_ROOT).as_posix(),
                "Source Page": (
                    f"https://www.nasdaq.com/market-activity/{page_kind}/"
                    f"{symbol.lower()}/historical"
                ),
                "Rows": len(frame),
                "First Observation": pd.to_datetime(frame["Date"]).min().date(),
                "Last Observation": pd.to_datetime(frame["Date"]).max().date(),
                "Availability Rule": "Same trading date; used after the market close.",
                "Timezone": "America/New_York market date",
                "Adjustment": "Provider-delivered OHLCV; verify corporate actions before reuse.",
                "Licence Note": (
                    "Private academic evaluation only; review Nasdaq terms before public deployment."
                ),
                "SHA256": sha256_file(destination),
                "Retrieved UTC": retrieved_at,
            }
        )

    for series_id, metadata in FRED_SERIES.items():
        frame = download_fred_series(series_id)
        destination = CONTEXT_DIR / f"{series_id}.csv"
        frame.to_csv(destination, index=False)
        manifest_rows.append(
            {
                "Dataset": series_id,
                "Group": metadata["group"],
                "Frequency": metadata["frequency"],
                "File": destination.relative_to(PROJECT_ROOT).as_posix(),
                "Source Page": f"https://fred.stlouisfed.org/series/{series_id}",
                "Rows": len(frame),
                "First Observation": frame["Date"].min().date(),
                "Last Observation": frame["Date"].max().date(),
                "Availability Rule": metadata["availability_rule"],
                "Timezone": "Source observation date",
                "Adjustment": "Not applicable",
                "Licence Note": (
                    "Cite the originating FRED series and review its series-specific notes."
                ),
                "SHA256": sha256_file(destination),
                "Retrieved UTC": retrieved_at,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(CONTEXT_DIR / "source_manifest.csv", index=False)
    print(
        f"Downloaded {len(TARGET_ASSETS)} target assets, {len(CONTEXT_ASSETS)} "
        f"market proxy, and {len(FRED_SERIES)} FRED series."
    )
    print(f"Snapshot date: {END_DATE.isoformat()}")
    print(f"Manifest: {CONTEXT_DIR / 'source_manifest.csv'}")


if __name__ == "__main__":
    main()
