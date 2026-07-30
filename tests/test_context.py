import hashlib
from pathlib import Path

import pandas as pd

from src.config import CANDIDATE_FEATURES, DATASET_PATHS, PROJECT_ROOT
from src.context import build_context_frame, load_source_manifest
from src.data import load_and_preprocess_csv
from src.features import add_features


def test_context_uses_backward_availability_only(cleaned_data):
    result = build_context_frame(cleaned_data["Date"])
    assert not result.audit.empty
    assert (result.audit["Future Availability Violations"] == 0).all()
    assert set(result.audit["Status"]) == {"Pass"}


def test_examiner_requested_context_features_are_created(cleaned_data):
    context = build_context_frame(cleaned_data["Date"])
    engineered = add_features(cleaned_data, context_data=context.frame)
    requested = {
        "SP500_Return",
        "NASDAQ_Return",
        "Fed_Funds_Rate",
        "Treasury_10Y_Yield",
        "Yield_Curve_10Y2Y",
        "CPI_YoY",
        "Unemployment_Rate",
        "VIX_Level",
        "Consumer_Sentiment",
    }
    assert requested.issubset(engineered.columns)
    assert requested.issubset(CANDIDATE_FEATURES)
    assert engineered[list(requested)].dropna().shape[0] >= 100


def test_monthly_context_does_not_backfill_before_availability():
    dates = pd.Series(pd.date_range("2026-06-01", "2026-07-20", freq="B"))
    frame = build_context_frame(dates).frame
    june_cpi_rows = frame.loc[frame["CPIAUCSL_Available_Date"].eq(pd.Timestamp("2026-07-15"))]
    assert not june_cpi_rows.empty
    assert june_cpi_rows["Date"].min() >= pd.Timestamp("2026-07-15")


def test_split_adjusted_snapshots_do_not_create_false_split_crashes():
    checks = {
        "AAPL": pd.Timestamp("2020-08-31"),
        "NVDA": pd.Timestamp("2024-06-10"),
    }
    for asset, split_date in checks.items():
        cleaned, _ = load_and_preprocess_csv(DATASET_PATHS[asset])
        row = cleaned.loc[cleaned["Date"].eq(split_date), "Daily_Return"]
        assert not row.empty
        assert abs(float(row.iloc[0])) < 0.50


def test_source_manifest_hashes_match_packaged_snapshots():
    manifest = load_source_manifest()
    assert not manifest.empty
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / Path(getattr(row, "File"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == getattr(row, "SHA256")
