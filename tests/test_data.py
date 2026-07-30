import pandas as pd

from src.data import (
    align_datasets_to_common_window,
    load_builtin_datasets,
    preprocessing_audit_table,
    preprocess_stock_data,
    split_continuity_audit,
)


def test_preprocess_supports_nasdaq_close_last():
    raw = pd.DataFrame(
        {
            "Date": ["01/03/2024", "01/02/2024"],
            "Close/Last": ["$250.50", "$245.00"],
            "Volume": ["1,000", "2,000"],
            "Open": ["$248.00", "$240.00"],
            "High": ["$252.00", "$247.00"],
            "Low": ["$246.00", "$239.00"],
        }
    )
    cleaned, profile = preprocess_stock_data(raw)
    assert list(cleaned["Date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert cleaned["Close"].tolist() == [245.0, 250.5]
    assert cleaned["Price"].equals(cleaned["Close"])
    assert cleaned["Volume"].tolist() == [2000, 1000]
    assert profile["raw_feature_count"] == 6
    assert profile["source_price_field"] == "Close/Last"
    assert profile["uses_explicit_adjusted_close"] is False


def test_preprocess_removes_duplicate_dates():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "Open": [1, 1, 2],
            "High": [2, 2, 3],
            "Low": [0.5, 0.5, 1.5],
            "Close": [1.5, 1.6, 2.5],
            "Volume": [100, 110, 120],
        }
    )
    cleaned, profile = preprocess_stock_data(raw)
    assert len(cleaned) == 2
    assert profile["duplicate_dates_before_removal"] == 1
    assert cleaned.loc[0, "Close"] == 1.6


def test_preprocessing_audit_exposes_raw_to_clean_steps():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02"],
            "Close/Last": ["$10.00", "$11.00"],
            "Volume": ["1,000", "2,000"],
            "Open": ["$9.00", "$10.00"],
            "High": ["$11.00", "$12.00"],
            "Low": ["$8.00", "$9.00"],
        }
    )
    _, profile = preprocess_stock_data(raw)
    audit = preprocessing_audit_table(profile)
    assert {"Schema mapping", "Numeric conversion", "Duplicate dates"}.issubset(
        set(audit["Step"])
    )
    assert isinstance(profile["raw_preview"], pd.DataFrame)


def test_bundled_external_validation_datasets_share_a_common_window():
    datasets, summary = load_builtin_datasets()
    aligned, common_start, common_end = align_datasets_to_common_window(datasets)
    assert set(datasets) == {"TSLA", "AAPL", "NVDA"}
    assert set(summary["Asset"]) == {"TSLA", "AAPL", "NVDA"}
    assert common_start < common_end
    assert all(frame["Date"].min() >= common_start for frame in aligned.values())
    assert all(frame["Date"].max() <= common_end for frame in aligned.values())


def test_tsla_split_continuity_audit_is_empirically_clean():
    datasets, _ = load_builtin_datasets()
    audit = split_continuity_audit(datasets["TSLA"], "TSLA")

    assert list(audit["Effective Date"].dt.strftime("%Y-%m-%d")) == [
        "2020-08-31",
        "2022-08-25",
    ]
    assert audit["Continuity Status"].eq(
        "Consistent with split-adjusted continuity"
    ).all()
    assert audit["Observed One-Day Return"].abs().max() < 0.50
