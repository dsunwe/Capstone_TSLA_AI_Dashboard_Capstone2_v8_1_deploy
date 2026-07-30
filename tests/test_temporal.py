import pandas as pd
import pytest

from src.config import (
    TARGET_DATE_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
    TARGET_VALUATION_DATE_COLUMN,
)
from src.temporal import (
    build_temporal_split_plan,
    rows_for_dates,
    split_frame_by_plan,
)


def test_canonical_plan_uses_prespecified_warmup_and_known_targets(engineered_data):
    plan = build_temporal_split_plan(engineered_data, max_lookback=50)
    chronological = engineered_data.sort_values("Date").reset_index(drop=True)
    expected = chronological.iloc[49:]
    expected = expected[
        expected[[TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN]].notna().all(axis=1)
    ]

    assert plan.eligible_dates == tuple(pd.to_datetime(expected["Date"]))
    assert plan.train_dates + plan.gap_dates + plan.test_dates == plan.eligible_dates
    assert plan.gap == TARGET_PURGE_ROWS
    assert len(plan.gap_dates) == TARGET_PURGE_ROWS
    assert (
        plan.train_end_date
        < plan.gap_start_date
        <= plan.gap_end_date
        < plan.test_start_date
    )
    assert plan.plan_id.endswith(plan.plan_hash[:16])


def test_training_target_dates_end_at_the_five_row_purge_boundary(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    training = rows_for_dates(
        engineered_data,
        plan.train_dates,
        required_columns=[TARGET_DATE_COLUMN, TARGET_VALUATION_DATE_COLUMN],
    )

    target_dates = pd.to_datetime(training[TARGET_DATE_COLUMN])
    decision_valuation_dates = pd.to_datetime(
        training[TARGET_VALUATION_DATE_COLUMN]
    )
    assert target_dates.notna().all()
    assert decision_valuation_dates.notna().all()
    assert target_dates.max() == plan.gap_end_date
    assert target_dates.max() < plan.test_start_date
    assert decision_valuation_dates.max() == plan.test_start_date


def test_plan_rejects_a_purge_shorter_than_the_five_session_target_horizon(
    engineered_data,
):
    with pytest.raises(ValueError, match="at least 5"):
        build_temporal_split_plan(engineered_data, gap=TARGET_PURGE_ROWS - 1)


def test_plan_hash_does_not_depend_on_feature_values_missingness_or_target_values(
    engineered_data,
):
    baseline = build_temporal_split_plan(engineered_data)
    changed = engineered_data.copy()
    changed.loc[changed.index[70:140], "SMA10_Ratio"] = pd.NA
    test_mask = changed["Date"].isin(baseline.test_dates)
    changed.loc[test_mask, TARGET_DIRECTION_COLUMN] = (
        1 - changed.loc[test_mask, TARGET_DIRECTION_COLUMN]
    )
    changed.loc[test_mask, "Momentum_5"] = 1_000_000

    repeated = build_temporal_split_plan(changed)
    assert repeated.plan_hash == baseline.plan_hash
    assert repeated.eligible_dates == baseline.eligible_dates
    assert repeated.test_dates == baseline.test_dates


def test_multi_asset_plan_uses_exact_common_eligible_dates(engineered_data):
    primary = engineered_data.copy()
    secondary = engineered_data.copy()
    removed_date = pd.Timestamp(secondary.iloc[120]["Date"])
    secondary = secondary[secondary["Date"] != removed_date]

    plan = build_temporal_split_plan({"A": primary, "B": secondary})

    assert removed_date not in plan.eligible_dates
    assert set(plan.eligible_dates).issubset(set(pd.to_datetime(primary["Date"])))
    assert set(plan.eligible_dates).issubset(set(pd.to_datetime(secondary["Date"])))


def test_rows_for_dates_preserves_exact_plan_dates_without_feature_row_drops(
    engineered_data,
):
    plan = build_temporal_split_plan(engineered_data)
    changed = engineered_data.copy()
    changed.loc[changed["Date"].isin(plan.train_dates[:8]), "Momentum_5"] = pd.NA

    train, gap, test = split_frame_by_plan(
        changed,
        plan,
        required_columns=["Momentum_5", TARGET_DIRECTION_COLUMN],
    )

    assert tuple(train["Date"]) == plan.train_dates
    assert tuple(gap["Date"]) == plan.gap_dates
    assert tuple(test["Date"]) == plan.test_dates
    assert len(train) == len(plan.train_dates)
    assert train["Momentum_5"].isna().sum() >= 8


def test_rows_for_dates_fails_closed_when_plan_date_is_missing(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    missing = engineered_data[engineered_data["Date"] != plan.test_dates[0]]

    with pytest.raises(ValueError, match="does not cover all dates"):
        rows_for_dates(missing, plan.test_dates)


def test_plan_manifest_is_complete_and_self_identifying(engineered_data):
    plan = build_temporal_split_plan(engineered_data)
    manifest = plan.manifest()

    assert len(manifest) == len(plan.eligible_dates)
    assert manifest["Plan ID"].nunique() == 1
    assert manifest["Plan Hash"].nunique() == 1
    assert set(manifest["Partition"]) == {"Train", "Purge", "Test"}
