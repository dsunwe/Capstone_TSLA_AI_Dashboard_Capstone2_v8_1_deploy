import copy

from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
import numpy as np
import pandas as pd
import pytest

from src.config import (
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_VALUATION_DATE_COLUMN,
)
from src.confirmatory import (
    assert_final_metrics_allowed,
    load_prospective_protocol,
    prospective_status_table,
    select_prospective_rows,
    validate_prospective_ledger,
    validate_prospective_protocol,
)


def _locked_protocol() -> dict:
    protocol = load_prospective_protocol()
    protocol["protocol_id"] = "TSLA-V8-FIVE-SESSION-LOCK-001"
    protocol["status"] = "locked_accruing"
    protocol["external_lock"] = {
        "evidence": "Supervisor acknowledgement reference SUP-2026-07-30",
        "timestamp_utc": "2026-07-30T12:00:00+00:00",
        "verified_by": "Capstone supervisor",
        "first_eligible_signal_date": "2026-07-31",
        "first_eligible_signal_close_utc": "2026-07-31T20:00:00+00:00",
    }
    for index, field in enumerate(
        (
            "model_artifact_hash",
            "feature_set_hash",
            "code_release_hash",
            "preprocessor_hash",
            "training_data_snapshot_hash",
        ),
        start=1,
    ):
        protocol["locked_design"][field] = f"{index:x}" * 64
    validate_prospective_protocol(protocol)
    return protocol


def _ledger(rows: int) -> pd.DataFrame:
    calendar = XNYSExchangeCalendar(
        start="2026-07-20",
        end="2027-04-30",
    )
    sessions = pd.DatetimeIndex(calendar.schedule.index)
    start = sessions.get_loc(pd.Timestamp("2026-07-31"))
    selected = sessions[start : start + rows + 6]
    signal = (np.arange(rows) % 2).astype(int)
    realised = np.where(signal == 1, 0.01, -0.01)
    return pd.DataFrame(
        {
            "Date": selected[:rows],
            TARGET_EXECUTION_DATE_COLUMN: selected[1 : rows + 1],
            TARGET_VALUATION_DATE_COLUMN: selected[6 : rows + 6],
            TARGET_DIRECTION_COLUMN: signal,
            TARGET_RETURN_COLUMN: realised,
            "Predicted Probability": np.where(signal == 1, 0.70, 0.30),
            "Signal": signal,
        }
    )


def test_packaged_prospective_protocol_is_honestly_draft():
    protocol = load_prospective_protocol()
    status = prospective_status_table(protocol).iloc[0]
    assert protocol["status"] == "draft_pending_external_timestamp"
    assert protocol["retrospective_evidence"]["may_be_relabelled_confirmatory"] is False
    assert status["Final Metrics Allowed"] == False
    assert status["External Timestamp Evidence"] == "Not supplied"
    with pytest.raises(RuntimeError, match="externally locked"):
        select_prospective_rows(
            pd.DataFrame({"Date": pd.date_range("2026-08-01", periods=3)}),
            protocol,
        )


def test_lock_requires_real_timestamps_design_hashes_and_future_boundary():
    protocol = load_prospective_protocol()
    protocol["status"] = "locked_accruing"
    with pytest.raises(ValueError, match="must not contain DRAFT"):
        validate_prospective_protocol(protocol)

    valid = _locked_protocol()
    no_timezone = copy.deepcopy(valid)
    no_timezone["external_lock"]["timestamp_utc"] = "2026-07-30T12:00:00"
    with pytest.raises(ValueError, match="explicit timezone"):
        validate_prospective_protocol(no_timezone)

    old_signal = copy.deepcopy(valid)
    old_signal["external_lock"]["timestamp_utc"] = "2026-07-19T12:00:00+00:00"
    old_signal["external_lock"]["first_eligible_signal_date"] = "2026-07-20"
    old_signal["external_lock"][
        "first_eligible_signal_close_utc"
    ] = "2026-07-20T20:00:00+00:00"
    with pytest.raises(ValueError, match="historical evidence cutoff"):
        validate_prospective_protocol(old_signal)

    missing_hash = copy.deepcopy(valid)
    missing_hash["locked_design"]["model_artifact_hash"] = None
    with pytest.raises(ValueError, match="valid SHA-256"):
        validate_prospective_protocol(missing_hash)


def test_minimum_is_fixed_and_complete_status_is_derived_only():
    protocol = load_prospective_protocol()
    protocol["stopping_rule"]["minimum_consecutive_labelled_signals"] = 59
    protocol["stopping_rule"]["confirmatory_analysis_at"] = 59
    with pytest.raises(ValueError, match="fixed at 60"):
        validate_prospective_protocol(protocol)

    protocol = _locked_protocol()
    protocol["status"] = "complete"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_prospective_protocol(protocol)


def test_final_metrics_require_sixty_real_consecutive_complete_rows():
    protocol = _locked_protocol()
    with pytest.raises(RuntimeError, match="prespecified observation count"):
        assert_final_metrics_allowed(protocol, ledger=_ledger(59))

    ledger = _ledger(61)
    analysis_rows = assert_final_metrics_allowed(protocol, ledger=ledger)
    assert len(analysis_rows) == 60
    status = prospective_status_table(protocol, ledger=ledger).iloc[0]
    assert status["Status"] == "complete"
    assert status["Accrued Labelled Signals"] == 61
    assert status["Final Metrics Allowed"] == True


def test_ledger_rejects_duplicates_gaps_missing_labels_and_bad_timing():
    protocol = _locked_protocol()
    duplicate = _ledger(60)
    duplicate.loc[1, "Date"] = duplicate.loc[0, "Date"]
    with pytest.raises(ValueError, match="unique"):
        validate_prospective_ledger(duplicate, protocol)

    gap = _ledger(60).drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="consecutive XNYS"):
        validate_prospective_ledger(gap, protocol)

    missing = _ledger(60)
    missing.loc[5, TARGET_RETURN_COLUMN] = np.nan
    with pytest.raises(ValueError, match="returns must be finite"):
        validate_prospective_ledger(missing, protocol)

    bad_execution = _ledger(60)
    bad_execution.loc[8, TARGET_EXECUTION_DATE_COLUMN] = bad_execution.loc[
        8, TARGET_VALUATION_DATE_COLUMN
    ]
    with pytest.raises(ValueError, match="Execution dates"):
        validate_prospective_ledger(bad_execution, protocol)

    bad_valuation = _ledger(60)
    bad_valuation.loc[8, TARGET_VALUATION_DATE_COLUMN] = bad_valuation.loc[
        8, TARGET_EXECUTION_DATE_COLUMN
    ]
    with pytest.raises(ValueError, match="six XNYS sessions"):
        validate_prospective_ledger(bad_valuation, protocol)


def test_locked_row_selection_starts_at_the_external_boundary():
    protocol = _locked_protocol()
    rows = select_prospective_rows(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2026-07-30", "2026-07-31", "2026-08-03"]
                ),
                "Signal": [0, 1, 0],
            }
        ),
        protocol,
    )
    assert rows["Date"].min() == pd.Timestamp("2026-07-31")
