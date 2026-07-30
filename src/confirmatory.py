"""Prospective-confirmation guardrails.

The current historical outputs are retrospective development evidence.  This
module prevents a draft local protocol from being presented as externally
locked or complete.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
import numpy as np
import pandas as pd

from .config import (
    PROJECT_ROOT,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_HORIZON_SESSIONS,
    TARGET_RETURN_COLUMN,
    TARGET_VALUATION_DATE_COLUMN,
)


PROTOCOL_PATH = PROJECT_ROOT / "prospective_protocol.json"
ALLOWED_STATUSES = {
    "draft_pending_external_timestamp",
    "locked_accruing",
}
MINIMUM_CONFIRMATORY_SIGNALS = 60
HASH_FIELDS = (
    "model_artifact_hash",
    "feature_set_hash",
    "code_release_hash",
    "preprocessor_hash",
    "training_data_snapshot_hash",
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def load_prospective_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_prospective_protocol(protocol)
    return protocol


def validate_prospective_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "protocol_id",
        "status",
        "retrospective_evidence",
        "external_lock",
        "locked_design",
        "stopping_rule",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"Prospective protocol is missing fields: {missing}")
    status = protocol["status"]
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported prospective protocol status: {status}")
    retrospective = protocol["retrospective_evidence"]
    if retrospective.get("may_be_relabelled_confirmatory") is not False:
        raise ValueError("Retrospective evidence must never be relabelled confirmatory.")
    minimum = int(protocol["stopping_rule"]["minimum_consecutive_labelled_signals"])
    if minimum != MINIMUM_CONFIRMATORY_SIGNALS:
        raise ValueError(
            "The prospective stopping rule must remain fixed at "
            f"{MINIMUM_CONFIRMATORY_SIGNALS} consecutive labelled signals."
        )
    if protocol["stopping_rule"].get("confirmatory_analysis_at") != minimum:
        raise ValueError(
            "The prospective protocol must pre-specify one confirmatory analysis "
            "at the fixed minimum."
        )
    if protocol["stopping_rule"].get("no_result_dependent_early_stopping") is not True:
        raise ValueError("Result-dependent early stopping must be disabled.")
    external_lock = protocol["external_lock"]
    if status == "locked_accruing":
        if "DRAFT" in str(protocol["protocol_id"]).upper():
            raise ValueError("A locked protocol ID must not contain DRAFT.")
        required_lock_values = {
            "evidence": external_lock.get("evidence"),
            "timestamp_utc": external_lock.get("timestamp_utc"),
            "verified_by": external_lock.get("verified_by"),
            "first_eligible_signal_date": external_lock.get(
                "first_eligible_signal_date"
            ),
            "first_eligible_signal_close_utc": external_lock.get(
                "first_eligible_signal_close_utc"
            ),
        }
        missing_lock = sorted(
            key for key, value in required_lock_values.items() if not value
        )
        if missing_lock:
            raise ValueError(
                "A locked protocol requires external evidence, timestamp and "
                f"first signal date; missing={missing_lock}"
            )
        lock_timestamp = _parse_utc_timestamp(
            external_lock["timestamp_utc"],
            name="external lock timestamp",
        )
        first_close = _parse_utc_timestamp(
            external_lock["first_eligible_signal_close_utc"],
            name="first eligible signal close",
        )
        first_signal = pd.Timestamp(
            external_lock["first_eligible_signal_date"]
        ).normalize()
        if first_close.tz_convert("America/New_York").date() != first_signal.date():
            raise ValueError(
                "The first eligible signal close must belong to the configured "
                "NYSE signal session."
            )
        if first_close <= lock_timestamp:
            raise ValueError(
                "The first eligible signal close must be strictly later than "
                "the external lock timestamp."
            )
        retrospective = protocol["retrospective_evidence"]
        historical_cutoff = max(
            pd.Timestamp(retrospective["raw_data_cutoff"]).normalize(),
            pd.Timestamp(
                retrospective["last_fully_labelled_signal_date"]
            ).normalize(),
        )
        if first_signal <= historical_cutoff:
            raise ValueError(
                "The first eligible signal date must be later than the packaged "
                "historical evidence cutoff."
            )
        locked_design = protocol["locked_design"]
        invalid_hashes = [
            field
            for field in HASH_FIELDS
            if not SHA256_PATTERN.fullmatch(str(locked_design.get(field) or ""))
        ]
        if invalid_hashes:
            raise ValueError(
                "A locked protocol requires valid SHA-256 design hashes; "
                f"invalid={invalid_hashes}"
            )


def _parse_utc_timestamp(value: object, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid ISO timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit timezone.")
    return timestamp.tz_convert("UTC")


def _normalised_session_dates(values: pd.Series, *, name: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid dates.") from exc
    return parsed.dt.tz_localize(None).dt.normalize()


def validate_prospective_ledger(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Validate a consecutive, fully labelled future TSLA signal ledger."""

    validate_prospective_protocol(protocol)
    if protocol["status"] != "locked_accruing":
        raise RuntimeError(
            "A prospective ledger cannot be accepted before the protocol is "
            "externally locked."
        )
    required = {
        date_column,
        TARGET_EXECUTION_DATE_COLUMN,
        TARGET_VALUATION_DATE_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_RETURN_COLUMN,
        "Predicted Probability",
        "Signal",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prospective ledger is missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    ledger = frame.copy()
    for column in (
        date_column,
        TARGET_EXECUTION_DATE_COLUMN,
        TARGET_VALUATION_DATE_COLUMN,
    ):
        ledger[column] = _normalised_session_dates(
            ledger[column],
            name=column,
        )
    ledger = ledger.sort_values(date_column).reset_index(drop=True)
    if ledger[date_column].duplicated().any():
        raise ValueError("Prospective signal dates must be unique.")

    first_eligible = pd.Timestamp(
        protocol["external_lock"]["first_eligible_signal_date"]
    ).normalize()
    if ledger[date_column].iloc[0] != first_eligible:
        raise ValueError(
            "The prospective ledger must start on the externally locked first "
            "eligible signal date."
        )

    start = ledger[date_column].min() - pd.Timedelta(days=10)
    end = ledger[TARGET_VALUATION_DATE_COLUMN].max() + pd.Timedelta(days=10)
    calendar = XNYSExchangeCalendar(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    sessions = pd.DatetimeIndex(calendar.schedule.index).normalize()
    signal_dates = pd.DatetimeIndex(ledger[date_column])
    expected_signals = sessions[
        (sessions >= signal_dates.min()) & (sessions <= signal_dates.max())
    ]
    if not signal_dates.equals(expected_signals):
        raise ValueError(
            "Prospective signal dates must be consecutive XNYS sessions with "
            "no omitted trading day."
        )
    signal_locations = sessions.get_indexer(signal_dates)
    valuation_offset = TARGET_HORIZON_SESSIONS + 1
    if (
        (signal_locations < 0).any()
        or (signal_locations + valuation_offset >= len(sessions)).any()
    ):
        raise ValueError("Prospective dates fall outside the XNYS calendar window.")
    expected_execution = sessions[signal_locations + 1]
    expected_valuation = sessions[signal_locations + valuation_offset]
    if not pd.DatetimeIndex(
        ledger[TARGET_EXECUTION_DATE_COLUMN]
    ).equals(expected_execution):
        raise ValueError("Execution dates must be the next XNYS sessions.")
    if not pd.DatetimeIndex(
        ledger[TARGET_VALUATION_DATE_COLUMN]
    ).equals(expected_valuation):
        raise ValueError(
            "Valuation dates must be six XNYS sessions after signal "
            "(the Open[t+6] same-sleeve endpoint)."
        )

    probability = pd.to_numeric(
        ledger["Predicted Probability"],
        errors="coerce",
    )
    signal = pd.to_numeric(ledger["Signal"], errors="coerce")
    actual = pd.to_numeric(
        ledger[TARGET_DIRECTION_COLUMN],
        errors="coerce",
    )
    returns = pd.to_numeric(
        ledger[TARGET_RETURN_COLUMN],
        errors="coerce",
    )
    if (
        probability.isna().any()
        or not np.isfinite(probability).all()
        or not probability.between(0.0, 1.0).all()
    ):
        raise ValueError("Prospective probabilities must be finite and within [0, 1].")
    if signal.isna().any() or not signal.isin([0, 1]).all():
        raise ValueError("Prospective signals must be binary.")
    if actual.isna().any() or not actual.isin([0, 1]).all():
        raise ValueError("Prospective realised directions must be binary.")
    if returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError("Prospective realised returns must be finite.")
    threshold = float(protocol["locked_design"]["probability_threshold"])
    if not np.array_equal(
        signal.astype(int).to_numpy(),
        (probability.to_numpy() >= threshold).astype(int),
    ):
        raise ValueError("Prospective signals do not match the locked threshold.")
    if not np.array_equal(
        actual.astype(int).to_numpy(),
        (returns.to_numpy() > 0.0).astype(int),
    ):
        raise ValueError("Prospective directions do not match realised returns.")
    return ledger


def prospective_status_table(
    protocol: dict[str, Any],
    *,
    ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    validate_prospective_protocol(protocol)
    minimum = int(protocol["stopping_rule"]["minimum_consecutive_labelled_signals"])
    status = protocol["status"]
    externally_locked = status == "locked_accruing"
    validated = (
        validate_prospective_ledger(ledger, protocol)
        if ledger is not None
        else pd.DataFrame()
    )
    accrued_labelled_signals = len(validated)
    final_metrics_allowed = externally_locked and accrued_labelled_signals >= minimum
    display_status = "complete" if final_metrics_allowed else status
    return pd.DataFrame(
        [
            {
                "Protocol ID": protocol["protocol_id"],
                "Status": display_status,
                "External Timestamp Evidence": (
                    protocol["external_lock"].get("evidence") or "Not supplied"
                ),
                "First Eligible Signal": (
                    protocol["external_lock"].get("first_eligible_signal_date")
                    or "Not assigned"
                ),
                "Accrued Labelled Signals": int(accrued_labelled_signals),
                "Minimum Required": minimum,
                "Final Metrics Allowed": final_metrics_allowed,
                "Historical Evidence Classification": protocol[
                    "retrospective_evidence"
                ]["classification"],
            }
        ]
    )


def select_prospective_rows(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Select only post-lock rows, refusing to use a draft protocol."""

    validate_prospective_protocol(protocol)
    if protocol["status"] == "draft_pending_external_timestamp":
        raise RuntimeError(
            "Prospective rows cannot be selected before the protocol is externally locked."
        )
    if date_column not in frame.columns:
        raise KeyError(f"Missing prospective date column: {date_column}")
    first_signal = pd.Timestamp(
        protocol["external_lock"]["first_eligible_signal_date"]
    )
    dates = pd.to_datetime(frame[date_column], errors="raise")
    selected = frame.loc[dates.ge(first_signal)].copy()
    selected[date_column] = dates.loc[selected.index]
    return selected.sort_values(date_column).reset_index(drop=True)


def assert_final_metrics_allowed(
    protocol: dict[str, Any],
    *,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Return the prespecified first 60 rows or block confirmatory metrics."""

    validated = validate_prospective_ledger(ledger, protocol)
    status = prospective_status_table(
        protocol,
        ledger=validated,
    ).iloc[0]
    if not bool(status["Final Metrics Allowed"]):
        raise RuntimeError(
            "Prospective final metrics are blocked until the protocol is "
            "externally locked and the prespecified observation count is reached."
        )
    minimum = int(protocol["stopping_rule"]["minimum_consecutive_labelled_signals"])
    return validated.iloc[:minimum].copy()
