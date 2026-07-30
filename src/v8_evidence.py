"""Paths, reviewed-result loaders, release checks, and derived headline facts.

This module deliberately has no Streamlit dependency, so the Dashboard can
load reviewed RQ2–RQ4 results without coupling evidence access to the UI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "research"
RQ2_ROOT = RESEARCH_ROOT / "rq2"
RQ3_ROOT = RESEARCH_ROOT / "rq3"
RQ4_ROOT = RESEARCH_ROOT / "rq4"
ROOT_CONTRACT_PATH = PROJECT_ROOT / "research_contract.json"
ROOT_RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest.json"

ASSETS = ("TSLA", "AAPL", "NVDA")
PERIODS = ("Development OOF", "Retrospective audit")
MODELS = (
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
)
BASELINES = (
    "EMA8/EMA13 State",
    "SMA10/SMA20 State",
    "Buy-and-Hold",
)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a packaged CSV without mutating it."""

    if not path.exists():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a packaged JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract() -> dict[str, Any]:
    """Load the authoritative root research contract."""

    return read_json(ROOT_CONTRACT_PATH)


def load_rq2_protocol() -> dict[str, Any]:
    return read_json(RQ2_ROOT / "protocol.json")


def load_rq3_protocol() -> dict[str, Any]:
    return read_json(RQ3_ROOT / "protocol.json")


def load_rq4_protocol() -> dict[str, Any]:
    return read_json(RQ4_ROOT / "protocol.json")


def load_rq2_summary() -> pd.DataFrame:
    return read_csv(RQ2_ROOT / "outputs" / "rq2_feature_family_summary.csv")


def load_rq2_technical_designs() -> pd.DataFrame:
    return read_csv(RQ2_ROOT / "outputs" / "rq2_technical_design_summary.csv")


def load_rq2_bootstrap() -> pd.DataFrame:
    return read_csv(
        RQ2_ROOT / "outputs" / "rq2_feature_family_paired_block_bootstrap.csv"
    )


def load_rq3_performance() -> pd.DataFrame:
    return read_csv(RQ3_ROOT / "outputs" / "rq3_performance_summary.csv")


def load_rq3_selection() -> pd.DataFrame:
    return read_csv(RQ3_ROOT / "outputs" / "rq3_one_se_selection.csv")


def load_rq3_bootstrap() -> pd.DataFrame:
    return read_csv(RQ3_ROOT / "outputs" / "rq3_paired_block_bootstrap.csv")


def load_rq4_model_results() -> pd.DataFrame:
    return read_csv(RQ4_ROOT / "outputs" / "rq4_model_results.csv")


def load_rq4_headline_metrics() -> pd.DataFrame:
    return read_csv(RQ4_ROOT / "outputs" / "rq4_headline_metrics.csv")


def load_rq4_equity_curves() -> pd.DataFrame:
    frame = read_csv(RQ4_ROOT / "outputs" / "rq4_equity_curves.csv")
    frame["Open Date"] = pd.to_datetime(frame["Open Date"], errors="raise")
    return frame


def load_cross_asset_classification() -> pd.DataFrame:
    """Load the three-asset, separately fitted RQ3 replication summary."""

    return read_csv(
        RQ4_ROOT / "outputs" / "rq4_cross_asset_classification_summary.csv"
    )


def load_cross_asset_configurations() -> pd.DataFrame:
    return read_csv(
        RQ4_ROOT / "outputs" / "rq4_cross_asset_selected_configurations.csv"
    )


def load_validation_tables() -> dict[str, pd.DataFrame]:
    return {
        "RQ2": read_csv(RQ2_ROOT / "outputs" / "rq2_validation_checks.csv"),
        "RQ3": read_csv(RQ3_ROOT / "outputs" / "rq3_validation_checks.csv"),
        "RQ4": read_csv(RQ4_ROOT / "outputs" / "rq4_validation_checks.csv"),
    }


def validation_summary() -> dict[str, Any]:
    """Return the saved RQ2–RQ4 analytical validation status."""

    tables = load_validation_tables()
    rows: list[dict[str, Any]] = []
    total = 0
    passed = 0
    for name, frame in tables.items():
        status_column = next(
            column for column in frame.columns if column.casefold() == "status"
        )
        stage_passed = int(
            frame[status_column].astype(str).str.casefold().eq("pass").sum()
        )
        stage_total = int(len(frame))
        rows.append(
            {
                "Module": name,
                "Passed": stage_passed,
                "Total": stage_total,
                "Status": "PASS" if stage_passed == stage_total else "FAIL",
            }
        )
        total += stage_total
        passed += stage_passed
    return {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "table": pd.DataFrame(rows),
    }


def dataset_summary() -> pd.DataFrame:
    """Summarise the three common-window data snapshots used by RQ4."""

    rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        path = RQ4_ROOT / "data" / f"{asset}_features_common_window.csv"
        frame = read_csv(path)
        date = pd.to_datetime(frame["Date"], errors="raise")
        required = ["Open", "High", "Low", "Close", "Volume"]
        rows.append(
            {
                "Asset": asset,
                "Rows": len(frame),
                "Start": date.min().date().isoformat(),
                "End": date.max().date().isoformat(),
                "Unique Dates": int(date.nunique()),
                "Missing OHLCV": int(frame[required].isna().sum().sum()),
                "Snapshot": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release_manifest(
    project_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Strictly verify the complete integrated release artifact set.

    The root manifest is built only after all integration work is complete.
    Until then this function returns a non-failing ``pending`` state so the
    native Streamlit pages remain renderable during development.  A completed
    manifest passes only when its recorded file set, artifact count, byte
    sizes, and SHA-256 digests exactly match the files that the release builder
    would package.
    """

    root = (project_root or PROJECT_ROOT).resolve()
    path = (manifest_path or (root / "release_manifest.json")).resolve()
    if not path.is_file():
        return {
            "passed": False,
            "pending": True,
            "status": "pending",
            "checked": 0,
            "failed": 0,
            "message": "Root release manifest has not been built yet.",
            "details": [],
            "release": read_json(root / "research_contract.json").get(
                "release", {}
            ),
        }
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "passed": False,
            "pending": False,
            "status": "failed",
            "checked": 0,
            "failed": 1,
            "message": f"Root release manifest could not be read: {exc}",
            "details": [],
            "release": {},
        }
    details: list[dict[str, Any]] = []
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        return {
            "passed": False,
            "pending": False,
            "status": "failed",
            "checked": 0,
            "failed": 1,
            "message": "Root release manifest contains no artifact records.",
            "details": [],
            "release": manifest.get("release", {}),
        }
    try:
        contract_release = read_json(root / "research_contract.json").get(
            "release", {}
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "passed": False,
            "pending": False,
            "status": "failed",
            "checked": 0,
            "failed": 1,
            "message": f"Research contract could not be read: {exc}",
            "details": [],
            "release": manifest.get("release", {}),
        }

    release = manifest.get("release", {})
    if not isinstance(release, dict):
        release = {}
    for field in ("version", "name", "status"):
        expected = contract_release.get(field)
        observed = release.get(field)
        if observed != expected:
            details.append(
                {
                    "Path": f"release.{field}",
                    "Status": "Metadata mismatch",
                    "Expected SHA-256": str(expected),
                    "Observed SHA-256": str(observed),
                }
            )

    declared_count = release.get("artifact_count")
    if (
        not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count != len(artifacts)
    ):
        details.append(
            {
                "Path": "release.artifact_count",
                "Status": "Count mismatch",
                "Expected SHA-256": str(len(artifacts)),
                "Observed SHA-256": str(declared_count),
            }
        )

    recorded: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            details.append(
                {
                    "Path": "",
                    "Status": "Invalid record",
                    "Expected SHA-256": "",
                    "Observed SHA-256": "",
                }
            )
            continue
        relative = str(artifact.get("path", ""))
        expected_hash = str(artifact.get("sha256", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative in recorded
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            details.append(
                {
                    "Path": relative,
                    "Status": "Unsafe or duplicate path",
                    "Expected SHA-256": expected_hash,
                    "Observed SHA-256": "",
                }
            )
            continue
        recorded[relative] = artifact

    # Import locally to avoid a module-import cycle: v8_release uses the
    # hashing and contract helpers defined above.
    from .v8_release import iter_release_files

    actual = {
        artifact_path.relative_to(root).as_posix(): artifact_path
        for artifact_path in iter_release_files(root)
    }
    missing = sorted(set(recorded) - set(actual))
    extra = sorted(set(actual) - set(recorded))
    for relative in missing:
        artifact = recorded[relative]
        details.append(
            {
                "Path": relative,
                "Status": "Missing",
                "Expected SHA-256": str(artifact.get("sha256", "")),
                "Observed SHA-256": "",
            }
        )
    for relative in extra:
        details.append(
            {
                "Path": relative,
                "Status": "Unrecorded artifact",
                "Expected SHA-256": "",
                "Observed SHA-256": sha256_file(actual[relative]),
            }
        )

    for relative in sorted(set(recorded) & set(actual)):
        artifact = recorded[relative]
        artifact_path = actual[relative]
        expected_hash = str(artifact.get("sha256", ""))
        expected_bytes = artifact.get("bytes")
        observed_bytes = artifact_path.stat().st_size
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes != observed_bytes
        ):
            details.append(
                {
                    "Path": relative,
                    "Status": "Size mismatch",
                    "Expected SHA-256": expected_hash,
                    "Observed SHA-256": (
                        f"expected bytes={expected_bytes}; "
                        f"observed bytes={observed_bytes}"
                    ),
                }
            )
            continue
        observed = sha256_file(artifact_path)
        details.append(
            {
                "Path": relative,
                "Status": "Pass" if observed == expected_hash else "Hash mismatch",
                "Expected SHA-256": expected_hash,
                "Observed SHA-256": observed,
            }
        )
    failed = sum(row["Status"] != "Pass" for row in details)
    return {
        "passed": bool(actual) and failed == 0,
        "pending": False,
        "status": "verified" if failed == 0 else "failed",
        "checked": len(details),
        "failed": failed,
        "message": (
            "The complete release artifact set, sizes, and hashes match."
            if failed == 0
            else "Release verification failed."
        ),
        "details": details,
        "release": release,
    }


def rq3_selection_facts() -> dict[str, Any]:
    selection = load_rq3_selection()
    best = selection.loc[selection["Best Model by Mean"].astype(bool)].iloc[0]
    selected = selection.loc[selection["Selected for RQ4"].astype(bool)].iloc[0]
    return {
        "best_observed_model": best["Model"],
        "best_mean_balanced_accuracy": float(best["Mean_Balanced_Accuracy"]),
        "best_standard_error": float(best["Best Model Standard Error"]),
        "one_se_cutoff": float(best["One-SE Cutoff"]),
        "eligible_count": int(selection["One-SE Eligible"].astype(bool).sum()),
        "selected_model": selected["Model"],
    }


def rq4_summary_facts() -> dict[str, int]:
    """Derive the comparison counts used in the honest RQ4 conclusion."""

    models = load_rq4_model_results()
    headline = load_rq4_headline_metrics()

    def period_counts(period: str) -> dict[str, int]:
        model_period = models.loc[models["Period"].eq(period)].copy()
        headline_period = headline.loc[headline["Period"].eq(period)].copy()
        baseline_lookup = (
            headline_period.loc[headline_period["Strategy"].isin(BASELINES)]
            .set_index(["Asset", "Strategy"])["Net Total Return"]
            .to_dict()
        )
        beats = {baseline: 0 for baseline in BASELINES}
        for _, row in model_period.iterrows():
            model_return = float(row["Net Total Return"])
            for baseline in BASELINES:
                if model_return > float(baseline_lookup[(row["Asset"], baseline)]):
                    beats[baseline] += 1
        return {
            "positive_return": int((model_period["Net Total Return"] > 0).sum()),
            "positive_timing": int(
                (
                    model_period[
                        "Net Timing Value vs Constant-Exposure Passive"
                    ]
                    > 0
                ).sum()
            ),
            "beats_buy_hold": beats["Buy-and-Hold"],
            "beats_ema": beats["EMA8/EMA13 State"],
            "beats_sma": beats["SMA10/SMA20 State"],
            "total": len(model_period),
        }

    development = period_counts("Development OOF")
    audit = period_counts("Retrospective audit")
    return {
        **{f"development_{key}": value for key, value in development.items()},
        **{f"audit_{key}": value for key, value in audit.items()},
    }


def percentage(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def percentage_points(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.{digits}f} pp"


def megabytes(value: int) -> str:
    return f"{value / (1024 * 1024):,.2f} MB"
