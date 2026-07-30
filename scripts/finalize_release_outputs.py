"""Combine validation ledgers, build the release manifest, and verify it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.release_manifest import verify_release_manifest, write_release_manifest


def combine_validation_outputs(project_root: str | Path) -> Path:
    """Write the retained plus sentiment validation ledger in stable order."""

    project = Path(project_root).resolve()
    tables = project / "outputs" / "tables"
    retained_path = tables / "validation_report.csv"
    sentiment_path = tables / "v7_sentiment_validation_report.csv"
    combined_path = tables / "v7_combined_validation_report.csv"

    missing = [
        path.relative_to(project).as_posix()
        for path in (retained_path, sentiment_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot finalise validation outputs; missing={missing}"
        )

    retained = pd.read_csv(retained_path)
    sentiment = pd.read_csv(sentiment_path)
    if list(retained.columns) != list(sentiment.columns):
        raise ValueError(
            "Validation ledgers must have identical columns before combination."
        )
    combined = pd.concat([retained, sentiment], ignore_index=True)
    if "Test ID" in combined.columns and combined["Test ID"].duplicated().any():
        duplicates = sorted(
            combined.loc[combined["Test ID"].duplicated(False), "Test ID"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Duplicate validation Test ID values: {duplicates}")

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    return combined_path


def finalize_release_outputs(
    project_root: str | Path,
    *,
    run_id: str | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Create the combined ledger, manifest all outputs, then verify the release."""

    project = Path(project_root).resolve()
    combined_path = combine_validation_outputs(project)
    manifest_path, manifest = write_release_manifest(
        project,
        run_id=run_id,
        generated_utc=generated_utc,
    )
    verification = verify_release_manifest(project)
    return {
        "combined_validation_path": combined_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--generated-utc")
    args = parser.parse_args()

    result = finalize_release_outputs(
        args.project_root,
        run_id=args.run_id,
        generated_utc=args.generated_utc,
    )
    verification = result["verification"]
    print(
        f"Finalised release {verification['release_version']} run "
        f"{verification['run_id']}: {verification['artifact_count']} artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
