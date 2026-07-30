from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts.finalize_release_outputs import finalize_release_outputs
from src.release_manifest import (
    ReleaseManifestError,
    verify_release_manifest,
    write_release_manifest,
)
from src.research_contract import PROJECT_ROOT


FIXED_RUN_ID = "test-release-run"
FIXED_GENERATED_UTC = "2026-07-26T12:00:00+00:00"


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "outputs" / "tables").mkdir(parents=True)
    (project / "outputs" / "figures").mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "research_contract.json", project)
    shutil.copy2(PROJECT_ROOT / "requirements-lock.txt", project)
    (project / "outputs" / ".gitkeep").write_text("", encoding="utf-8")
    (project / "outputs" / "figures" / ".gitkeep").write_text("", encoding="utf-8")
    return project


def write_example_artifacts(project: Path) -> None:
    pd.DataFrame(
        [{"Metric": "Macro F1", "Value": 0.5}, {"Metric": "MDD", "Value": -0.2}]
    ).to_csv(project / "outputs" / "tables" / "metrics.csv", index=False)
    (project / "outputs" / "figures" / "figure.png").write_bytes(b"example-png")


def test_manifest_records_release_roots_artifacts_and_csv_rows(tmp_path):
    project = make_project(tmp_path)
    write_example_artifacts(project)

    manifest_path, manifest = write_release_manifest(
        project,
        run_id=FIXED_RUN_ID,
        generated_utc=FIXED_GENERATED_UTC,
    )
    result = verify_release_manifest(project)

    assert manifest_path == project / "outputs" / "release_manifest.json"
    assert manifest["release"]["version"] == "8.1"
    assert manifest["run"] == {
        "id": FIXED_RUN_ID,
        "generated_utc": FIXED_GENERATED_UTC,
    }
    assert manifest["artifact_count"] == 2
    records = {record["path"]: record for record in manifest["artifacts"]}
    assert records["outputs/tables/metrics.csv"]["rows"] == 2
    assert "rows" not in records["outputs/figures/figure.png"]
    assert "outputs/release_manifest.json" not in records
    assert all(not path.endswith(".gitkeep") for path in records)
    assert len(manifest["integrity_roots"]["research_contract"]["sha256"]) == 64
    assert len(manifest["integrity_roots"]["requirements_lock"]["sha256"]) == 64
    assert result["status"] == "verified"
    assert result["artifact_count"] == 2
    assert result["release_name"] == "Examiner-Ready Literature and Lean Dashboard"
    assert result["release_status"] == "retrospective research package"
    assert result["generated_utc"] == FIXED_GENERATED_UTC


def test_verifier_rejects_tampering_missing_and_extra_artifacts(tmp_path):
    project = make_project(tmp_path)
    write_example_artifacts(project)
    write_release_manifest(
        project,
        run_id=FIXED_RUN_ID,
        generated_utc=FIXED_GENERATED_UTC,
    )

    figure = project / "outputs" / "figures" / "figure.png"
    original = figure.read_bytes()
    figure.write_bytes(b"tampered")
    with pytest.raises(ReleaseManifestError, match="mismatch"):
        verify_release_manifest(project)

    figure.write_bytes(original)
    figure.unlink()
    with pytest.raises(ReleaseManifestError, match="missing="):
        verify_release_manifest(project)

    figure.write_bytes(original)
    extra = project / "outputs" / "tables" / "unexpected.csv"
    extra.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ReleaseManifestError, match="extra="):
        verify_release_manifest(project)


def test_verifier_rejects_manifest_with_omitted_record(tmp_path):
    project = make_project(tmp_path)
    write_example_artifacts(project)
    manifest_path, manifest = write_release_manifest(
        project,
        run_id=FIXED_RUN_ID,
        generated_utc=FIXED_GENERATED_UTC,
    )
    manifest["artifacts"] = manifest["artifacts"][:-1]
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReleaseManifestError, match="Artifact set mismatch"):
        verify_release_manifest(project)


def test_finalize_combines_validation_before_manifesting(tmp_path):
    project = make_project(tmp_path)
    columns = [
        "Test ID",
        "Area",
        "Check",
        "Expected Result",
        "Actual Result",
        "Status",
        "Evidence",
    ]
    retained = pd.DataFrame(
        [["D01", "Data", "schema", "Pass", "Pass", "Pass", "ok"]],
        columns=columns,
    )
    sentiment = pd.DataFrame(
        [["N01", "News", "timing", "Pass", "Pass", "Pass", "ok"]],
        columns=columns,
    )
    tables = project / "outputs" / "tables"
    retained.to_csv(tables / "validation_report.csv", index=False)
    sentiment.to_csv(tables / "v7_sentiment_validation_report.csv", index=False)

    result = finalize_release_outputs(
        project,
        run_id=FIXED_RUN_ID,
        generated_utc=FIXED_GENERATED_UTC,
    )
    combined = pd.read_csv(tables / "v7_combined_validation_report.csv")
    records = {
        record["path"]: record for record in result["manifest"]["artifacts"]
    }

    assert combined["Test ID"].tolist() == ["D01", "N01"]
    assert records[
        "outputs/tables/v7_combined_validation_report.csv"
    ]["rows"] == 2
    assert result["verification"]["status"] == "verified"
