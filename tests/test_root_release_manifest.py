from __future__ import annotations

import json
from pathlib import Path
import shutil

from src.research_contract import PROJECT_ROOT
from src.v8_evidence import verify_release_manifest
from src.v8_release import write_root_manifest


def _make_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    root.mkdir()
    shutil.copy2(PROJECT_ROOT / "research_contract.json", root)
    (root / "app.py").write_text("print('release')\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# Evidence\n", encoding="utf-8")
    (root / "outputs").mkdir()
    (root / "outputs" / "release_manifest.json").write_text(
        '{"role": "nested legacy evidence"}\n',
        encoding="utf-8",
    )
    write_root_manifest(root=root)
    return root


def test_root_manifest_strictly_verifies_complete_artifact_set(tmp_path):
    root = _make_release(tmp_path)

    result = verify_release_manifest(project_root=root)

    assert result["passed"]
    assert result["status"] == "verified"
    assert result["checked"] == 4
    assert result["failed"] == 0
    assert any(
        row["Path"] == "outputs/release_manifest.json"
        for row in result["details"]
    )


def test_root_manifest_rejects_tampering_missing_and_extra_files(tmp_path):
    root = _make_release(tmp_path)
    app = root / "app.py"
    original = app.read_text(encoding="utf-8")

    app.write_text("print('altered')\n", encoding="utf-8")
    result = verify_release_manifest(project_root=root)
    assert not result["passed"]
    assert any(row["Status"] == "Hash mismatch" for row in result["details"])

    app.write_text(original, encoding="utf-8")
    app.unlink()
    result = verify_release_manifest(project_root=root)
    assert not result["passed"]
    assert any(row["Status"] == "Missing" for row in result["details"])

    app.write_text(original, encoding="utf-8")
    (root / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    result = verify_release_manifest(project_root=root)
    assert not result["passed"]
    assert any(
        row["Status"] == "Unrecorded artifact" for row in result["details"]
    )


def test_root_manifest_rejects_size_and_declared_count_mismatches(tmp_path):
    root = _make_release(tmp_path)
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["artifacts"][0]["bytes"] += 1
    manifest["release"]["artifact_count"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    result = verify_release_manifest(project_root=root)

    assert not result["passed"]
    statuses = {row["Status"] for row in result["details"]}
    assert "Size mismatch" in statuses
    assert "Count mismatch" in statuses
