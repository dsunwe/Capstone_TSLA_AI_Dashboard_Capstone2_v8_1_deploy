from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.reproduce_all import STAGES
from src.research_contract import PROJECT_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_validators_are_read_only_by_default():
    for relative in (
        "research/rq3/validate_rq3.py",
        "research/rq4/validate_rq4.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "def main(*, write_artifacts: bool = False)" in source
        assert '"--write-artifacts"' in source
        assert "if write_artifacts:" in source


def test_formal_reproduction_explicitly_writes_validation_artifacts():
    assert ["validate_rq3.py", "--write-artifacts"] in STAGES["rq3"]["commands"]
    assert ["validate_rq4.py", "--write-artifacts"] in STAGES["rq4"]["commands"]
    assert [
        "validate_rq4.py",
        "--write-artifacts",
    ] in STAGES["rq4-cross-asset"]["commands"]


def test_rq4_internal_result_manifest_matches_every_declared_file():
    root = PROJECT_ROOT / "research" / "rq4"
    manifest = json.loads(
        (root / "freeze" / "rq4_result_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    files = manifest["files"]

    assert manifest["file_count"] == len(files)
    assert files
    for entry in files:
        path = root / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]
