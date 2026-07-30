from __future__ import annotations

import json
from pathlib import Path

from scripts.reproduce_all import ROOT, STAGES


def _pinned_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        versions[name.casefold()] = version.split(";", maxsplit=1)[0].strip()
    return versions


def test_cross_asset_reproduction_uses_recorded_isolated_environment():
    settings = STAGES["rq4-cross-asset"]
    requirements = settings["requirements"]

    assert settings["environment"] == "rq4-cross-asset"
    assert requirements == (
        ROOT / "research" / "rq4" / "requirements-cross-asset.txt"
    )
    assert requirements.is_file()

    pinned = _pinned_versions(requirements)
    expected_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "xgboost": "xgboost",
        "tensorflow": "tensorflow-cpu",
    }
    for asset in ("AAPL", "NVDA"):
        metadata_path = (
            ROOT
            / "research"
            / "rq4"
            / "outputs"
            / f"{asset}_rq3_run_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        software = metadata["software"]
        assert software["python"] == "3.12.13"
        for metadata_name, requirement_name in expected_names.items():
            assert pinned[requirement_name] == software[metadata_name]


def test_formal_tensorflow_requirements_support_darwin_and_other_platforms():
    requirement_paths = [
        ROOT / "research" / "rq3" / "requirements.txt",
        ROOT / "research" / "rq4" / "requirements.txt",
        ROOT / "research" / "rq4" / "requirements-cross-asset.txt",
    ]
    for path in requirement_paths:
        text = path.read_text(encoding="utf-8")
        assert (
            'tensorflow-cpu==2.21.0; platform_system != "Darwin"' in text
        )
        assert 'tensorflow==2.21.0; platform_system == "Darwin"' in text


def test_rq2_reconstruction_stack_is_pinned_and_honestly_scoped():
    requirements = ROOT / "research" / "rq2" / "requirements.txt"
    pinned = _pinned_versions(requirements)

    assert pinned == {
        "numpy": "2.5.1",
        "pandas": "3.0.5",
        "scikit-learn": "1.8.0",
        "matplotlib": "3.10.8",
    }
    metadata = json.loads(
        (
            ROOT
            / "research"
            / "rq2"
            / "outputs"
            / "rq2_experiment_metadata.json"
        ).read_text(encoding="utf-8")
    )
    environment = metadata["reproduction_environment"]
    assert "not a bitwise environment guarantee" in environment["status"]
