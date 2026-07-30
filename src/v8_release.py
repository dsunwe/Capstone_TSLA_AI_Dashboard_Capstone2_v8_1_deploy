"""Build the root manifest for the integrated Capstone II v8.1 deployment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterator

from .research_contract import load_research_contract
from .v8_evidence import PROJECT_ROOT, sha256_file


IGNORED_PARTS = {
    ".git",
    ".venv",
    ".research-venvs",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "qa",
}
IGNORED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


def iter_release_files(root: Path = PROJECT_ROOT) -> Iterator[Path]:
    """Yield every packaged file except generated/cache/archive material."""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative == Path("release_manifest.json"):
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.name in IGNORED_NAMES:
            continue
        if path.suffix.casefold() in {".pyc", ".zip"}:
            continue
        yield path


def build_root_manifest(root: Path = PROJECT_ROOT) -> dict:
    """Return a complete SHA-256 manifest for one local release folder."""

    contract = load_research_contract(root / "research_contract.json")
    generated_utc = datetime.now(timezone.utc).isoformat()
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in iter_release_files(root)
    ]
    release = {
        **contract["release"],
        "generated_utc": generated_utc,
        "run_id": "v8.1-deploy-release",
        "artifact_count": len(artifacts),
    }
    return {
        "release": release,
        "artifacts": artifacts,
    }


def write_root_manifest(
    path: Path | None = None,
    *,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Write ``release_manifest.json`` after all other release files."""

    output = path or (root / "release_manifest.json")
    output.write_text(
        json.dumps(
            build_root_manifest(root),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
