"""Build the final result manifest and integrity-checked experiment archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "freeze" / "rq4_result_manifest.json"
ARCHIVE = ROOT.parent / "v8_4_rq4_five_session_cross_asset_20260729.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if (
            "__pycache__" in relative.parts
            or path.suffix == ".pyc"
            or relative == Path("freeze/rq4_result_manifest.json")
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item.relative_to(ROOT)))


def write_manifest(files: list[Path]) -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    payload = {
        "protocol_id": protocol["protocol_id"],
        "manifest_scope": "Every formal file in the final archive except the manifest itself.",
        "quarterly_rolling_retraining_included": False,
        "file_count": len(files),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_archive(files: list[Path]) -> None:
    with zipfile.ZipFile(
        ARCHIVE,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in [*files, MANIFEST]:
            arcname = Path(ROOT.name) / path.relative_to(ROOT)
            archive.write(path, arcname)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad is not None:
            raise RuntimeError(f"Corrupt archive member: {bad}")
        if len(names) != len(set(names)):
            raise RuntimeError("Archive contains duplicate paths.")
        if any(name.lower().endswith(".zip") for name in names):
            raise RuntimeError("Archive unexpectedly contains a nested ZIP.")


def main() -> None:
    files = formal_files()
    write_manifest(files)
    build_archive(files)
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "archive_sha256": sha256(ARCHIVE),
                "archive_bytes": ARCHIVE.stat().st_size,
                "manifest_files": len(files),
                "zip_test": "passed",
                "duplicate_paths": False,
                "nested_zip": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
