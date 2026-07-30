from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .research_contract import load_research_contract


MANIFEST_SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUTS_DIRECTORY = "outputs"
DEFAULT_MANIFEST_NAME = "release_manifest.json"
EXCLUDED_FILENAMES = frozenset({".gitkeep"})


class ReleaseManifestError(ValueError):
    """Raised when a release manifest cannot be built or verified safely."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_data_rows(path: Path) -> int:
    """Count CSV records excluding the header row."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ReleaseManifestError(
            f"Path is outside the project root: {path}"
        ) from exc


def _normalise_paths(
    project_root: str | Path,
    outputs_root: str | Path | None,
    manifest_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    project = Path(project_root).resolve()
    outputs = (
        project / DEFAULT_OUTPUTS_DIRECTORY
        if outputs_root is None
        else Path(outputs_root)
    )
    if not outputs.is_absolute():
        outputs = project / outputs
    outputs = outputs.resolve()
    try:
        outputs.relative_to(project)
    except ValueError as exc:
        raise ReleaseManifestError("outputs_root must be inside project_root.") from exc

    manifest = (
        outputs / DEFAULT_MANIFEST_NAME
        if manifest_path is None
        else Path(manifest_path)
    )
    if not manifest.is_absolute():
        manifest = project / manifest
    manifest = manifest.resolve()
    try:
        manifest.relative_to(outputs)
    except ValueError as exc:
        raise ReleaseManifestError("manifest_path must be inside outputs_root.") from exc
    return project, outputs, manifest


def _iter_artifact_paths(outputs_root: Path, manifest_path: Path) -> list[Path]:
    if not outputs_root.exists() or not outputs_root.is_dir():
        raise ReleaseManifestError(f"Outputs directory not found: {outputs_root}")

    artifacts: list[Path] = []
    for path in sorted(outputs_root.rglob("*")):
        if path.name in EXCLUDED_FILENAMES or path.resolve() == manifest_path:
            continue
        if path.is_symlink():
            raise ReleaseManifestError(f"Symlinks are not allowed in outputs: {path}")
        if path.is_file():
            artifacts.append(path)
    return artifacts


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative_posix(path, project_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".csv":
        record["rows"] = csv_data_rows(path)
    return record


def build_release_manifest(
    project_root: str | Path,
    *,
    outputs_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    run_id: str | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build an in-memory manifest for every non-excluded output artifact."""

    project, outputs, manifest = _normalise_paths(
        project_root, outputs_root, manifest_path
    )
    contract_path = project / "research_contract.json"
    requirements_path = project / "requirements-lock.txt"
    contract = load_research_contract(contract_path)
    if not requirements_path.is_file():
        raise ReleaseManifestError(
            f"Requirements lock file not found: {requirements_path}"
        )

    resolved_run_id = run_id or uuid4().hex
    if not isinstance(resolved_run_id, str) or not resolved_run_id.strip():
        raise ReleaseManifestError("run_id must be a non-empty string.")
    resolved_generated_utc = generated_utc or datetime.now(timezone.utc).isoformat()
    try:
        parsed_timestamp = datetime.fromisoformat(
            resolved_generated_utc.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ReleaseManifestError(
            "generated_utc must be an ISO-8601 timestamp."
        ) from exc
    if parsed_timestamp.tzinfo is None:
        raise ReleaseManifestError("generated_utc must include a timezone.")

    artifact_paths = _iter_artifact_paths(outputs, manifest)
    artifacts = [_file_record(path, project) for path in artifact_paths]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "release": {
            "version": contract["release"]["version"],
            "name": contract["release"]["name"],
            "status": contract["release"]["status"],
        },
        "run": {
            "id": resolved_run_id,
            "generated_utc": resolved_generated_utc,
        },
        "integrity_roots": {
            "research_contract": _file_record(contract_path, project),
            "requirements_lock": _file_record(requirements_path, project),
        },
        "outputs_root": _relative_posix(outputs, project),
        "exclusions": [
            _relative_posix(manifest, project),
            "**/.gitkeep",
        ],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_release_manifest(
    project_root: str | Path,
    *,
    outputs_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    run_id: str | None = None,
    generated_utc: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and atomically write the release manifest."""

    project, outputs, manifest_path_resolved = _normalise_paths(
        project_root, outputs_root, manifest_path
    )
    manifest = build_release_manifest(
        project,
        outputs_root=outputs,
        manifest_path=manifest_path_resolved,
        run_id=run_id,
        generated_utc=generated_utc,
    )
    manifest_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path_resolved.with_suffix(
        manifest_path_resolved.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path_resolved)
    return manifest_path_resolved, manifest


def load_release_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseManifestError(f"Manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReleaseManifestError("Manifest must be a JSON object.")
    return manifest


def _validate_record_path(
    record_path: Any,
    outputs_relative: PurePosixPath,
) -> PurePosixPath:
    if not isinstance(record_path, str) or not record_path.strip():
        raise ReleaseManifestError("Every artifact requires a non-empty path.")
    relative = PurePosixPath(record_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseManifestError(f"Unsafe artifact path: {record_path!r}")
    try:
        relative.relative_to(outputs_relative)
    except ValueError as exc:
        raise ReleaseManifestError(
            f"Artifact is outside outputs_root: {record_path}"
        ) from exc
    if relative.name in EXCLUDED_FILENAMES:
        raise ReleaseManifestError(f"Excluded file recorded as artifact: {record_path}")
    return relative


def _verify_integrity_root(
    project_root: Path,
    label: str,
    record: Any,
    expected_path: str,
) -> None:
    if not isinstance(record, dict) or record.get("path") != expected_path:
        raise ReleaseManifestError(
            f"integrity_roots.{label}.path must be {expected_path!r}."
        )
    path = project_root / expected_path
    if not path.is_file():
        raise ReleaseManifestError(f"Integrity root missing: {expected_path}")
    expected_bytes = path.stat().st_size
    expected_hash = sha256_file(path)
    if record.get("bytes") != expected_bytes:
        raise ReleaseManifestError(f"Size mismatch for {expected_path}.")
    if record.get("sha256") != expected_hash:
        raise ReleaseManifestError(f"SHA-256 mismatch for {expected_path}.")


def verify_release_manifest(
    project_root: str | Path,
    *,
    outputs_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly verify metadata, roots and the complete output artifact set."""

    project, outputs, manifest_path_resolved = _normalise_paths(
        project_root, outputs_root, manifest_path
    )
    manifest = load_release_manifest(manifest_path_resolved)
    contract = load_research_contract(project / "research_contract.json")

    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReleaseManifestError("Unsupported manifest_schema_version.")
    release = manifest.get("release")
    expected_release = {
        key: contract["release"][key]
        for key in ("version", "name", "status")
    }
    if not isinstance(release, dict) or release != expected_release:
        raise ReleaseManifestError(
            "Manifest release metadata does not match research_contract.json."
        )
    run = manifest.get("run")
    if not isinstance(run, dict) or not str(run.get("id", "")).strip():
        raise ReleaseManifestError("Manifest run.id must be non-empty.")
    generated_utc = run.get("generated_utc")
    if not isinstance(generated_utc, str):
        raise ReleaseManifestError("Manifest run.generated_utc is missing.")
    try:
        timestamp = datetime.fromisoformat(generated_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseManifestError(
            "Manifest run.generated_utc is not ISO-8601."
        ) from exc
    if timestamp.tzinfo is None:
        raise ReleaseManifestError("Manifest run.generated_utc lacks a timezone.")

    outputs_relative_string = _relative_posix(outputs, project)
    if manifest.get("outputs_root") != outputs_relative_string:
        raise ReleaseManifestError("Manifest outputs_root does not match verification.")
    outputs_relative = PurePosixPath(outputs_relative_string)

    roots = manifest.get("integrity_roots")
    if not isinstance(roots, dict):
        raise ReleaseManifestError("Manifest integrity_roots is missing.")
    _verify_integrity_root(
        project,
        "research_contract",
        roots.get("research_contract"),
        "research_contract.json",
    )
    _verify_integrity_root(
        project,
        "requirements_lock",
        roots.get("requirements_lock"),
        "requirements-lock.txt",
    )

    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ReleaseManifestError("Manifest artifacts must be a list.")
    if manifest.get("artifact_count") != len(records):
        raise ReleaseManifestError("artifact_count does not match artifacts.")

    recorded: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReleaseManifestError("Every artifact record must be an object.")
        relative = _validate_record_path(record.get("path"), outputs_relative)
        key = relative.as_posix()
        if key == _relative_posix(manifest_path_resolved, project):
            raise ReleaseManifestError("The manifest must exclude itself.")
        if key in recorded:
            raise ReleaseManifestError(f"Duplicate artifact path: {key}")
        recorded[key] = record

    actual_paths = _iter_artifact_paths(outputs, manifest_path_resolved)
    actual = {_relative_posix(path, project): path for path in actual_paths}
    missing = sorted(set(recorded) - set(actual))
    extra = sorted(set(actual) - set(recorded))
    if missing or extra:
        raise ReleaseManifestError(
            f"Artifact set mismatch; missing={missing}; extra={extra}"
        )

    for relative_path, path in actual.items():
        record = recorded[relative_path]
        if record.get("bytes") != path.stat().st_size:
            raise ReleaseManifestError(f"Size mismatch for {relative_path}.")
        if record.get("sha256") != sha256_file(path):
            raise ReleaseManifestError(f"SHA-256 mismatch for {relative_path}.")
        if path.suffix.lower() == ".csv":
            if record.get("rows") != csv_data_rows(path):
                raise ReleaseManifestError(f"CSV row-count mismatch for {relative_path}.")
        elif "rows" in record:
            raise ReleaseManifestError(
                f"Non-CSV artifact must not declare rows: {relative_path}"
            )

    return {
        "release_version": release["version"],
        "release_name": release["name"],
        "release_status": release["status"],
        "run_id": run["id"],
        "generated_utc": generated_utc,
        "artifact_count": len(actual),
        "manifest_path": _relative_posix(manifest_path_resolved, project),
        "status": "verified",
    }
