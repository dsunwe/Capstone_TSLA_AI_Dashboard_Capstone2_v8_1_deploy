"""Strictly verify outputs/release_manifest.json and every recorded artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.release_manifest import ReleaseManifestError, verify_release_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    try:
        result = verify_release_manifest(args.project_root)
    except ReleaseManifestError as exc:
        print(f"Release manifest verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Verified release {result['release_version']} run {result['run_id']}: "
        f"{result['artifact_count']} artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
