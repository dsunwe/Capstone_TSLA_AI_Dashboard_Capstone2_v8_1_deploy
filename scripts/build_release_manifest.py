"""Build outputs/release_manifest.json for the complete frozen output set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.release_manifest import write_release_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--generated-utc")
    args = parser.parse_args()

    path, manifest = write_release_manifest(
        args.project_root,
        run_id=args.run_id,
        generated_utc=args.generated_utc,
    )
    print(
        f"Wrote {path} for release {manifest['release']['version']} "
        f"with {manifest['artifact_count']} artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
