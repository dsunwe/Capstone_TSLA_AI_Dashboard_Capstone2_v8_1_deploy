"""Create or refresh the integrated root release manifest."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v8_release import write_root_manifest


def main() -> int:
    path = write_root_manifest(root=ROOT)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
