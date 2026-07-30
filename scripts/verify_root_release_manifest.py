"""Verify the integrated root release manifest."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v8_evidence import verify_release_manifest


def main() -> int:
    result = verify_release_manifest()
    print(
        f"Root release verification: {result['status']} · "
        f"{result['checked']} checked · {result['failed']} failed"
    )
    if not result["passed"]:
        print(result["message"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
