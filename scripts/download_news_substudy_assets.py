"""Download frozen v7 research-only news and FinBERT assets.

The large source files and model weights are intentionally excluded from the
submission zip. This script records hashes and exact revisions so the cached
article probabilities and daily aggregates can be independently regenerated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


FNSPID_REVISION = "c5cebb37e76d64d27d79ca72ede8f93b84e1a8aa"
FINBERT_REVISION = "1b7521e9fc2e2a12e93de7e3e19373b0505d353f"

ASSETS = {
    "fnspid_train_0.parquet": (
        "https://huggingface.co/datasets/benstaf/"
        "FNSPID-filtered-nasdaq-100/resolve/"
        f"{FNSPID_REVISION}/data/train-00000-of-00002.parquet?download=true"
    ),
    "fnspid_train_1.parquet": (
        "https://huggingface.co/datasets/benstaf/"
        "FNSPID-filtered-nasdaq-100/resolve/"
        f"{FNSPID_REVISION}/data/train-00001-of-00002.parquet?download=true"
    ),
    "prosus_finbert.onnx": (
        "https://huggingface.co/ProsusAI/finbert/resolve/"
        f"{FINBERT_REVISION}/onnx/model.onnx?download=true"
    ),
    "prosus_finbert_tokenizer.json": (
        "https://huggingface.co/ProsusAI/finbert/resolve/"
        f"{FINBERT_REVISION}/onnx/tokenizer.json?download=true"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Capstone-v7-academic-reproduction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_assets"),
        help="Local directory for large research-only source files.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Download only FNSPID parquet shards.",
    )
    args = parser.parse_args()
    selected = {
        name: url
        for name, url in ASSETS.items()
        if not args.skip_model or "finbert" not in name
    }
    manifest_rows: list[dict[str, object]] = []
    for filename, url in selected.items():
        destination = args.output_dir / filename
        print(f"Preparing {filename}...")
        download(url, destination, force=args.force)
        manifest_rows.append(
            {
                "filename": filename,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "source_url_without_query": url.split("?")[0],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    manifest = {
        "fnspid_dataset_id": "benstaf/FNSPID-filtered-nasdaq-100",
        "fnspid_revision": FNSPID_REVISION,
        "fnspid_license": "CC BY-NC 4.0 (upstream FNSPID dataset card)",
        "finbert_model_id": "ProsusAI/finbert",
        "finbert_revision": FINBERT_REVISION,
        "files": manifest_rows,
    }
    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
