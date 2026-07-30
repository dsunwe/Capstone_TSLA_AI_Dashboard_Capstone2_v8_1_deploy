"""Create the frozen article-level FinBERT probability cache for v7.

This research-only script reads the revision-pinned FNSPID Parquet shards,
filters the three target equities, removes near-duplicate headlines, and
scores the remaining headlines with the pinned ProsusAI/finbert ONNX model.
The distributable cache contains hashes and probabilities, not copyrighted
headline text or URLs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import onnxruntime as ort
import pandas as pd
import pyarrow.dataset as ds
from tokenizers import Tokenizer


ASSETS = ("AAPL", "NVDA", "TSLA")
MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "1b7521e9fc2e2a12e93de7e3e19373b0505d353f"
LABELS = ("positive", "negative", "neutral")


def normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(value: object) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_and_deduplicate(
    parquet_shards: list[Path],
    *,
    start: str,
    end: str,
    duplicate_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = ds.dataset([str(path) for path in parquet_shards], format="parquet")
    table = dataset.to_table(
        columns=["Date", "Article_title", "Stock_symbol", "Url", "Publisher"],
        filter=ds.field("Stock_symbol").isin(list(ASSETS)),
    )
    frame = table.to_pandas()
    frame = frame.rename(
        columns={
            "Date": "source_timestamp_raw",
            "Article_title": "headline",
            "Stock_symbol": "Asset",
            "Url": "source_url",
            "Publisher": "publisher",
        }
    )
    frame["source_timestamp_utc"] = pd.to_datetime(
        frame["source_timestamp_raw"], utc=True, errors="coerce"
    )
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    frame = frame.loc[
        frame["Asset"].isin(ASSETS)
        & frame["source_timestamp_utc"].ge(start_time)
        & frame["source_timestamp_utc"].lt(end_time)
    ].copy()
    frame["headline"] = frame["headline"].map(normalise_text)
    frame["normalised_headline"] = frame["headline"].str.casefold()
    frame["canonical_url"] = frame["source_url"].map(canonical_url)
    frame["source_domain"] = frame["source_url"].map(
        lambda value: urlsplit(str(value or "")).netloc.lower()
        if str(value or "").strip()
        else ""
    )
    frame["valid_headline"] = frame["normalised_headline"].ne("")
    frame["valid_timestamp"] = frame["source_timestamp_utc"].notna()
    frame = frame.loc[frame["valid_headline"] & frame["valid_timestamp"]].copy()
    frame = frame.sort_values(
        ["Asset", "source_timestamp_utc", "normalised_headline", "canonical_url"],
        kind="mergesort",
    ).reset_index(drop=True)

    window = pd.Timedelta(hours=duplicate_hours)
    title_difference = frame.groupby(
        ["Asset", "normalised_headline"], sort=False
    )["source_timestamp_utc"].diff()
    url_difference = frame.loc[frame["canonical_url"].ne("")].groupby(
        ["Asset", "canonical_url"], sort=False
    )["source_timestamp_utc"].diff()
    duplicate_title = title_difference.notna() & title_difference.le(window)
    duplicate_url = pd.Series(False, index=frame.index)
    duplicate_url.loc[url_difference.index] = (
        url_difference.notna() & url_difference.le(window)
    )
    frame["duplicate_title_48h"] = duplicate_title
    frame["duplicate_url_48h"] = duplicate_url
    frame["duplicate_any"] = duplicate_title | duplicate_url

    audit = (
        frame.groupby("Asset", as_index=False)
        .agg(
            **{
                "Raw Articles": ("Asset", "size"),
                "Duplicate URL Rows": ("duplicate_url_48h", "sum"),
                "Duplicate Title Rows": ("duplicate_title_48h", "sum"),
                "Duplicate Rows Removed": ("duplicate_any", "sum"),
            }
        )
        .sort_values("Asset")
    )
    kept = frame.loc[~frame["duplicate_any"]].copy()
    kept["text_hash"] = kept["normalised_headline"].map(sha256_text)
    kept["story_id"] = (
        kept["Asset"]
        + "|"
        + kept["text_hash"]
        + "|"
        + kept["source_timestamp_utc"].astype(str)
    ).map(sha256_text)
    kept["timestamp_precision"] = np.where(
        (
            kept["source_timestamp_utc"].dt.hour.eq(0)
            & kept["source_timestamp_utc"].dt.minute.eq(0)
            & kept["source_timestamp_utc"].dt.second.eq(0)
        ),
        "date_only_placeholder",
        "intraday",
    )
    unique_counts = kept.groupby("Asset").size().rename("Unique Stories")
    audit = audit.merge(unique_counts, on="Asset", how="left")
    return kept.reset_index(drop=True), audit


def score_headlines(
    frame: pd.DataFrame,
    *,
    model_path: Path,
    tokenizer_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    probabilities: list[np.ndarray] = []
    headlines = frame["headline"].tolist()
    for start in range(0, len(headlines), batch_size):
        encodings = tokenizer.encode_batch(headlines[start : start + batch_size])
        feeds = {
            "input_ids": np.asarray(
                [encoding.ids for encoding in encodings], dtype=np.int64
            ),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            ),
        }
        logits = session.run(["logits"], feeds)[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities.append(exp_logits / exp_logits.sum(axis=1, keepdims=True))
        processed = min(start + batch_size, len(headlines))
        print(f"Processed {processed:,}/{len(headlines):,}", flush=True)
    matrix = np.vstack(probabilities)
    scored = frame.copy()
    for index, label in enumerate(LABELS):
        scored[f"prob_{label}"] = matrix[:, index]
    scored["predicted_label"] = np.asarray(LABELS)[matrix.argmax(axis=1)]
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--duplicate-hours", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--runtime-note",
        default="ONNX Runtime CPU; dynamic INT8 quantized from pinned ONNX weights",
    )
    args = parser.parse_args()

    articles, audit = load_and_deduplicate(
        args.parquet,
        start=args.start,
        end=args.end,
        duplicate_hours=args.duplicate_hours,
    )
    scored = score_headlines(
        articles,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
    )
    inference_time = datetime.now(timezone.utc).isoformat()
    scored["finbert_model_id"] = MODEL_ID
    scored["finbert_model_revision"] = MODEL_REVISION
    scored["inference_runtime"] = args.runtime_note
    scored["inference_timestamp_utc"] = inference_time
    distributable_columns = [
        "story_id",
        "text_hash",
        "Asset",
        "source_timestamp_raw",
        "source_timestamp_utc",
        "timestamp_precision",
        "source_domain",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "predicted_label",
        "finbert_model_id",
        "finbert_model_revision",
        "inference_runtime",
        "inference_timestamp_utc",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    output_temp = args.output.with_suffix(args.output.suffix + ".part")
    audit_temp = args.audit_output.with_suffix(args.audit_output.suffix + ".part")
    scored[distributable_columns].to_csv(output_temp, index=False)
    audit.to_csv(audit_temp, index=False)
    output_temp.replace(args.output)
    audit_temp.replace(args.audit_output)
    print(f"Wrote {len(scored):,} scored stories to {args.output}")


if __name__ == "__main__":
    main()
