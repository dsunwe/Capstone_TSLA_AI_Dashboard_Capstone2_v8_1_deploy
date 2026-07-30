from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


DEFAULT_FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_PROBABILITY_COLUMNS = [
    "positive_probability",
    "negative_probability",
    "neutral_probability",
]
FINBERT_CACHE_COLUMNS = [
    "cache_key",
    "text_hash",
    "model_id",
    "model_revision",
    "max_length",
    "preprocessing_version",
    *FINBERT_PROBABILITY_COLUMNS,
    "sentiment_score",
]


def normalize_finbert_text(title: Any, summary: Any = None) -> str:
    """Create the stable title/summary text used for both inference and caching."""

    def clean(value: Any) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return ""
        normalized = unicodedata.normalize("NFKC", str(value))
        return re.sub(r"\s+", " ", normalized).strip()

    title_text = clean(title)
    summary_text = clean(summary)
    if title_text and summary_text:
        return f"{title_text}\n{summary_text}"
    return title_text or summary_text


def finbert_text_hash(text: Any) -> str:
    normalized = normalize_finbert_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def finbert_cache_key(
    text: Any,
    *,
    model_id: str,
    model_revision: str,
    max_length: int = 256,
    preprocessing_version: str = "v1",
) -> str:
    if not str(model_id).strip():
        raise ValueError("model_id must be non-empty.")
    if not str(model_revision).strip():
        raise ValueError("A pinned model_revision is required.")
    if int(max_length) < 1:
        raise ValueError("max_length must be positive.")
    payload = {
        "text_hash": finbert_text_hash(text),
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "max_length": int(max_length),
        "preprocessing_version": str(preprocessing_version),
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_finbert_probabilities(
    values: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """Validate and return positive/negative/neutral probabilities."""
    if isinstance(values, Mapping):
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            label = str(key).strip().lower()
            label = label.removesuffix("_probability")
            normalized[label] = value
        missing = {"positive", "negative", "neutral"}.difference(normalized)
        if missing:
            raise ValueError(f"Missing FinBERT probabilities: {sorted(missing)}")
        array = np.asarray(
            [
                normalized["positive"],
                normalized["negative"],
                normalized["neutral"],
            ],
            dtype=float,
        )
    else:
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) != 3:
            raise ValueError("FinBERT probabilities must contain exactly three values.")

    if not np.isfinite(array).all():
        raise ValueError("FinBERT probabilities must be finite.")
    if bool(((array < 0) | (array > 1)).any()):
        raise ValueError("FinBERT probabilities must be between zero and one.")
    if not np.isclose(float(array.sum()), 1.0, atol=tolerance, rtol=0):
        raise ValueError("FinBERT probabilities must sum to one.")
    return {
        "positive_probability": float(array[0]),
        "negative_probability": float(array[1]),
        "neutral_probability": float(array[2]),
    }


def _cache_record(
    text: str,
    probabilities: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    *,
    model_id: str,
    model_revision: str,
    max_length: int,
    preprocessing_version: str,
) -> dict[str, Any]:
    validated = validate_finbert_probabilities(probabilities)
    return {
        "cache_key": finbert_cache_key(
            text,
            model_id=model_id,
            model_revision=model_revision,
            max_length=max_length,
            preprocessing_version=preprocessing_version,
        ),
        "text_hash": finbert_text_hash(text),
        "model_id": model_id,
        "model_revision": model_revision,
        "max_length": int(max_length),
        "preprocessing_version": preprocessing_version,
        **validated,
        "sentiment_score": (
            validated["positive_probability"]
            - validated["negative_probability"]
        ),
    }


def validate_finbert_cache(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a cache snapshot and return it in canonical column order."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=FINBERT_CACHE_COLUMNS)
    missing = set(FINBERT_CACHE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing FinBERT cache columns: {sorted(missing)}")
    if frame["cache_key"].duplicated().any():
        raise ValueError("FinBERT cache keys must be unique.")
    result = frame.copy()
    for row in result.itertuples(index=False):
        validate_finbert_probabilities(
            {
                "positive": getattr(row, "positive_probability"),
                "negative": getattr(row, "negative_probability"),
                "neutral": getattr(row, "neutral_probability"),
            }
        )
    return result[FINBERT_CACHE_COLUMNS].reset_index(drop=True)


class OnnxFinBertScorer:
    """Optional CPU ONNX scorer.

    ``tokenizers`` and ``onnxruntime`` are imported only on the first scoring
    call. The research environment may therefore build and test the dashboard
    without installing either package. The caller must provide a pinned model
    revision so cached probabilities remain reproducible.
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        *,
        model_revision: str,
        model_id: str = DEFAULT_FINBERT_MODEL_ID,
        max_length: int = 256,
        label_order: Sequence[str] = ("positive", "negative", "neutral"),
        preprocessing_version: str = "v1",
    ) -> None:
        if not str(model_revision).strip():
            raise ValueError("A pinned model_revision is required.")
        if int(max_length) < 1:
            raise ValueError("max_length must be positive.")
        normalized_labels = tuple(str(label).strip().lower() for label in label_order)
        if set(normalized_labels) != {"positive", "negative", "neutral"}:
            raise ValueError(
                "label_order must contain positive, negative, and neutral exactly once."
            )
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.max_length = int(max_length)
        self.label_order = normalized_labels
        self.preprocessing_version = str(preprocessing_version)
        self._tokenizer: Any = None
        self._session: Any = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._session is not None:
            return
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Optional ONNX FinBERT scoring requires the 'tokenizers' package."
            ) from exc
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Optional ONNX FinBERT scoring requires the 'onnxruntime' package."
            ) from exc
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer JSON not found: {self.tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(
            length=self.max_length,
            pad_id=0,
            pad_token="[PAD]",
        )
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def score_texts(self, texts: Iterable[Any]) -> pd.DataFrame:
        normalized = [normalize_finbert_text(text) for text in texts]
        if not normalized:
            return pd.DataFrame(columns=FINBERT_CACHE_COLUMNS)
        self._ensure_loaded()
        encodings = self._tokenizer.encode_batch(normalized)
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [item.attention_mask for item in encodings], dtype=np.int64
        )
        token_type_ids = np.asarray(
            [item.type_ids for item in encodings], dtype=np.int64
        )
        available_inputs = {item.name for item in self._session.get_inputs()}
        candidates = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        feed = {
            name: value for name, value in candidates.items() if name in available_inputs
        }
        missing_required = {"input_ids", "attention_mask"}.difference(feed)
        if missing_required:
            raise RuntimeError(
                f"ONNX model lacks expected inputs: {sorted(missing_required)}"
            )
        logits = np.asarray(self._session.run(None, feed)[0], dtype=float)
        if logits.ndim != 2 or logits.shape != (len(normalized), 3):
            raise RuntimeError(
                f"Expected FinBERT logits with shape ({len(normalized)}, 3), "
                f"received {logits.shape}."
            )
        probabilities = self._softmax(logits)
        rows: list[dict[str, Any]] = []
        for text, values in zip(normalized, probabilities):
            by_label = {
                label: float(values[index])
                for index, label in enumerate(self.label_order)
            }
            rows.append(
                _cache_record(
                    text,
                    by_label,
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    max_length=self.max_length,
                    preprocessing_version=self.preprocessing_version,
                )
            )
        return pd.DataFrame(rows, columns=FINBERT_CACHE_COLUMNS)


def score_texts_with_cache(
    texts: Iterable[Any],
    *,
    scorer: Any,
    cache: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score only cache misses and return input-ordered results plus updated cache."""
    required_attributes = [
        "model_id",
        "model_revision",
        "max_length",
        "preprocessing_version",
    ]
    missing_attributes = [
        name for name in required_attributes if not hasattr(scorer, name)
    ]
    if missing_attributes:
        raise ValueError(
            f"Scorer lacks cache identity attributes: {missing_attributes}"
        )
    normalized = [normalize_finbert_text(text) for text in texts]
    keys = [
        finbert_cache_key(
            text,
            model_id=scorer.model_id,
            model_revision=scorer.model_revision,
            max_length=scorer.max_length,
            preprocessing_version=scorer.preprocessing_version,
        )
        for text in normalized
    ]
    existing = validate_finbert_cache(cache) if cache is not None else pd.DataFrame(
        columns=FINBERT_CACHE_COLUMNS
    )
    existing_keys = set(existing["cache_key"])

    unique_missing: dict[str, str] = {}
    for key, text in zip(keys, normalized):
        if key not in existing_keys:
            unique_missing.setdefault(key, text)
    if unique_missing:
        scored = scorer.score_texts(list(unique_missing.values()))
        scored = validate_finbert_cache(scored)
        expected = set(unique_missing)
        actual = set(scored["cache_key"])
        if expected != actual:
            raise ValueError(
                "Scorer returned cache keys inconsistent with its declared model identity."
            )
        updated = (
            scored.copy()
            if existing.empty
            else pd.concat([existing, scored], ignore_index=True)
        )
    else:
        updated = existing.copy()
    updated = validate_finbert_cache(
        updated.drop_duplicates("cache_key", keep="first")
    )
    indexed = updated.set_index("cache_key", drop=False)
    ordered = indexed.loc[keys].reset_index(drop=True)
    return ordered, updated
