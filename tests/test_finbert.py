import pandas as pd
import pytest

from src.finbert import (
    DEFAULT_FINBERT_MODEL_ID,
    FINBERT_CACHE_COLUMNS,
    OnnxFinBertScorer,
    finbert_cache_key,
    finbert_text_hash,
    normalize_finbert_text,
    score_texts_with_cache,
    validate_finbert_probabilities,
)


class FakeScorer:
    model_id = DEFAULT_FINBERT_MODEL_ID
    model_revision = "fixture-commit"
    max_length = 128
    preprocessing_version = "v1"

    def __init__(self):
        self.calls = []

    def score_texts(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            probabilities = validate_finbert_probabilities([0.6, 0.1, 0.3])
            rows.append(
                {
                    "cache_key": finbert_cache_key(
                        text,
                        model_id=self.model_id,
                        model_revision=self.model_revision,
                        max_length=self.max_length,
                        preprocessing_version=self.preprocessing_version,
                    ),
                    "text_hash": finbert_text_hash(text),
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "max_length": self.max_length,
                    "preprocessing_version": self.preprocessing_version,
                    **probabilities,
                    "sentiment_score": 0.5,
                }
            )
        return pd.DataFrame(rows, columns=FINBERT_CACHE_COLUMNS)


def test_text_normalization_and_hash_are_deterministic():
    first = normalize_finbert_text("  Tesla   rises ", " Strong\t deliveries ")
    second = normalize_finbert_text("Tesla rises", "Strong deliveries")

    assert first == "Tesla rises\nStrong deliveries"
    assert finbert_text_hash(first) == finbert_text_hash(second)
    assert finbert_text_hash(first) != finbert_text_hash("Tesla falls")


def test_cache_key_changes_with_model_revision_and_inference_contract():
    base = dict(model_id=DEFAULT_FINBERT_MODEL_ID, model_revision="commit-a")
    key = finbert_cache_key("Headline", **base, max_length=128)

    assert key != finbert_cache_key(
        "Headline",
        model_id=DEFAULT_FINBERT_MODEL_ID,
        model_revision="commit-b",
        max_length=128,
    )
    assert key != finbert_cache_key("Headline", **base, max_length=256)


def test_probability_validation_rejects_invalid_rows():
    valid = validate_finbert_probabilities(
        {"positive": 0.2, "negative": 0.3, "neutral": 0.5}
    )
    assert valid["neutral_probability"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="sum to one"):
        validate_finbert_probabilities([0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="between zero and one"):
        validate_finbert_probabilities([1.1, -0.1, 0.0])


def test_cache_scores_each_unique_miss_once_and_reuses_existing_rows():
    scorer = FakeScorer()
    first_results, cache = score_texts_with_cache(
        ["Headline A", "Headline A", "Headline B"],
        scorer=scorer,
    )
    assert len(scorer.calls) == 1
    assert scorer.calls[0] == ["Headline A", "Headline B"]
    assert len(first_results) == 3
    assert len(cache) == 2

    second_results, second_cache = score_texts_with_cache(
        ["Headline B", "Headline A"],
        scorer=scorer,
        cache=cache,
    )
    assert len(scorer.calls) == 1
    assert len(second_results) == 2
    assert len(second_cache) == 2


def test_onnx_scorer_requires_pinned_revision_without_importing_optional_runtime():
    with pytest.raises(ValueError, match="model_revision"):
        OnnxFinBertScorer(
            "model.onnx",
            "tokenizer.json",
            model_revision="",
        )

    scorer = OnnxFinBertScorer(
        "model.onnx",
        "tokenizer.json",
        model_revision="fixture-commit",
    )
    assert scorer.model_revision == "fixture-commit"
