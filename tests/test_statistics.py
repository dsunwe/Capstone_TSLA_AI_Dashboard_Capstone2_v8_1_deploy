import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    benjamini_hochberg_qvalues,
    fold_aware_moving_block_indices,
    joint_circular_block_shift,
    paired_moving_block_bootstrap,
    plus_one_empirical_p_value,
    probability_metric,
)


def test_shared_moving_block_indices_are_deterministic_and_stay_within_folds():
    folds = np.repeat([1, 2, 3], 12)
    first = fold_aware_moving_block_indices(
        folds,
        n_bootstrap=40,
        block_length=5,
        random_state=7,
    )
    second = fold_aware_moving_block_indices(
        folds,
        n_bootstrap=40,
        block_length=5,
        random_state=7,
    )
    assert first.shape == (40, len(folds))
    assert np.array_equal(first, second)
    # Every resample retains the exact row count allocated to each original fold.
    for indices in first:
        sampled_folds = folds[indices]
        assert [np.sum(sampled_folds == fold) for fold in [1, 2, 3]] == [12, 12, 12]


def test_supported_probability_metrics_share_higher_is_better_direction():
    actual = np.array([0, 1, 0, 1, 0, 1])
    perfect = np.array([0.01, 0.99, 0.02, 0.98, 0.03, 0.97])

    assert probability_metric(actual, perfect, metric="macro_f1") == pytest.approx(1.0)
    assert probability_metric(actual, perfect, metric="average_precision") == pytest.approx(1.0)
    expected_inverse_brier = 1.0 - np.mean((perfect - actual) ** 2)
    assert probability_metric(
        actual,
        perfect,
        metric="inverse_brier",
    ) == pytest.approx(expected_inverse_brier)


def test_probability_metric_rejects_invalid_probabilities_and_single_class_pr_auc():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probability_metric([0, 1], [-0.1, 1.1], metric="Macro F1")
    with pytest.raises(ValueError, match="both target classes"):
        probability_metric([1, 1], [0.4, 0.6], metric="PR-AUC")


def test_paired_bootstrap_is_deterministic_and_detects_clear_improvement():
    actual = np.tile([0, 1], 40)
    baseline = np.full(len(actual), 0.50)
    comparison = np.where(actual == 1, 0.90, 0.10)
    folds = np.repeat(np.arange(4), 20)

    first = paired_moving_block_bootstrap(
        actual,
        baseline,
        comparison,
        folds=folds,
        metric="Macro F1",
        n_bootstrap=200,
        block_length=5,
        random_state=17,
    )
    second = paired_moving_block_bootstrap(
        actual,
        baseline,
        comparison,
        folds=folds,
        metric="Macro F1",
        n_bootstrap=200,
        block_length=5,
        random_state=17,
    )

    assert first.metric == "Macro F1"
    assert first.fold_count == 4
    assert first.valid_replicates == 200
    assert first.difference > 0
    assert first.ci_lower > 0
    assert first.probability_positive == 1.0
    assert np.array_equal(
        first.bootstrap_differences,
        second.bootstrap_differences,
    )
    assert first.to_record()["Valid Replicates"] == 200


def test_paired_bootstrap_samples_blocks_within_each_fold():
    # Each fold contains one class. Fold-aware resampling must retain exactly
    # half zeros and half ones in every replicate, giving a fixed AP difference.
    actual = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    baseline = np.full(40, 0.50)
    comparison = np.where(actual == 1, 0.90, 0.10)
    folds = np.repeat(["early", "late"], 20)

    result = paired_moving_block_bootstrap(
        actual,
        baseline,
        comparison,
        folds=folds,
        metric="PR-AUC",
        n_bootstrap=100,
        block_length=6,
        random_state=9,
    )

    assert result.baseline_score == pytest.approx(0.5)
    assert result.comparison_score == pytest.approx(1.0)
    assert np.allclose(result.bootstrap_differences, 0.5)


def test_inverse_brier_bootstrap_difference_is_positive_when_brier_is_lower():
    actual = np.tile([0, 1], 25)
    baseline = np.full(50, 0.50)
    comparison = np.where(actual == 1, 0.80, 0.20)

    result = paired_moving_block_bootstrap(
        actual,
        baseline,
        comparison,
        metric="Inverse Brier",
        n_bootstrap=80,
        block_length=5,
        random_state=3,
    )

    assert result.comparison_score > result.baseline_score
    assert result.difference == pytest.approx(
        np.mean((baseline - actual) ** 2)
        - np.mean((comparison - actual) ** 2)
    )


def test_joint_placebo_shift_is_grouped_deterministic_and_joint():
    frame = pd.DataFrame(
        {
            "Asset": np.repeat(["TSLA", "AAPL"], 12),
            "Date": pd.date_range("2025-01-01", periods=24, freq="B"),
            "Target": np.arange(24) % 2,
            "News_Count": np.r_[np.arange(12), np.arange(20, 32)],
        }
    )
    frame["News_Sentiment"] = frame["News_Count"] + 100
    frame["No_News"] = (frame["News_Count"] % 2 == 0).astype(int)
    original = frame.copy(deep=True)

    first = joint_circular_block_shift(
        frame,
        ["News_Count", "News_Sentiment", "No_News"],
        group_columns=["Asset"],
        block_length=3,
        random_state=11,
    )
    second = joint_circular_block_shift(
        frame,
        ["News_Count", "News_Sentiment", "No_News"],
        group_columns=["Asset"],
        block_length=3,
        random_state=11,
    )

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.audit, second.audit)
    assert first.data["Target"].equals(frame["Target"])
    assert (first.audit["Shift Rows"] > 0).all()
    assert (first.audit["Shift Rows"] % 3 == 0).all()
    assert np.allclose(
        first.data["News_Sentiment"] - first.data["News_Count"],
        100,
    )

    for _, audit in first.audit.iterrows():
        asset = audit["Asset"]
        shift_rows = int(audit["Shift Rows"])
        source = frame.loc[frame["Asset"].eq(asset), "News_Count"].to_numpy()
        shifted = first.data.loc[
            first.data["Asset"].eq(asset),
            "News_Count",
        ].to_numpy()
        assert np.array_equal(shifted, np.roll(source, shift_rows))


def test_joint_placebo_shift_rejects_groups_too_short_for_nonzero_block_shift():
    frame = pd.DataFrame({"Asset": ["TSLA"] * 5, "News": np.arange(5)})
    with pytest.raises(ValueError, match="more rows"):
        joint_circular_block_shift(
            frame,
            ["News"],
            group_columns=["Asset"],
            block_length=5,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        joint_circular_block_shift(
            pd.DataFrame({"Asset": ["TSLA"] * 6}),
            ["Asset"],
            group_columns=["Asset"],
            block_length=2,
        )


def test_plus_one_empirical_p_value_includes_ties_and_prevents_zero():
    null = [0.1, 0.2, 0.4, 0.5]
    assert plus_one_empirical_p_value(0.4, null) == pytest.approx(3 / 5)
    assert plus_one_empirical_p_value(0.9, null) == pytest.approx(1 / 5)
    assert plus_one_empirical_p_value(
        -0.4,
        [-0.5, -0.4, 0.1],
        alternative="less",
    ) == pytest.approx(3 / 4)
    assert plus_one_empirical_p_value(
        0.4,
        [-0.5, -0.2, 0.1, 0.4],
        alternative="two-sided",
    ) == pytest.approx(3 / 5)


def test_benjamini_hochberg_qvalues_are_monotone_and_preserve_nan():
    p_values = np.array([0.01, 0.04, np.nan, 0.03, 0.002])
    q_values = benjamini_hochberg_qvalues(p_values)

    assert np.isnan(q_values[2])
    assert q_values[[0, 1, 3, 4]] == pytest.approx([0.02, 0.04, 0.04, 0.008])
    finite = np.isfinite(p_values)
    order = np.argsort(p_values[finite])
    assert np.all(np.diff(q_values[finite][order]) >= 0)


def test_benjamini_hochberg_rejects_out_of_range_p_values():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg_qvalues([0.1, 1.2])
    with pytest.raises(ValueError, match="infinite"):
        benjamini_hochberg_qvalues([0.1, np.inf])
