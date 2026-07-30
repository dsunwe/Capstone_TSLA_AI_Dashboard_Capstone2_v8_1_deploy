from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.v8_charts import annotated_heatmap, grouped_percentage_bars


def test_rq3_development_and_audit_series_use_distinct_colours():
    figure = grouped_percentage_bars(
        categories=["LR", "RF"],
        series={
            "Mean outer-fold Balanced Accuracy": [0.68, 0.67],
            "Audit Balanced Accuracy": [0.65, 0.64],
        },
        title="RQ3 comparison",
        reference=0.50,
    )
    axis = figure.axes[0]
    development_colour = axis.patches[0].get_facecolor()
    audit_colour = axis.patches[2].get_facecolor()
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert development_colour != audit_colour
    assert "Mean outer-fold Balanced Accuracy" in legend_labels
    assert "Audit Balanced Accuracy" in legend_labels
    plt.close(figure)


def test_heatmap_supports_a_shared_explicit_percentage_scale():
    matrix = pd.DataFrame(
        [[0.62, 0.67], [0.65, 0.70]],
        index=["A", "B"],
        columns=["LR", "LSTM"],
    )
    figure = annotated_heatmap(
        matrix,
        "Balanced Accuracy",
        value_range=(0.60, 0.70),
    )

    assert figure.axes[0].images[0].get_clim() == pytest.approx((60.0, 70.0))
    assert len({text.get_color() for text in figure.axes[0].texts}) >= 2
    plt.close(figure)


def test_default_positive_heatmap_still_anchors_returns_at_zero():
    matrix = pd.DataFrame([[0.10, 0.20]], index=["A"], columns=["LR", "LSTM"])
    figure = annotated_heatmap(matrix, "Positive returns")

    assert figure.axes[0].images[0].get_clim() == pytest.approx((0.0, 20.0))
    plt.close(figure)
