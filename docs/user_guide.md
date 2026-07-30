# User Guide

## Recommended presentation route

1. **Data and Study Protocol** — state that the completed Dashboard answers
   RQ1; show the target, timing rules, data source, and date coverage.
2. **RQ2 Feature Families** — explain the four-family comparison, then use the
   visible six-row table to justify each selected feature and its literature.
3. **RQ3 Model Comparison** — show the common `20 × 6` design and the LR, RF,
   XGBoost, and Compact LSTM comparison.
4. **Cross-Asset Replication** — show that AAPL and NVDA were processed under
   the same protocol and fitted independently.
5. **RQ4 Downstream Evaluation** — show the five-session decision layer,
   filters, model/baseline table, and equity paths.
6. **Testing and Limitations** — show 84/84 analytical checks and the
   literature traceability expander if requested.

## Section summary

| Section | User action |
|---:|---|
| 1 | Select a bundled CSV or upload compatible OHLCV data; inspect protocol and audit |
| 2 | Inspect price, returns, volatility, and selected-input correlation |
| 3 | Switch Development/audit evidence and inspect the six-feature rationale |
| 4 | Inspect four-model and one-SE results |
| 5 | Switch replication period; compare TSLA/AAPL/NVDA |
| 6 | Filter period, asset, models, baselines, timing, and equity paths |
| 7 | Optionally choose a model and click **Run Sandbox Model** |
| 8 | Inspect checks, limitations, design mapping, and references |

## Formal results versus exploratory analysis

Pages 3–6 show reviewed saved results. Navigating them is deterministic and
does not train a model. This is the intended presentation path.

Page 7 is optional. It uses the current bundled or uploaded CSV, the six RQ2
features, and the selected exploratory controls. Training begins only after
the user clicks **Run Sandbox Model**. Page 7 results never replace formal
RQ2–RQ4 evidence.

## Compatible upload

Provide a daily CSV with:

```text
Date, Open, High, Low, Close (or Adj Close), Volume
```

Use enough chronological history for 20-session features, the five-session
label, and a held-out test segment.
