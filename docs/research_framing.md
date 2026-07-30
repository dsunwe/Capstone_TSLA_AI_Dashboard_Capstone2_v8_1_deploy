# Canonical Research Contract for Capstone II v8.1

> This file is generated from `research_contract.json` by
> `scripts/sync_research_framing.py`. Edit the contract, not this file.

## Contract Metadata

- **Release:** 8.1
- **Release name:** Examiner-Ready Literature and Lean Dashboard
- **Date:** 2026-07-30
- **Status:** retrospective research package

## Title

**Design and Evaluation of a Reproducible Five-Session Machine-Learning Dashboard for Stock-Trend Classification and Downstream Decision Analysis**

## Research Lineage

- **Code and data base:** Historical v7.5.3 maintenance baseline, fully integrated into the v8.1 release
- **Interface contract:** One Streamlit application provides eight visible sections, bundled or uploaded CSV analysis, reviewed RQ2-RQ4 result pages, examiner-facing literature traceability, and a separate optional Interactive Exploratory Sandbox.
- **Integration boundary:** Reviewed Q3 and Q4 pages read the packaged experiment outputs without training. Only the optional Analyze Your Data page can fit an exploratory model after an explicit button click.

## Aim

To design and technically validate a reproducible local AI research dashboard that links a five-session stock-trend target, pre-specified feature-family evaluation, common-information model comparison, and cross-asset downstream decision analysis.

## Shared Five-Session Target

- **Formula:** `1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`
- **Information cut-off:** after Close[t]
- **Target horizon:** 5 trading sessions
- **Target date:** trading session `t+5`
- **Purged origin rows:** 5
- **Interpretation boundary:** This is a five-session mean-price trend class, not a trading win rate or a fifth-session endpoint-return label.

The classification target compares the mean of the next five closes with
the trailing five-close mean available at the signal origin. It is not the
same-session trading win rate and is not a fifth-session endpoint-return label.

## Strict RQ-to-Objective Alignment

| Research question | Objective |
|---|---|
| RQ1 | O1 |
| RQ2 | O2 |
| RQ3 | O3 |
| RQ4 | O4 |

## Research Questions, Objectives and Evidence

### RQ1 ↔ O1

**RQ1.** Can the Capstone I proof of concept be delivered as a working local research Dashboard that visibly connects data preparation, feature evaluation, machine-learning comparison, cross-asset replication, and downstream analysis?

**O1.** To implement and display a working local Streamlit Dashboard that connects the RQ1-RQ4 workflow and keeps reviewed results separate from optional exploratory training.

**Scope boundary.** Successful implementation and display of the working eight-section Dashboard, with traceable reviewed results and a separate optional exploratory user-data workflow.

**Reviewed evidence:**

- `app.py`
- `research_contract.json`
- `README.md`
- `docs/literature_and_design_rationale.md`
- `research/rq2/outputs/rq2_validation_checks.csv`
- `research/rq3/outputs/rq3_validation_checks.csv`
- `research/rq4/outputs/rq4_validation_checks.csv`

### RQ2 ↔ O2

**RQ2.** Under an identical regularised Logistic Regression framework, how does retrospective walk-forward classification performance differ across four pre-specified standalone feature families—EMA-aware technical, cross-market, macroeconomic, and broad sentiment/risk proxies—when predicting Tesla's five-session forward-versus-trailing mean-close direction?

**O2.** To compare four standalone TSLA feature families using a common regularised Logistic Regression and five-session target.

**Scope boundary.** TSLA-only feature-family comparison under one common linear model; 1,250 Development OOF predictions and a 407-row retrospective audit.

**Reviewed evidence:**

- `research/rq2/protocol.json`
- `research/rq2/outputs/rq2_feature_family_summary.csv`
- `research/rq2/outputs/rq2_feature_family_oof_predictions.csv`
- `research/rq2/outputs/rq2_feature_family_audit_predictions.csv`
- `research/rq2/outputs/rq2_validation_checks.csv`
- `docs/literature_and_design_rationale.md`

### RQ3 ↔ O3

**RQ3.** Using the pre-specified EMA-aware Technical feature specification identified in RQ2, how do regularised Logistic Regression, Random Forest, XGBoost and a compact LSTM compare in retrospective walk-forward prediction of Tesla's rolling five-session mean-price trend, and does any nonlinear or sequential model provide robust incremental predictive value over linear and transparent trend baselines?

**O3.** To compare four predictive model families under the pre-specified EMA-aware six-feature, 20-session information design and select conservatively using Development data only.

**Scope boundary.** TSLA common-information 20 by 6 model-family comparison, nested temporal evaluation, Development-only one-SE selection, and retrospective temporal audit.

**Reviewed evidence:**

- `research/rq3/protocol.json`
- `research/rq3/outputs/rq3_performance_summary.csv`
- `research/rq3/outputs/rq3_oof_predictions.csv`
- `research/rq3/outputs/rq3_audit_predictions.csv`
- `research/rq3/outputs/rq3_one_se_selection.csv`
- `research/rq3/outputs/rq3_validation_checks.csv`

### RQ4 ↔ O4

**RQ4.** To what extent do the predictions generated by Lagged Logistic Regression, Random Forest, XGBoost and Compact LSTM translate into five-session downstream decision performance relative to Buy-and-Hold and transparent moving-average baselines, and how consistent are the observed outcomes across TSLA, AAPL and NVDA?

**O4.** To operationalise four reviewed model prediction streams through one five-session decision layer and describe their cross-asset return, drawdown, exposure, and temporal-stability patterns.

**Scope boundary.** Descriptive five-session application-layer evaluation across three separately fitted assets; no RQ4 bootstrap, p-value, significance claim, return-driven model reselection, or rolling-retraining comparison.

**Reviewed evidence:**

- `research/rq4/protocol.json`
- `research/rq4/outputs/rq4_model_results.csv`
- `research/rq4/outputs/rq4_equity_curves.csv`
- `research/rq4/outputs/rq4_signal_ledger.csv`
- `research/rq4/outputs/rq4_sleeve_trade_ledger.csv`
- `research/rq4/outputs/rq4_validation_checks.csv`

## RQ3 Common-Information Design

- **Lookback:** 20 trading sessions
- **Sequence shape:** `20 × 6`
- **Feature channels:** Daily_Return, Momentum_5, EMA8_13_State, Volatility_20, Drawdown, Volume_Ratio
- **Common-information rule:** All four RQ3 models receive the same six feature streams over t-19 through t; tabular models receive the ordered 120-value flattening and Compact LSTM receives the 20 by 6 tensor.

The tabular estimators receive the ordered 120-value flattening. Compact
LSTM retains the same 20 × 6 tensor. RQ3 model selection uses Development
evidence only; the later audit and RQ4 returns do not select the model.

## RQ4 Five-Sleeve Decision Layer

- **Assets:** TSLA, AAPL, NVDA
- **Models:** Lagged Logistic Regression, Random Forest, XGBoost, Compact LSTM
- **Signal rule:** Probability >= 0.50 maps to long; otherwise cash.
- **Execution:** `Open[t+1]`
- **Same-sleeve reconsideration/valuation:** `Open[t+6]`
- **Self-financing sleeves:** 5
- **Initial weight per sleeve:** 20%
- **One-way friction:** 15 bps
- **Target-to-decision boundary:** The five-session decision horizon is aligned with RQ3, but Open[t+1] to Open[t+6] return is not mathematically identical to the future-mean-close target.

RQ4 is a descriptive application-layer evaluation. It does not use an
RQ4 bootstrap, p-value, statistical-superiority claim, return-driven
model reselection, or quarterly rolling-retraining comparison.

## Evidence-Scope Matrix

| Evidence scope | Purpose | Assets | Models | Evaluation | Claim boundary |
|---|---|---|---|---|---|
| `system_evidence` | RQ1 completion through a working eight-section Dashboard with technical traceability. | TSLA, AAPL, NVDA, User-uploaded CSV | Not applicable | Local Streamlit execution, eight-section interaction, CSV upload, reviewed RQ2-RQ4 pages that do not train models, an optional button-triggered Interactive Exploratory Sandbox, and automated technical checks. | RQ1 is answered by delivery and display of the working Dashboard; formal pages render reviewed saved results rather than retraining during presentation. |
| `quantitative_feature_family_comparison` | RQ2 standalone feature-family comparison. | TSLA | Regularised Logistic Regression | Five retrospective Development walk-forward folds plus a later retrospective audit under the common five-session target. | The comparison does not establish that one feature family is universally superior across models, assets, or future periods. |
| `quantitative_model_comparison` | RQ3 common-information model-family comparison. | TSLA | Lagged Logistic Regression, Random Forest, XGBoost, Compact LSTM | Nested retrospective temporal evaluation with a common 20 by 6 input, five Development outer folds, a fixed 0.50 threshold, Development-only one-SE selection, and a later retrospective audit. | Observed classification ranks and the one-SE selection do not prove unique model superiority or future stability. |
| `descriptive_downstream_evaluation` | RQ4 prediction-to-decision conversion and cross-asset replication. | TSLA, AAPL, NVDA | Lagged Logistic Regression, Random Forest, XGBoost, Compact LSTM | Five self-financing 20% sleeves, next-open execution, five-session holding, fixed 15 bps one-way friction, Buy-and-Hold, EMA8/13 and SMA10/20 references, exposure measures, and Development/later-period reporting. | RQ4 is descriptive: it uses no RQ4 bootstrap, p-value, significance claim, quarterly rolling retraining, or return-driven tuning. |

## Saved Analytical Validation

| Module | Passed |
|---|---:|
| RQ2 | 22/22 |
| RQ3 | 24/24 |
| RQ4 | 38/38 |
| **Total** | **84/84** |

Saved analytical checks support reproducibility and internal consistency; they are not prospective validation.

## Claim Boundaries

- All 2016-2026 evidence is retrospective; the later period is not a prospectively untouched holdout.
- AAPL and NVDA are asset-specific pipeline replications, not zero-shot transfer of a TSLA-trained model.
- RQ4 return results do not alter RQ2 feature choice, RQ3 model selection, thresholds, or hyperparameters.
- RQ4 is descriptive and does not use a bootstrap, p-value, statistical-superiority claim, or rolling-retraining comparison.
- Historical backtests under simplified fixed costs do not establish future or live profitability.
- Interactive Sandbox outputs are exploratory and cannot replace the reviewed experiment results.

## Contribution Claim

The contribution is a reproducible local AI research system that connects
a working eight-section Streamlit interface with the final
five-session target, standalone feature-family comparison, common 20 × 6
model comparison, and three-asset five-sleeve downstream evaluation.
It is not a claim of guaranteed profitability, universal model
superiority, or a deployable live-trading signal.
