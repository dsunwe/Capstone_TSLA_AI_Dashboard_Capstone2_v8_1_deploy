# RQ2 Reviewed Feature-Family Experiment

This directory preserves the reviewed RQ2 experiment that is integrated into
the Capstone II v8.1 release. Its pre-specified six-feature Technical design is
the upstream information set used by the current RQ3 model comparison and RQ4
downstream evaluation.

## Final RQ2

> Under an identical regularised Logistic Regression framework, how does
> retrospective walk-forward classification performance differ across four
> pre-specified standalone feature families - EMA-aware technical,
> cross-market, macroeconomic, and broad sentiment/risk proxies - when
> predicting Tesla's five-session forward-versus-trailing mean-close direction?

The target is:

`1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`

It is a classification target, not a trading win rate or an executable
five-session endpoint return.

## Main technical family

| Feature | Market state and inclusion reason | 2020+ literature support | Project-specific boundary |
|---|---|---|---|
| `Daily_Return` | Scale-free immediate price response that anchors the longer trend and risk states | Gu et al. (2020); Kumbure et al. (2022); Pabuccu and Barbu (2024) | One-day return is not assumed to predict the five-session class |
| `Momentum_5` | Short-horizon persistence aligned with approximately one trading week | Gu et al. (2020); Peng et al. (2021); Mostafavi and Hooman (2025) | Momentum is literature-motivated; five sessions is fixed by this study |
| `EMA8_13_State` | Compact short-versus-long exponential trend regime | Lusindah and Sumirat (2021); Yang et al. (2022) | Moving-average information is supported; the exact 8/13 pair is not claimed optimal |
| `Volatility_20` | Recent uncertainty and risk regime, complementary to direction | Gu et al. (2020); Yang et al. (2022); Mostafavi and Hooman (2025) | Twenty sessions approximate a trading month and are fixed by this study |
| `Drawdown` | Path-dependent distance below the previous peak | van Hemert et al. (2020); Choi (2021) | RQ2 evaluates rather than assumes its predictive contribution |
| `Volume_Ratio` | Relative activity that complements price-only inputs | Peng et al. (2021); Htun et al. (2023); Pabuccu and Barbu (2024) | The ten-session volume normaliser is a fixed project choice |

The literature motivates the six complementary constructs; it does not prove
that the exact 5-, 8/13-, 20-, or 10-session parameters are universally
optimal. The empirical reason for retaining this Technical family comes from
the controlled RQ2 comparison below. Complete APA references and DOI links are
provided in
[`../../docs/literature_and_design_rationale.md`](../../docs/literature_and_design_rationale.md).

`EMA8_13_State` is a persistent trend state rather than a one-day crossover
event. The same definition can later be used as the transparent EMA8/13 rule
comparator in RQ4.

The following bounded alternatives are retained:

- Core-5 without a moving-average state.
- Core-5 plus `SMA10_20_State`.
- The previous six-feature design using `SMA50_Ratio`.

## Common RQ2 model

All four feature families use the same pipeline:

- training-fold median imputation;
- training-fold standardisation;
- L2 Logistic Regression with `C=0.5`;
- balanced class weights;
- `liblinear`, `max_iter=2000`, and seed 42;
- fixed 0.50 classification threshold.

Development uses five expanding chronological folds, 250 test rows per fold,
and a five-row purge. The later 26 November 2024 to 15 July 2026 period is a
retrospective temporal audit, not a prospectively untouched holdout.

## Reviewed results

| Feature family | Development mean Accuracy | Fold SD | Audit Accuracy | Audit Balanced Accuracy |
|---|---:|---:|---:|---:|
| Technical (EMA-aware) | **69.60%** | **2.62 pp** | **63.88%** | **63.77%** |
| Cross-market | 64.96% | 4.64 pp | 57.99% | 58.34% |
| Broad sentiment/risk proxies | 55.44% | 5.42 pp | 58.97% | 58.22% |
| Macroeconomic | 49.84% | 7.70 pp | 50.12% | 50.83% |

The primary Technical family exceeded weekly-trend persistence by 7.62
percentage points in the audit. Its 20-session paired moving-block bootstrap
95% interval was 0.74 to 13.51 points.

After Holm correction across the three family contrasts:

- Technical versus Macro was clearly positive.
- Technical versus Cross-market was borderline and not conclusive.
- Technical versus Broad sentiment was not statistically conclusive.

The EMA state itself added only 0.24 points over Core-5 in development OOF
(95% interval -0.48 to 0.96 points). EMA is therefore retained for conceptual
continuity, slightly lower fold variability, and RQ4 alignment - not because
it produced a large or statistically secure Accuracy gain.

## Phase-offset robustness

`Eligible_Row_ID modulo 5` defines five immutable phases. Every phase:

- keeps every fifth eligible session;
- uses its own development rows;
- refits its own Logistic Regression;
- uses five development folds with 50 test rows and a one-phase-row purge;
- predicts 81 or 82 audit rows.

EMA-aware Technical Accuracy ranged from 60.49% to 74.07% and pooled to 63.88%.
It was the highest family in four of five phases. The SMA10/20 sensitivity
pooled to 64.86% in this stricter phase-refit check, so EMA must not be
described as uniformly superior.

## Macro release correction

The monthly CPI and unemployment changes are now computed at source frequency
before the release-aware as-of merge. This preserves genuinely unchanged
unemployment releases instead of incorrectly carrying forward the previous
non-zero change. The correction changed Macro-only development Accuracy from
50.88% to 49.84% and audit Accuracy from 50.61% to 50.12%; it did not change
the overall RQ2 conclusion.

## Run

From the package root:

```bash
python run_rq2_experiment.py
python validate_rq2.py
python build_notebook.py
```

`requirements.txt` pins the current reviewed RQ2 reconstruction stack. The
original frozen RQ2 run did not record its software versions, so this supports
repeatable recomputation of the supplied code but is not claimed as a bitwise
environment match to the original artefacts.

The experiment writes reviewed evidence to `outputs/`, figures to `figures/`,
and the diagnostic notebook to `notebooks/`.

All 22 saved validation checks pass. The notebook has 16 cells. Its six code
cells were executed sequentially in a shared Python namespace during build
validation. Kernel-generated outputs are not stored because Jupyter,
`nbformat`, and `nbclient` were unavailable in the build environment.

## Main handoff files

- `protocol.json`: frozen retrospective RQ2 specification.
- `run_rq2_experiment.py`: complete reproducible analysis.
- `validate_rq2.py`: calculation and design checks.
- `notebooks/v8_2_rq2_ema_aware_diagnostic.ipynb`: reader-facing notebook.

Firm-news FinBERT S1 is frozen and is not loaded or recomputed in this package.
