# v8.3 Focused RQ3 Common-Information Model Comparison

This reviewed RQ3 package inherits the pre-specified v8.2 RQ2 target, six
EMA-aware Technical features, development/audit boundary, and `gap=5`. It
uses the current five-session common-information protocol throughout.

## Final RQ3

> Using the frozen EMA-aware Technical feature specification identified in
> RQ2, how do regularised Logistic Regression, Random Forest, XGBoost and a
> compact LSTM compare in retrospective walk-forward prediction of Tesla's
> rolling five-session mean-price trend, and does any nonlinear or sequential
> model provide robust incremental predictive value over linear and
> transparent trend baselines?

The target remains:

`1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`

This is a classification target, not a trading win rate.

## Fair common-information design

All four main models receive the same six feature streams:

1. `Daily_Return`
2. `Momentum_5`
3. `EMA8_13_State`
4. `Volatility_20`
5. `Drawdown`
6. `Volume_Ratio`

Each origin uses the 20 consecutive trading sessions from `t-19` through `t`.
LR, RF, and XGBoost flatten the common `20×6` tensor to 120 chronological
inputs. LSTM retains the tensor. The data snapshot begins early enough to
construct the history for 3 October 2016, so no development or audit origin is
lost.

### Literature basis for the 20-session window and one-SE rule

The 20-session history was pre-specified as a compact approximation of four
trading weeks, rather than selected after inspecting RQ3 or audit performance.
In a related but different task—one-step-ahead daily equity-volatility
forecasting—Petrozziello et al. (2022) used walk-forward validation and
selected a 20-day LSTM lookback from a pre-defined grid. This provides a
task-adjacent methodological precedent, but it does not establish that 20
sessions is optimal for TSLA direction classification or for every model
family.

Model choice used an **adapted one-standard-error rule**. Among candidates no
more than one estimated standard error below the highest mean outer-fold
Balanced Accuracy, the least complex model was retained. Chen and Yang (2021)
describe the one-standard-error rule as a parsimony-oriented cross-validation
heuristic, but also document limitations in its standard-error estimate and
predictive behaviour. It is therefore used here as a conservative selection
rule, not as a model-equivalence test or a guarantee of superior
generalisation.

The supplementary RQ2 bridge control uses only the six values at `t`. It
exactly reproduces the frozen RQ2 probabilities but is not part of the
equal-information RQ3 model-family selection.

## Nested temporal evaluation

- Development: 2,046 origins, 3 October 2016 to 18 November 2024.
- Outer OOF: five expanding folds, 250 test origins per fold.
- Inner tuning: three expanding folds, 125 validation origins per fold.
- Every train/evaluation boundary: five purged origin rows.
- Retrospective audit: 407 origins, 26 November 2024 to 15 July 2026.
- Probability threshold: 0.50.
- RF, XGBoost, and LSTM: seed 42/43/44 probability ensembles.
- Model selection: development outer-fold Balanced Accuracy only.

LSTM early stopping occurs only inside inner validation folds. The median
inner best epoch is then used to refit on the full applicable training period.
Neither outer tests nor the retrospective audit are used for early stopping.

## Reviewed results

| Model | Development mean Accuracy | Development mean Balanced Accuracy | Fold BA SD | Audit Accuracy | Audit Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Lagged Logistic Regression | **69.68%** | 68.57% | 4.69 pp | 66.83% | 66.56% |
| Random Forest | 69.20% | 68.57% | 4.43 pp | 62.65% | 62.46% |
| XGBoost | 69.60% | 68.81% | 3.75 pp | 63.14% | 63.18% |
| Compact LSTM | 68.40% | **69.54%** | **2.59 pp** | **67.08%** | **66.84%** |

Compact LSTM had the highest mean fold Balanced Accuracy. Its standard error
was 1.16 percentage points, producing a one-SE cutoff of 68.38%. All four
models were eligible. The pre-specified simplicity order therefore selected:

> **Lagged Logistic Regression**

This is a conservative selection result, not a claim that LR was uniquely
best.

The full-development configuration frozen for the audit and RQ4 is:

- `C=5.0`
- 20-session lookback
- 120 chronological lag inputs
- threshold 0.50

## Incremental-value tests

No complex model showed a statistically conclusive advantage over lagged LR
after the paired 20-session moving-block Bootstrap and Holm correction.

| Contrast | Period | Balanced Accuracy difference | 95% CI | Holm-adjusted p |
|---|---|---:|---:|---:|
| RF − LR | Development OOF | −0.61 pp | −4.09 to 2.73 pp | 1.000 |
| XGBoost − LR | Development OOF | −0.13 pp | −3.13 to 2.67 pp | 1.000 |
| LSTM − LR | Development OOF | −0.34 pp | −3.69 to 2.79 pp | 1.000 |
| RF − LR | Audit | −4.11 pp | −8.09 to −0.07 pp | 0.134 |
| XGBoost − LR | Audit | −3.38 pp | −6.68 to 0.02 pp | 0.134 |
| LSTM − LR | Audit | +0.28 pp | −3.20 to 3.84 pp | 0.875 |

The LSTM development mean-fold result and pooled OOF result differ because
Balanced Accuracy is nonlinear when class proportions vary across folds. The
one-SE rule uses the pre-specified mean of five outer folds; the paired
Bootstrap recomputes pooled resampled Balanced Accuracy. Both results are
saved, and they must not be conflated.

The selected LR exceeded weekly persistence in the retrospective audit by
10.35 Balanced-Accuracy points after Holm correction. Its audit advantage over
the EMA-state baseline was 7.53 points, but that interval crossed zero after
correction. The evidence therefore does not establish a stable advantage over
EMA state in both periods.

## Phase-offset robustness

The already frozen daily OOF and audit predictions are evaluated separately
within five immutable `Eligible_Row_ID mod 5` phases. Development has exactly
250 predictions per phase; audit phases contain 81 or 82. This removes overlap
among the five-session forward windows within each phase without changing the
selected models.

In development, phase-level Balanced-Accuracy winners were split across all
four models. In the audit, LR ranked first in three phases and LSTM in two.
This supports the conclusion that no complex model had a uniformly dominant
ranking.

## Defensible conclusion

The supported RQ3 conclusion is:

> After controlling the feature set and 20-session information window,
> increasing model complexity did not provide statistically conclusive or
> consistently stable incremental predictive value over regularised Logistic
> Regression. LSTM produced the highest observed mean fold Balanced Accuracy
> and a slightly higher audit point estimate, but the paired uncertainty
> intervals did not establish superiority.

Do not write that:

- LSTM significantly outperformed LR;
- LR significantly outperformed all complex models;
- 66–70% classification Accuracy is a trading win rate;
- the 2024–2026 audit is prospectively untouched confirmation.

## RQ4 handoff

`freeze/rq3_frozen_rq4_signal.csv` contains the selected model's 1,250
development OOF and 407 retrospective-audit probabilities and binary signals.
RQ4 must read this frozen file. It must not reselect the model or threshold
after seeing returns.

## Reproduce

From the package root:

```bash
python run_rq3_experiment.py
python build_notebook.py
python validate_rq3.py --write-artifacts
```

The complete experiment took approximately 19 minutes on the reviewed CPU
environment. All 24 saved validation checks pass. The reader-facing notebook
contains 13 cells; all seven code cells were executed top to bottom in a shared
in-process IPython namespace because local Jupyter-kernel TCP startup is
unavailable in the current sandbox.

For a release-integrity check without changing any packaged file, run
`python validate_rq3.py` without the flag.

## Main handoff files

- `protocol.json`: frozen retrospective RQ3 specification.
- `run_rq3_experiment.py`: complete nested model comparison.
- `notebooks/v8_3_rq3_model_comparison.ipynb`: executed evidence notebook.
- `outputs/rq3_performance_summary.csv`: reviewed headline metrics.
- `outputs/rq3_paired_block_bootstrap.csv`: paired uncertainty and Holm tests.
- `outputs/rq3_validation_checks.csv`: 24 independent checks.
- `freeze/rq3_freeze_manifest.json`: hashes and selected-model identity.
- `freeze/rq3_frozen_rq4_signal.csv`: immutable RQ4 input.

## References added for RQ3 design

Chen, Y., & Yang, Y. (2021). The one standard error rule for model selection:
Does it work? *Stats, 4*(4), 868–892.
https://doi.org/10.3390/stats4040051

Petrozziello, A., Troiano, L., Serra, A., Jordanov, I., Storti, G.,
Tagliaferri, R., & La Rocca, M. (2022). Deep learning for volatility
forecasting in asset management. *Soft Computing, 26*(17), 8553–8574.
https://doi.org/10.1007/s00500-022-07161-1
