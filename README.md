# Capstone II v8.1 Deploy · Examiner-Ready ML Dashboard

## LSTM correction under review

This branch corrects the formal RQ3 LSTM refit-epoch bookkeeping. The bundled
RQ3/RQ4 results and all numerical claims below remain the original archived
results; they have not been regenerated with the correction. This branch is
not a new validated research release. See
[the correction and rerun guide](docs/paper_epoch_correction.md) before using
the LSTM comparisons in a manuscript. The original release manifests are
intentionally retained, so source/release integrity checks must flag this
work in progress until a complete corrected evidence set is produced.

This GitHub-ready package contains a working Streamlit Dashboard for the
complete RQ1–RQ4 study. It connects data preparation, feature-family
evaluation, LR/RF/XGBoost/LSTM comparison, TSLA–AAPL–NVDA replication, and
downstream five-session evaluation in one eight-section interface.

RQ1 is satisfied by the completed Dashboard itself. The formal research pages
display reviewed outputs produced by the project experiment pipelines; they do
not retrain models during a presentation. Optional user-data training is
isolated on **Analyze Your Data** and starts only after an explicit button
click.

> Educational research only. This is not financial advice, a live execution
> system, or evidence of future profitability.

## Research chain

| RQ | Dashboard evidence |
|---|---|
| RQ1 | Working Streamlit Dashboard with eight connected sections, CSV input, reviewed results, and automated checks |
| RQ2 | Four standalone TSLA feature families compared under the same regularised Logistic Regression |
| RQ3 | Lagged Logistic Regression, Random Forest, XGBoost, and Compact LSTM compared with the same `20 × 6` information |
| RQ4 | The four saved model streams evaluated across TSLA, AAPL, and NVDA through one common five-session decision layer |

The shared classification target is:

```text
1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]
```

Information is cut off after `Close[t]`, with a five-origin purge at temporal
boundaries.

RQ3 uses six feature streams:

```text
Daily_Return, Momentum_5, EMA8_13_State,
Volatility_20, Drawdown, Volume_Ratio
```

Tabular models receive the ordered 120-value flattening of 20 sessions × 6
features. Compact LSTM receives the same information as a `20 × 6` tensor.

RQ4 maps `probability >= 0.50` to long and otherwise to cash. A signal formed
after `Close[t]` is implemented at `Open[t+1]`. Five self-financing 20%
sleeves allow one decision per session, with the same sleeve reconsidered at
`Open[t+6]`. The scenario includes 10 bps transaction cost and 5 bps adverse
slippage per one-way position change.

## Why the six RQ2 features were selected

The six inputs represent complementary market states:

| Feature | Role |
|---|---|
| `Daily_Return` | immediate price response |
| `Momentum_5` | short-horizon persistence |
| `EMA8_13_State` | responsive trend regime |
| `Volatility_20` | recent uncertainty |
| `Drawdown` | path-dependent downside state |
| `Volume_Ratio` | relative trading activity |

Published research motivates these broad return, momentum, trend, volatility,
drawdown, and volume concepts. The exact 5-day, 8/13-day, 20-day, and 10-day
parameters are pre-specified study choices rather than claimed universal
optima. The empirical reason for retaining the Technical family comes from the
controlled, time-ordered RQ2 comparison.

The Dashboard shows this reasoning directly on Page 3. The complete
decision-to-literature map and APA-style references are in
[`docs/literature_and_design_rationale.md`](docs/literature_and_design_rationale.md).

## Dashboard sections

| Section | Content |
|---:|---|
| 1 | Data and Study Protocol; explicit RQ1 completion |
| 2 | Exploratory Data Analysis |
| 3 | RQ2 Feature Families and six-feature literature rationale |
| 4 | RQ3 Model Comparison, including Compact LSTM |
| 5 | Cross-Asset Replication across TSLA, AAPL, and NVDA |
| 6 | RQ4 Downstream Evaluation |
| 7 | Analyze Your Data, optional exploratory training |
| 8 | Testing, limitations, literature traceability, and full references |

## Data and reviewed evidence

- Bundled daily OHLCV snapshots: `data/raw/`
- Causal market and macro context snapshots: `data/context/`
- RQ2 reviewed package: `research/rq2/`
- RQ3 reviewed package: `research/rq3/`
- RQ4 reviewed package: `research/rq4/`
- Canonical RQ/method contract: `research_contract.json`

TSLA, AAPL, and NVDA are processed under the same protocol and fitted
independently. The cross-asset pages therefore evaluate pipeline portability,
not zero-shot transfer of a TSLA-trained model.

## Streamlit Community Cloud deployment

The root `requirements.txt` is the lightweight cloud deployment profile. It
loads all reviewed RQ2–RQ4 results and supports live LR, RF, and XGBoost
Sandbox training without installing TensorFlow. The reviewed Compact LSTM
results remain available on Pages 4–6; only optional live LSTM training on
Page 7 is disabled in the cloud profile. Public CSV uploads are capped at
10 MB and 10,000 rows, and interactive model caches are bounded.
Live hyperparameter search is disabled in the cloud profile to keep the
examiner instance responsive.

1. Create an empty **private** GitHub repository.
2. Upload the **contents of this folder** so that `app.py`,
   `requirements.txt`, and `.streamlit/config.toml` are at repository root.
3. In Streamlit Community Cloud, choose **Create app**, select the repository
   and branch, and set the entry point to `app.py`.
4. Open **Advanced settings** and explicitly select **Python 3.12**.
5. Keep the app private and invite the examiner by email.
6. Deploy, then verify Pages 1, 3, 4, 5, 6, and 8. Page 7 should offer LR, RF,
   and XGBoost while explaining that live LSTM training is disabled.

Do not add `runtime.txt`; Community Cloud currently selects Python in its
deployment interface. Do not commit `.streamlit/secrets.toml`. This project
does not require secrets for its bundled examiner demonstration.

The bundled Nasdaq snapshot manifest states that the files are for private
academic evaluation and that provider terms must be reviewed before public
deployment. Keep both repository and app private unless public redistribution
and display rights have been confirmed. A private Community Cloud viewer uses
Google sign-in or a single-use email link and does not need GitHub access.

The detailed GitHub, Cloud, CI, rollback, and presentation checklist is in
[`docs/deployment.md`](docs/deployment.md).

## Local full setup

Python 3.11 or 3.12 is recommended.

### Windows

```bat
setup_windows.bat
run_dashboard.bat
```

### macOS / Linux

```bash
chmod +x setup_macos_linux.sh run_dashboard.sh
./setup_macos_linux.sh
./run_dashboard.sh
```

The browser opens at `http://localhost:8501`. To use another port:

```bash
.venv/bin/python -m streamlit run app.py --server.port 8502
```

On Windows:

```bat
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
```

The setup scripts install `requirements-local.txt`, which includes TensorFlow
and pytest. This restores live LSTM Sandbox training and the full local test
suite. The lightweight root `requirements.txt` is reserved for Community
Cloud.

## Presentation behaviour

Pages 3–6 read the reviewed RQ2–RQ4 CSV, JSON, prediction, and decision
ledgers. This makes the presentation fast and deterministic. There is no need
to retrain the formal LR, RF, XGBoost, or LSTM models during the presentation.

Page 7 is explicitly exploratory. It can train an available model against a
bundled or uploaded compatible OHLCV CSV after the user clicks **Run Sandbox
Model**. The cloud profile supports LR, RF, and XGBoost; the local full profile
also supports LSTM. Sandbox output never changes the reviewed formal results.

## Validation

The reviewed modules contain 84 saved analytical checks:

| Module | Passed |
|---|---:|
| RQ2 | 22/22 |
| RQ3 | 24/24 |
| RQ4 | 38/38 |
| **Total** | **84/84** |

These checks verify internal consistency among protocols, features,
predictions, signals, trades, metrics, and packaged files. They are not a
claim of future predictive validity.

Run local checks after setup:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/verify_root_release_manifest.py
.venv/bin/python scripts/validate_integrated_release.py
```

The RQ3 and RQ4 validators are read-only by default, so these checks do not
change reviewed evidence or invalidate the release manifest. The explicit
`--write-artifacts` option is reserved for an intentional formal
recomputation, after which the manifest must be rebuilt.

## Formal recomputation

Formal recomputation is optional and is not part of the presentation flow.
Each research stage has its own pinned requirements:

```bash
python scripts/reproduce_all.py --stage rq2 --install
python scripts/reproduce_all.py --stage rq3 --install
python scripts/reproduce_all.py --stage rq4 --install
```

The standard RQ4 stage rebuilds the decision layer from packaged prediction
ledgers. Refitting the cross-asset models is a separate explicit stage:

```bash
python scripts/reproduce_all.py --stage rq4-cross-asset --install
```

Formal recomputation passes `--write-artifacts` to the stage validators. It can
change reviewed files and therefore requires rebuilding and revalidating the
root release manifest.

## Interpretation boundaries

- Classification performance is not a trading win rate.
- The five-session classification label and the `Open[t+1]` to `Open[t+6]`
  decision return are horizon-aligned but not mathematically identical.
- AAPL and NVDA are separately fitted replications.
- RQ4 results do not change RQ2 feature choice, RQ3 model selection,
  thresholds, or hyperparameters.
- The fixed 15 bps one-way friction is a transparent scenario assumption.
- A later-period robustness audit is retained in the result tables and
  summarised once in the Dashboard limitations.
