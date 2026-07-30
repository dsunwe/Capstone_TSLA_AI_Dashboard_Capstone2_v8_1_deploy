# RQ4 Five-Session Cross-Asset Downstream Evaluation

## Purpose

This experiment evaluates whether the frozen predictions from Lagged Logistic Regression, Random Forest, XGBoost and Compact LSTM can be converted into a consistent five-session downstream decision process. The primary emphasis is the machine-learning comparison across TSLA, AAPL and NVDA; Buy-and-Hold and two moving-average rules are transparent reference points.

The results are descriptive historical backtests. No RQ4 bootstrap, p-value or statistical-superiority claim is used. No static-versus-quarterly-rolling-retraining extension is included.

## Frozen Evaluation Design

- RQ3 target: `1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`.
- Four models use the same six feature streams, 20-session lookback, 0.50 threshold and nested temporal protocol.
- TSLA reuses the frozen v8.3 RQ3 predictions; AAPL and NVDA are asset-specific replications under the same protocol.
- Each signal observed after `Close[t]` is executed at `Open[t+1]`.
- Five independent self-financing sleeves each start with 20% of capital and retain a decision through `Open[t+6]`.
- Continuous `1→1` states roll without an artificial exit and re-entry.
- One-way friction is 10 bps transaction cost plus 5 bps slippage.
- Average exposure is NAV-weighted across the five sleeves.
- Static exposure-matched Buy-and-Hold uses an ex-post initial allocation whose mean value-weighted exposure equals the corresponding model path.
- A separate constant-exposure passive path matches the model's average exposure each interval and provides the main descriptive Timing Value.

The five-session implementation aligns the decision horizon with RQ3, but it is not mathematically identical to the classification target because the target uses the mean of five future closing prices.

## Development OOF Results

### Predictive-model net total return

| Asset | Lagged LR | Random Forest | XGBoost | Compact LSTM |
| --- | --- | --- | --- | --- |
| TSLA | 597.20% | 820.03% | 555.96% | 821.87% |
| AAPL | 83.05% | 135.26% | 131.65% | 113.65% |
| NVDA | 566.26% | 795.76% | 528.21% | 551.55% |

### Transparent baseline net total return

| Asset | EMA8/EMA13 State | SMA10/SMA20 State | Buy-and-Hold |
| --- | --- | --- | --- |
| TSLA | 2,069.87% | 1,645.92% | 1,433.18% |
| AAPL | 144.52% | 123.55% | 260.24% |
| NVDA | 1,117.04% | 796.21% | 2,620.29% |

### Net timing value versus constant-exposure passive reference (pp)

| Asset | Lagged LR | Random Forest | XGBoost | Compact LSTM |
| --- | --- | --- | --- | --- |
| TSLA | 71.66 pp | 290.73 pp | 37.58 pp | 425.07 pp |
| AAPL | -27.67 pp | 11.28 pp | 12.15 pp | -4.16 pp |
| NVDA | -107.15 pp | 16.00 pp | -214.48 pp | -52.30 pp |

Observed model leaders in the Development OOF period:

- TSLA: Compact LSTM recorded the highest observed model return at 821.87%.
- AAPL: Random Forest recorded the highest observed model return at 135.26%.
- NVDA: Random Forest recorded the highest observed model return at 795.76%.

These are historical rankings only. They did not change any RQ3 model, threshold, feature or hyperparameter choice.

## Later-Period Results

The later period is retained as a concise temporal-stability check. It is a retrospective audit, not an untouched external validation.

### Predictive-model net total return

| Asset | Lagged LR | Random Forest | XGBoost | Compact LSTM |
| --- | --- | --- | --- | --- |
| TSLA | -15.39% | -1.13% | -8.90% | -3.91% |
| AAPL | 4.41% | -0.45% | -8.70% | -2.74% |
| NVDA | -4.70% | -7.22% | -10.69% | -11.91% |

### Transparent baseline net total return

| Asset | EMA8/EMA13 State | SMA10/SMA20 State | Buy-and-Hold |
| --- | --- | --- | --- |
| TSLA | 12.87% | 32.42% | 9.54% |
| AAPL | 12.03% | 22.79% | 39.42% |
| NVDA | 6.77% | 8.28% | 51.98% |

### Net timing value versus constant-exposure passive reference (pp)

| Asset | Lagged LR | Random Forest | XGBoost | Compact LSTM |
| --- | --- | --- | --- | --- |
| TSLA | -26.49 pp | -12.48 pp | -20.83 pp | -15.12 pp |
| AAPL | -17.62 pp | -25.30 pp | -31.57 pp | -24.39 pp |
| NVDA | -36.19 pp | -42.03 pp | -43.66 pp | -42.43 pp |

Observed model leaders in the later period:

- TSLA: Random Forest recorded the highest observed model return at -1.13%.
- AAPL: Lagged Logistic Regression recorded the highest observed model return at 4.41%.
- NVDA: Lagged Logistic Regression recorded the highest observed model return at -4.70%.

The raw later-period ledger contains 407 predictions per asset-model. The final 2026-07-15 origin is excluded from every economic strategy because the price snapshot ends on 2026-07-22 and therefore lacks `Open[t+6]`; 406 complete signals remain.

## Validation Status

38/38 independent validation checks passed.

`python validate_rq4.py` recomputes and prints these checks in read-only mode.
After an intentional formal recomputation, use
`python validate_rq4.py --write-artifacts` and rebuild the root release
manifest.

- Raw model prediction rows: 19,884.
- Asset-level signal origins: 4,971.
- Development economic signals: 1,250 per asset.
- Later-period economic signals: 406 per asset.
- Quarterly rolling retraining used: no.
- RQ4 bootstrap used: no.

## Main Files

- `notebooks/v8_4_rq4_five_session_cross_asset.ipynb`: executed reader-facing analysis.
- `outputs/rq4_model_results.csv`: four-model return, risk, exposure and timing-value results.
- `outputs/rq4_headline_metrics.csv`: four models, two moving-average baselines and Full Buy-and-Hold.
- `outputs/rq4_exposure_matched_buy_hold.csv`: 24 model-specific static Buy-and-Hold matches plus constant-exposure passive references.
- `outputs/rq4_signal_ledger.csv`: signal-to-execution and terminal-date mapping.
- `outputs/rq4_equity_curves.csv`: complete gross/net NAV paths.
- `outputs/rq4_sleeve_trade_ledger.csv`: phase-level state changes and cost factors.
- `outputs/rq4_validation_checks.csv`: independent validation evidence.
- `protocol.json`: frozen scope, rules and claim boundaries.

## Interpretation Boundary

The evaluated assets are three large US technology-related equities and the inputs are price-return series rather than a complete total-return investment universe. Results therefore show how the AI pipeline behaved under one frozen downstream mapping; they do not establish future or live profitability.
