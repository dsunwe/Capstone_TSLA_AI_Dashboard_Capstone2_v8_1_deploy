# Data Sources and Preprocessing

## Bundled assets

The Dashboard includes daily TSLA, AAPL, and NVDA OHLCV snapshots under
`data/raw/`. The source page for each bundled file is shown on Page 1 and
recorded in `src/config.py`.

The common RQ4 research window contains 2,513 complete daily observations per
asset. Formal RQ packages retain their own reviewed data snapshots so that
prediction and decision ledgers remain independently checkable.

## Required CSV schema

Uploaded files must contain:

```text
Date, Open, High, Low, Close (or Adj Close), Volume
```

The preprocessing pipeline:

1. normalises column names and the price field;
2. parses and sorts dates;
3. removes duplicate dates;
4. coerces OHLCV fields to numeric values;
5. removes unusable rows;
6. checks positive prices, non-negative volume, and OHLC consistency; and
7. returns a visible raw-to-clean audit.

## Context data

Market and macroeconomic context snapshots are stored under `data/context/`.
Each series has an explicit availability rule. Backward joins use only values
available on or before the trading session being constructed.

## Six downstream features

| Feature | Formula |
|---|---|
| `Daily_Return` | `Close[t] / Close[t−1] − 1` |
| `Momentum_5` | `Close[t] / Close[t−5] − 1` |
| `EMA8_13_State` | `1[EMA8[t] > EMA13[t]]` |
| `Volatility_20` | rolling 20-session SD of `Daily_Return` |
| `Drawdown` | `Close[t] / running_max(Close)[t] − 1` |
| `Volume_Ratio` | `Volume[t] / rolling_mean(Volume, 10)[t]` |

Missing training values are handled inside training folds. No held-out label is
used to alter the six-feature specification.

The literature rationale and parameter boundaries are documented in
`docs/literature_and_design_rationale.md` and displayed directly on Page 3.

## Target and leakage control

```text
y[t] = 1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]
```

Information is cut off after `Close[t]`. At each temporal boundary, five
origin rows are purged because their future outcome windows would otherwise
overlap the evaluation period.
