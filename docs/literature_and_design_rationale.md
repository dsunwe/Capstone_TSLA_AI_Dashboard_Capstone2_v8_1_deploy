# Literature and Design Rationale

This document maps each major Dashboard decision to peer-reviewed literature
published from 2020 onward. The same mapping appears in the literature
expanders on Pages 1, 3, 4, 5, 6, and 8.

The governing rule is:

> Literature motivates a construct or method category. The project protocol
> specifies the exact target, window, asset set, split count, threshold, and
> cost scenario; project results evaluate those choices.

No source is presented as proof that EMA8/13, momentum 5, volatility 20,
volume normalisation 10, lookback 20, AAPL/NVDA, or 15 bps is universally
optimal.

## Coverage by research question

| Part | Design supported | Main 2020+ sources |
|---|---|---|
| RQ1 | Construction and technical evaluation of a working AI research artifact | vom Brocke et al. (2020); Abbasi et al. (2024) |
| Data and target | Trend label, chronological evaluation, and boundary purge | Wu et al. (2020); Cerqueira et al. (2020) |
| RQ2 | Technical inputs, feature selection, and redundancy control | Gu et al. (2020); Peng et al. (2021); Kumbure et al. (2022); Htun et al. (2023); Pabuccu and Barbu (2024); Mostafavi and Hooman (2025) |
| RQ3 | Linear, bagged-tree, boosted-tree, and LSTM comparison; one-SE parsimony | Nti et al. (2020); Yun et al. (2021); Chen and Yang (2021); Ghosh et al. (2022); Vuong et al. (2022); Petrozziello et al. (2022); Al-Selwi et al. (2024) |
| Cross-asset | Common-protocol replication with independent asset fitting | Kumbure et al. (2022); Kim et al. (2024); Caparrini et al. (2024) |
| RQ4 | Transparent references, temporal discipline, and trading-cost inclusion | Patton and Weller (2020); Lusindah and Sumirat (2021); Cerqueira et al. (2020); Wu et al. (2020) |
| Validation | Artifact evaluation and chronological analytical integrity | vom Brocke et al. (2020); Cerqueira et al. (2020); Abbasi et al. (2024) |

## RQ1: why the Dashboard answers the question

Design-science research treats a purposeful artifact and its evaluation as a
valid research contribution (vom Brocke et al., 2020). Recent AI design
research likewise emphasises building and critically evaluating working AI
artifacts (Abbasi et al., 2024). Here the artifact is the eight-section local
Streamlit Dashboard and its linked data, RQ2, RQ3, RQ4, cross-asset, Sandbox,
and validation flows.

The literature motivates artifact construction and evaluation. The running
interface, automated checks, and traceable result files demonstrate this
project's completion. RQ1 does not require the formal models to be retrained
during a presentation.

## Data, target, and temporal design

Daily OHLCV records are converted into return, momentum, trend, volatility,
drawdown, and relative-volume states. Recent financial-ML research supports
using technical and return-derived information while controlling redundant
inputs (Gu et al., 2020; Peng et al., 2021; Kumbure et al., 2022; Htun et al.,
2023; Pabuccu & Barbu, 2024; Mostafavi & Hooman, 2025).

Trend-based multi-session labels have precedent in financial time-series
classification (Wu et al., 2020). This project's exact target is:

```text
1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]
```

It is a project-specific operationalisation, not a formula copied from Wu et
al. (2020).

Chronological evaluation is required because random shuffling can expose a
learner to later-period information (Cerqueira et al., 2020). The five-origin
purge follows algebraically from this project's five-session future label: it
prevents a training label's outcome window from crossing an evaluation
boundary. The source supports time-ordered evaluation, not the number five
independently of this target.

## RQ2: feature-by-feature rationale

The six final inputs represent complementary market states rather than six
minor variants of one indicator.

| Feature | Exact definition | Inclusion reason | 2020+ literature support | Project boundary |
|---|---|---|---|---|
| `Daily_Return` | `Close[t] / Close[t−1] − 1` | Scale-free immediate price response that anchors longer trend and risk states | Gu et al. (2020); Kumbure et al. (2022); Pabuccu and Barbu (2024) | One-day return is not assumed to predict the five-session class |
| `Momentum_5` | `Close[t] / Close[t−5] − 1` | Short-horizon persistence aligned with approximately one trading week | Gu et al. (2020); Peng et al. (2021); Mostafavi and Hooman (2025) | Momentum is literature-motivated; five sessions is pre-specified here |
| `EMA8_13_State` | `1[EMA8[t] > EMA13[t]]` | Compact short-versus-long exponential trend regime | Lusindah and Sumirat (2021); Yang et al. (2022) | Moving-average information is supported; the exact 8/13 pair is not claimed optimal |
| `Volatility_20` | rolling 20-session SD of `Daily_Return` | Recent uncertainty and risk regime, complementary to direction | Gu et al. (2020); Yang et al. (2022); Mostafavi and Hooman (2025) | Twenty sessions approximate a trading month and are fixed by this study |
| `Drawdown` | `Close[t] / running_max(Close)[t] − 1` | Path-dependent distance below the previous peak | van Hemert et al. (2020); Choi (2021) | Drawdown is a supported state; RQ2 tests rather than assumes predictive contribution |
| `Volume_Ratio` | `Volume[t] / rolling_mean(Volume, 10)[t]` | Relative activity that complements price-only inputs | Peng et al. (2021); Htun et al. (2023); Pabuccu and Barbu (2024) | Volume information is supported; the ten-session normaliser is project-specific |

### Why these six rather than a larger indicator library

Peng et al. (2021), Htun et al. (2023), Pabuccu and Barbu (2024), and
Mostafavi and Hooman (2025) support feature extraction, selection, and
redundancy control in stock forecasting. The project therefore begins with
literature-motivated candidate roles and uses RQ2 to compare four standalone
information families under the same regularised Logistic Regression.

The selection chain is:

1. literature motivates complementary candidate constructs;
2. exact formulas and windows are pre-specified;
3. one classifier and one chronological protocol are held constant;
4. the observed RQ2 evidence supplies the empirical reason to carry the
   six-feature Technical family into RQ3 and RQ4.

## RQ3: why these four ML models

| Model | Inclusion reason | 2020+ literature |
|---|---|---|
| Lagged Logistic Regression | Transparent regularised linear classification reference | Gu et al. (2020); Kumbure et al. (2022) |
| Random Forest | Bagged trees capture nonlinear thresholds and interactions | Nti et al. (2020); Ghosh et al. (2022) |
| XGBoost | Boosted trees provide a distinct sequential error-correction learner | Yun et al. (2021); Vuong et al. (2022) |
| Compact LSTM | Gated recurrence preserves ordered `20 × 6` temporal information | Ghosh et al. (2022); Vuong et al. (2022); Petrozziello et al. (2022); Al-Selwi et al. (2024) |

All four models receive the same six channels over the same 20 sessions. LR,
RF, and XGBoost use the ordered 120-value flattening; LSTM receives the
`20 × 6` tensor. Petrozziello et al. (2022) provide direct, task-adjacent
precedent for an LSTM using a 20-day look-back in financial volatility
forecasting. This project applies 20 sessions to a different target and all
four models as a fixed fair-comparison choice, not a published universal
optimum.

The one-standard-error rule is a transparent parsimony device: candidates
within one estimated standard error of the observed leader remain eligible,
then the pre-declared simpler model is retained. Chen and Yang (2021) analyse
this rule and caution that it does not automatically guarantee future
superiority.

## Why TSLA, AAPL, and NVDA are compared

Financial feature and model performance can vary by security and period,
which motivates replication beyond one ticker (Kumbure et al., 2022; Kim et
al., 2024; Caparrini et al., 2024). AAPL and NVDA are liquid,
technology-related comparators with complete common-window histories.

The literature motivates cross-asset replication generally; it does not
prescribe AAPL and NVDA as uniquely correct controls. Each asset is fitted
independently using the same target, inputs, lookback, candidate models,
splits, threshold, and metrics. This tests pipeline portability and asset
sensitivity, not zero-shot transfer of a TSLA model.

## RQ4: decision-layer rationale

Moving-average rules provide transparent technical references (Lusindah &
Sumirat, 2021). They contextualise probability-derived signals but do not
establish EMA8/13 or SMA10/20 as universally optimal.

The decision layer forms a signal after `Close[t]`, executes at `Open[t+1]`,
uses five self-financing 20% sleeves, reconsiders the assigned sleeve at
`Open[t+6]`, and charges 10 bps cost plus 5 bps adverse slippage per one-way
change. The execution separation and five-sleeve construction are
project-specific, horizon-aligned rules. Cerqueira et al. (2020) and Wu et al.
(2020) support chronological discipline and the multi-session target context.
Patton and Weller (2020) support accounting for implementation costs; they do
not establish 15 bps as a universal empirical constant.

## Validation and claim boundary

The 84 analytical checks verify relationships among data, features,
protocols, predictions, signals, sleeve trades, metrics, and packaged files.
This is consistent with technical artifact evaluation and chronological
research discipline (vom Brocke et al., 2020; Cerqueira et al., 2020; Abbasi
et al., 2024).

The checks support internal consistency and reproducibility. They do not prove
future predictive validity or profitability.

## References

Abbasi, A., Parsons, J., Pant, G., Sheng, O. R. L., & Sarker, S. (2024).
Pathways for design research on artificial intelligence. *Information Systems
Research, 35*(2), 441–459.
https://doi.org/10.1287/isre.2024.editorial.v35.n2

Akşehir, Z. D., & Kılıç, E. (2024). Analyzing the critical steps in deep
learning-based stock forecasting: A literature review. *PeerJ Computer
Science, 10*, e2312. https://doi.org/10.7717/peerj-cs.2312

Al-Selwi, S. M., Hassan, M. F., Abdulkadir, S. J., Muneer, A., Sumiea,
E. H. H., Alqushaibi, A., & Ragab, M. G. (2024). RNN-LSTM: From applications
to modeling techniques and beyond—Systematic review. *Journal of King Saud
University - Computer and Information Sciences, 36*, 102068.
https://doi.org/10.1016/j.jksuci.2024.102068

Caparrini, A., Arroyo, J., & Escayola Mansilla, J. (2024). S&P 500 stock
selection using machine learning classifiers: A look into the changing role of
factors. *Research in International Business and Finance, 70*, 102336.
https://doi.org/10.1016/j.ribaf.2024.102336

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series
forecasting models: An empirical study on performance estimation methods.
*Machine Learning, 109*(11), 1997–2028.
https://doi.org/10.1007/s10994-020-05910-7

Chen, Y., & Yang, Y. (2021). The one standard error rule for model selection:
Does it work? *Stats, 4*(4), 868–892.
https://doi.org/10.3390/stats4040051

Choi, J. (2021). Maximum drawdown, recovery, and momentum. *Journal of Risk
and Financial Management, 14*(11), 542.
https://doi.org/10.3390/jrfm14110542

Ghosh, P., Neufeld, A., & Sahoo, J. K. (2022). Forecasting directional
movements of stock prices for intraday trading using LSTM and random forests.
*Finance Research Letters, 46*, 102280.
https://doi.org/10.1016/j.frl.2021.102280

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
learning. *The Review of Financial Studies, 33*(5), 2223–2273.
https://doi.org/10.1093/rfs/hhaa009

Htun, H. H., Biehl, M., & Petkov, N. (2023). Survey of feature selection and
extraction techniques for stock market prediction. *Financial Innovation, 9*,
26. https://doi.org/10.1186/s40854-022-00441-7

Kim, W., Jeon, J., Jang, M., Kim, S., Lee, H., Yoo, S., & Ahn, J. (2024).
Developing a dynamic feature selection system (DFSS) for stock market
prediction. *Applied Sciences, 14*(16), 7314.
https://doi.org/10.3390/app14167314

Kumbure, M. M., Lohrmann, C., Luukka, P., & Porras, J. (2022). Machine
learning techniques and data for stock market forecasting: A literature
review. *Expert Systems with Applications, 197*, 116659.
https://doi.org/10.1016/j.eswa.2022.116659

Lusindah, L., & Sumirat, E. (2021). Implementation of Fibonacci retracements
and exponential moving average trading strategy in Indonesia Stock Exchange.
*European Journal of Business and Management Research, 6*(4), 402–408.
https://doi.org/10.24018/ejbmr.2021.6.4.1033

Mostafavi, S. M., & Hooman, A. R. (2025). Key technical indicators for stock
market prediction. *Machine Learning with Applications, 20*, 100631.
https://doi.org/10.1016/j.mlwa.2025.100631

Nti, I. K., Adekoya, A. F., & Weyori, B. A. (2020). A comprehensive
evaluation of ensemble learning for stock-market prediction. *Journal of Big
Data, 7*, 20. https://doi.org/10.1186/s40537-020-00299-5

Pabuccu, H., & Barbu, A. (2024). Feature selection with annealing for
forecasting financial time series. *Financial Innovation, 10*, 87.
https://doi.org/10.1186/s40854-024-00617-3

Patton, A. J., & Weller, B. M. (2020). What you see is not what you get: The
costs of trading market anomalies. *Journal of Financial Economics, 137*(2),
515–549. https://doi.org/10.1016/j.jfineco.2020.02.012

Petrozziello, A., Troiano, L., Serra, A., Jordanov, I., Storti, G.,
Tagliaferri, R., & La Rocca, M. (2022). Deep learning for volatility
forecasting in asset management. *Soft Computing, 26*, 8553–8574.
https://doi.org/10.1007/s00500-022-07161-1

Peng, Y., Albuquerque, P. H. M., Kimura, H., & Saavedra, C. A. P. B. (2021).
Feature selection and deep neural networks for stock price direction
forecasting using technical analysis indicators. *Machine Learning with
Applications, 5*, 100060. https://doi.org/10.1016/j.mlwa.2021.100060

van Hemert, O., Ganz, M., Harvey, C. R., Rattray, S., Martin, E. S., &
Yawitch, D. (2020). Drawdowns. *The Journal of Portfolio Management, 46*(8),
34–50. https://doi.org/10.3905/jpm.2020.1.170

vom Brocke, J., Hevner, A., & Maedche, A. (2020). Introduction to design
science research. In J. vom Brocke, A. Hevner, & A. Maedche (Eds.), *Design
science research. Cases* (pp. 1–13). Springer.
https://doi.org/10.1007/978-3-030-46781-4_1

Vuong, P. H., Dat, T. T., Mai, T. K., Uyen, P. H., & The Bao, P. (2022).
Stock-price forecasting based on XGBoost and LSTM. *Computer Systems Science
and Engineering, 40*(1), 237–246.
https://doi.org/10.32604/csse.2022.017685

Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. (2020). A labeling method for
financial time series prediction based on trends. *Entropy, 22*(10), 1162.
https://doi.org/10.3390/e22101162

Yang, J., Wang, Y., & Li, X. (2022). Prediction of stock price direction using
the LASSO-LSTM model combines technical indicators and financial sentiment
analysis. *PeerJ Computer Science, 8*, e1148.
https://doi.org/10.7717/peerj-cs.1148

Yun, K. K., Yoon, S. W., & Won, D. (2021). Prediction of stock price
direction using a hybrid GA-XGBoost algorithm with a three-stage feature
engineering process. *Expert Systems with Applications, 186*, 115716.
https://doi.org/10.1016/j.eswa.2021.115716
