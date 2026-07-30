"""Current literature-to-design traceability for the eight-section Dashboard.

Every cited publication is from 2020 or later. The tables distinguish
literature-supported constructs from exact project parameters: a source may
motivate a feature or method category without proving that this study's
window, asset set, threshold, or cost assumption is universally optimal.
"""

from __future__ import annotations

import pandas as pd


REFERENCE_URLS = {
    "vom Brocke et al. (2020)": (
        "https://doi.org/10.1007/978-3-030-46781-4_1"
    ),
    "Gu et al. (2020)": "https://doi.org/10.1093/rfs/hhaa009",
    "Cerqueira et al. (2020)": (
        "https://doi.org/10.1007/s10994-020-05910-7"
    ),
    "Wu et al. (2020)": "https://doi.org/10.3390/e22101162",
    "van Hemert et al. (2020)": (
        "https://doi.org/10.3905/jpm.2020.1.170"
    ),
    "Patton & Weller (2020)": (
        "https://doi.org/10.1016/j.jfineco.2020.02.012"
    ),
    "Nti et al. (2020)": "https://doi.org/10.1186/s40537-020-00299-5",
    "Lusindah & Sumirat (2021)": (
        "https://doi.org/10.24018/ejbmr.2021.6.4.1033"
    ),
    "Peng et al. (2021)": "https://doi.org/10.1016/j.mlwa.2021.100060",
    "Chen & Yang (2021)": "https://doi.org/10.3390/stats4040051",
    "Yun et al. (2021)": "https://doi.org/10.1016/j.eswa.2021.115716",
    "Choi (2021)": "https://doi.org/10.3390/jrfm14110542",
    "Kumbure et al. (2022)": (
        "https://doi.org/10.1016/j.eswa.2022.116659"
    ),
    "Ghosh et al. (2022)": "https://doi.org/10.1016/j.frl.2021.102280",
    "Vuong et al. (2022)": "https://doi.org/10.32604/csse.2022.017685",
    "Yang et al. (2022)": "https://doi.org/10.7717/peerj-cs.1148",
    "Petrozziello et al. (2022)": (
        "https://doi.org/10.1007/s00500-022-07161-1"
    ),
    "Htun et al. (2023)": "https://doi.org/10.1186/s40854-022-00441-7",
    "Abbasi et al. (2024)": (
        "https://doi.org/10.1287/isre.2024.editorial.v35.n2"
    ),
    "Al-Selwi et al. (2024)": (
        "https://doi.org/10.1016/j.jksuci.2024.102068"
    ),
    "Akşehir & Kılıç (2024)": "https://doi.org/10.7717/peerj-cs.2312",
    "Kim et al. (2024)": "https://doi.org/10.3390/app14167314",
    "Caparrini et al. (2024)": (
        "https://doi.org/10.1016/j.ribaf.2024.102336"
    ),
    "Pabuccu & Barbu (2024)": (
        "https://doi.org/10.1186/s40854-024-00617-3"
    ),
    "Mostafavi & Hooman (2025)": (
        "https://doi.org/10.1016/j.mlwa.2025.100631"
    ),
}


def feature_rationale_table() -> pd.DataFrame:
    """Return the examiner-facing rationale for the six RQ2 features."""

    rows = [
        {
            "Feature": "Daily_Return",
            "Definition": "Close[t] / Close[t−1] − 1",
            "Market state captured": "Immediate price response",
            "Why included": (
                "Provides a scale-free one-session change and anchors the longer "
                "trend, risk, and activity states."
            ),
            "Literature support": (
                "Gu et al. (2020); Kumbure et al. (2022); "
                "Pabuccu & Barbu (2024)"
            ),
            "Project-specific boundary": (
                "The sources motivate return-derived inputs; they do not establish "
                "one-day return as a predictor of this five-session class."
            ),
        },
        {
            "Feature": "Momentum_5",
            "Definition": "Close[t] / Close[t−5] − 1",
            "Market state captured": "Short-horizon persistence",
            "Why included": (
                "Adds a one-trading-week direction state aligned with the study's "
                "short decision horizon."
            ),
            "Literature support": (
                "Gu et al. (2020); Peng et al. (2021); "
                "Mostafavi & Hooman (2025)"
            ),
            "Project-specific boundary": (
                "Momentum is literature-motivated; five sessions is a pre-specified "
                "project horizon rather than a published universal optimum."
            ),
        },
        {
            "Feature": "EMA8_13_State",
            "Definition": "1[EMA8[t] > EMA13[t]]",
            "Market state captured": "Responsive trend regime",
            "Why included": (
                "Compresses short-versus-long exponential trend into one state "
                "without retaining two closely related price levels."
            ),
            "Literature support": (
                "Lusindah & Sumirat (2021); Yang et al. (2022)"
            ),
            "Project-specific boundary": (
                "The sources support moving-average trend information. The exact "
                "8/13 pair was fixed by this study and is not claimed globally best."
            ),
        },
        {
            "Feature": "Volatility_20",
            "Definition": "Rolling 20-session SD of Daily_Return",
            "Market state captured": "Recent uncertainty",
            "Why included": (
                "Adds a risk and regime channel not represented by price direction "
                "or momentum alone; 20 sessions approximate one trading month."
            ),
            "Literature support": (
                "Gu et al. (2020); Yang et al. (2022); "
                "Mostafavi & Hooman (2025)"
            ),
            "Project-specific boundary": (
                "Volatility is literature-motivated; the 20-session return window "
                "is this project's fixed implementation."
            ),
        },
        {
            "Feature": "Drawdown",
            "Definition": "Close[t] / running_max(Close)[t] − 1",
            "Market state captured": "Path-dependent downside state",
            "Why included": (
                "Distinguishes prices near a prior peak from prices still below it, "
                "even when their latest daily returns are similar."
            ),
            "Literature support": (
                "van Hemert et al. (2020); Choi (2021)"
            ),
            "Project-specific boundary": (
                "The sources support drawdown as a path-dependent market state; "
                "its predictive contribution is evaluated by RQ2 rather than assumed."
            ),
        },
        {
            "Feature": "Volume_Ratio",
            "Definition": "Volume[t] / rolling_mean(Volume, 10)[t]",
            "Market state captured": "Relative trading activity",
            "Why included": (
                "Normalises absolute volume into an activity state that complements "
                "price-only trend and risk inputs."
            ),
            "Literature support": (
                "Peng et al. (2021); Htun et al. (2023); "
                "Pabuccu & Barbu (2024)"
            ),
            "Project-specific boundary": (
                "The sources motivate volume and technical inputs; the ten-session "
                "normaliser is a fixed study choice, not a published optimum."
            ),
        },
    ]
    return pd.DataFrame(rows)


def method_rationale_table(section: str) -> pd.DataFrame:
    """Return current literature and design support for a Dashboard section."""

    mappings: dict[str, list[dict[str, str]]] = {
        "rq1": [
            {
                "Design decision": (
                    "Build and technically evaluate a working research artifact"
                ),
                "Reason in this project": (
                    "RQ1 is answered by the completed Streamlit Dashboard, its "
                    "traceable result flow, and technical verification."
                ),
                "Literature support": (
                    "vom Brocke et al. (2020); Abbasi et al. (2024)"
                ),
                "Boundary": (
                    "Design-science guidance motivates artifact construction and "
                    "evaluation; the project's tests provide its specific evidence."
                ),
            },
            {
                "Design decision": "Five-session trend label",
                "Reason in this project": (
                    "A bounded multi-session direction class matches the project's "
                    "short-horizon research objective."
                ),
                "Literature support": "Wu et al. (2020)",
                "Boundary": (
                    "The exact forward-mean versus trailing-mean formula is this "
                    "project's pre-specified operationalisation."
                ),
            },
            {
                "Design decision": (
                    "Expanding-window evaluation and five-row purge"
                ),
                "Reason in this project": (
                    "Chronology is preserved and the purge stops a training label's "
                    "future window from crossing an evaluation boundary."
                ),
                "Literature support": "Cerqueira et al. (2020)",
                "Boundary": (
                    "Time-ordered evaluation is literature-supported; purge=5 "
                    "follows from this project's five-session label."
                ),
            },
        ],
        "eda": [
            {
                "Design decision": (
                    "Describe price, return, volatility, correlation, and volume states"
                ),
                "Reason in this project": (
                    "EDA checks scale, missingness, regimes, and redundancy before "
                    "model results are interpreted."
                ),
                "Literature support": (
                    "Kumbure et al. (2022); Yang et al. (2022); "
                    "Mostafavi & Hooman (2025)"
                ),
                "Boundary": (
                    "EDA is descriptive and does not by itself select features or "
                    "establish predictive value."
                ),
            }
        ],
        "rq2": [
            {
                "Design decision": (
                    "Compare four standalone feature families with the same LR"
                ),
                "Reason in this project": (
                    "Holding the classifier and temporal protocol constant isolates "
                    "differences associated with each information family."
                ),
                "Literature support": (
                    "Peng et al. (2021); Kumbure et al. (2022); "
                    "Htun et al. (2023); Akşehir & Kılıç (2024); "
                    "Pabuccu & Barbu (2024); "
                    "Mostafavi & Hooman (2025)"
                ),
                "Boundary": (
                    "The literature motivates feature extraction and redundancy "
                    "control; this project's RQ2 evidence retains the Technical family."
                ),
            }
        ],
        "rq3": [
            {
                "Design decision": "LR, RF, XGBoost, and Compact LSTM",
                "Reason in this project": (
                    "The set spans a transparent linear classifier, bagged trees, "
                    "boosted trees, and a gated sequence model."
                ),
                "Literature support": (
                    "Nti et al. (2020); Yun et al. (2021); "
                    "Kumbure et al. (2022); Ghosh et al. (2022); "
                    "Vuong et al. (2022); Al-Selwi et al. (2024)"
                ),
                "Boundary": (
                    "Recent studies motivate the model families; relative performance "
                    "is established only by this project's common protocol."
                ),
            },
            {
                "Design decision": "Common 20 × 6 information design",
                "Reason in this project": (
                    "All models receive the same ordered history; only representation "
                    "and model inductive bias differ."
                ),
                "Literature support": (
                    "Ghosh et al. (2022); Vuong et al. (2022); "
                    "Petrozziello et al. (2022); Akşehir & Kılıç (2024)"
                ),
                "Boundary": (
                    "Petrozziello et al. provide task-adjacent precedent for a "
                    "20-day LSTM look-back. This project's 20 sessions remain a "
                    "fixed fair-comparison choice, not a universal optimum."
                ),
            },
            {
                "Design decision": "Development-only one-standard-error rule",
                "Reason in this project": (
                    "Select the pre-ranked simpler model when its result is within "
                    "one estimated standard error of the observed leader."
                ),
                "Literature support": "Chen & Yang (2021)",
                "Boundary": (
                    "The rule supports transparent parsimony; it does not guarantee "
                    "the best future model."
                ),
            },
        ],
        "cross_asset": [
            {
                "Design decision": "TSLA, AAPL, and NVDA under one protocol",
                "Reason in this project": (
                    "AAPL and NVDA provide liquid, long-history US technology-related "
                    "comparators for checking whether the pipeline is asset-bound."
                ),
                "Literature support": (
                    "Kumbure et al. (2022); Kim et al. (2024); "
                    "Caparrini et al. (2024)"
                ),
                "Boundary": (
                    "The literature motivates replication and model comparison; it "
                    "does not nominate AAPL and NVDA as universal controls."
                ),
            },
            {
                "Design decision": "Independent fitting by asset",
                "Reason in this project": (
                    "Each asset uses its own history with the same target, features, "
                    "models, splits, threshold, and metrics."
                ),
                "Literature support": (
                    "Kim et al. (2024); Caparrini et al. (2024)"
                ),
                "Boundary": (
                    "This evaluates pipeline portability, not zero-shot transfer "
                    "from a TSLA-trained model."
                ),
            },
        ],
        "rq4": [
            {
                "Design decision": (
                    "Transparent moving-average and Buy-and-Hold references"
                ),
                "Reason in this project": (
                    "Simple, inspectable references contextualise the downstream "
                    "behaviour of probability-derived signals."
                ),
                "Literature support": (
                    "Nti et al. (2020); Lusindah & Sumirat (2021)"
                ),
                "Boundary": (
                    "The sources motivate technical and benchmark comparisons; the "
                    "exact EMA8/13 and SMA10/20 pairs are project choices."
                ),
            },
            {
                "Design decision": (
                    "Next-open execution, five sleeves, and five-session holding"
                ),
                "Reason in this project": (
                    "Signal formation is separated from execution and one new "
                    "horizon-aligned decision is permitted per session."
                ),
                "Literature support": (
                    "Cerqueira et al. (2020); Wu et al. (2020)"
                ),
                "Boundary": (
                    "The exact execution and sleeve construction is this project's "
                    "decision-layer implementation."
                ),
            },
            {
                "Design decision": "10 bps cost plus 5 bps slippage",
                "Reason in this project": (
                    "A transparent one-way friction scenario avoids reporting only "
                    "a frictionless historical result."
                ),
                "Literature support": "Patton & Weller (2020)",
                "Boundary": (
                    "The source motivates implementation-cost accounting; 15 bps is "
                    "a fixed scenario, not a universal empirical constant."
                ),
            },
        ],
        "validation": [
            {
                "Design decision": "Saved-ledger and protocol consistency checks",
                "Reason in this project": (
                    "Validators recompute relationships among data, predictions, "
                    "signals, trades, metrics, and declared methods."
                ),
                "Literature support": (
                    "vom Brocke et al. (2020); Cerqueira et al. (2020)"
                ),
                "Boundary": (
                    "Passing checks supports technical integrity and reproducibility; "
                    "it is not proof of future predictive validity."
                ),
            }
        ],
    }
    try:
        rows = mappings[section]
    except KeyError as exc:
        raise ValueError(f"Unknown literature section: {section}") from exc
    return pd.DataFrame(rows)


def references_table() -> pd.DataFrame:
    """Return the complete APA-style reference list used in the Dashboard."""

    entries = [
        (
            "vom Brocke, J., Hevner, A., & Maedche, A. (2020). Introduction "
            "to design science research. In J. vom Brocke, A. Hevner, & "
            "A. Maedche (Eds.), Design science research. Cases (pp. 1–13). "
            "Springer.",
            "vom Brocke et al. (2020)",
        ),
        (
            "Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via "
            "machine learning. The Review of Financial Studies, 33(5), "
            "2223–2273.",
            "Gu et al. (2020)",
        ),
        (
            "Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time "
            "series forecasting models: An empirical study on performance "
            "estimation methods. Machine Learning, 109(11), 1997–2028.",
            "Cerqueira et al. (2020)",
        ),
        (
            "Wu, D., Wang, X., Su, J., Tang, B., & Wu, S. (2020). A labeling "
            "method for financial time series prediction based on trends. "
            "Entropy, 22(10), 1162.",
            "Wu et al. (2020)",
        ),
        (
            "van Hemert, O., Ganz, M., Harvey, C. R., Rattray, S., Martin, "
            "E. S., & Yawitch, D. (2020). Drawdowns. The Journal of Portfolio "
            "Management, 46(8), 34–50.",
            "van Hemert et al. (2020)",
        ),
        (
            "Patton, A. J., & Weller, B. M. (2020). What you see is not what "
            "you get: The costs of trading market anomalies. Journal of "
            "Financial Economics, 137(2), 515–549.",
            "Patton & Weller (2020)",
        ),
        (
            "Nti, I. K., Adekoya, A. F., & Weyori, B. A. (2020). A "
            "comprehensive evaluation of ensemble learning for stock-market "
            "prediction. Journal of Big Data, 7, 20.",
            "Nti et al. (2020)",
        ),
        (
            "Lusindah, L., & Sumirat, E. (2021). Implementation of Fibonacci "
            "retracements and exponential moving average trading strategy in "
            "Indonesia Stock Exchange. European Journal of Business and "
            "Management Research, 6(4), 402–408.",
            "Lusindah & Sumirat (2021)",
        ),
        (
            "Peng, Y., Albuquerque, P. H. M., Kimura, H., & Saavedra, "
            "C. A. P. B. (2021). Feature selection and deep neural networks "
            "for stock price direction forecasting using technical analysis "
            "indicators. Machine Learning with Applications, 5, 100060.",
            "Peng et al. (2021)",
        ),
        (
            "Chen, Y., & Yang, Y. (2021). The one standard error rule for "
            "model selection: Does it work? Stats, 4(4), 868–892.",
            "Chen & Yang (2021)",
        ),
        (
            "Yun, K. K., Yoon, S. W., & Won, D. (2021). Prediction of stock "
            "price direction using a hybrid GA-XGBoost algorithm with a "
            "three-stage feature engineering process. Expert Systems with "
            "Applications, 186, 115716.",
            "Yun et al. (2021)",
        ),
        (
            "Choi, J. (2021). Maximum drawdown, recovery, and momentum. "
            "Journal of Risk and Financial Management, 14(11), 542.",
            "Choi (2021)",
        ),
        (
            "Kumbure, M. M., Lohrmann, C., Luukka, P., & Porras, J. (2022). "
            "Machine learning techniques and data for stock market forecasting: "
            "A literature review. Expert Systems with Applications, 197, 116659.",
            "Kumbure et al. (2022)",
        ),
        (
            "Ghosh, P., Neufeld, A., & Sahoo, J. K. (2022). Forecasting "
            "directional movements of stock prices for intraday trading using "
            "LSTM and random forests. Finance Research Letters, 46, 102280.",
            "Ghosh et al. (2022)",
        ),
        (
            "Vuong, P. H., Dat, T. T., Mai, T. K., Uyen, P. H., & The Bao, "
            "P. (2022). Stock-price forecasting based on XGBoost and LSTM. "
            "Computer Systems Science and Engineering, 40(1), 237–246.",
            "Vuong et al. (2022)",
        ),
        (
            "Yang, J., Wang, Y., & Li, X. (2022). Prediction of stock price "
            "direction using the LASSO-LSTM model combines technical indicators "
            "and financial sentiment analysis. PeerJ Computer Science, 8, e1148.",
            "Yang et al. (2022)",
        ),
        (
            "Petrozziello, A., Troiano, L., Serra, A., Jordanov, I., Storti, G., "
            "Tagliaferri, R., & La Rocca, M. (2022). Deep learning for volatility "
            "forecasting in asset management. Soft Computing, 26, 8553–8574.",
            "Petrozziello et al. (2022)",
        ),
        (
            "Htun, H. H., Biehl, M., & Petkov, N. (2023). Survey of feature "
            "selection and extraction techniques for stock market prediction. "
            "Financial Innovation, 9, 26.",
            "Htun et al. (2023)",
        ),
        (
            "Abbasi, A., Parsons, J., Pant, G., Sheng, O. R. L., & Sarker, S. "
            "(2024). Pathways for design research on artificial intelligence. "
            "Information Systems Research, 35(2), 441–459.",
            "Abbasi et al. (2024)",
        ),
        (
            "Al-Selwi, S. M., Hassan, M. F., Abdulkadir, S. J., Muneer, A., "
            "Sumiea, E. H. H., Alqushaibi, A., & Ragab, M. G. (2024). "
            "RNN-LSTM: From applications to modeling techniques and "
            "beyond—Systematic review. Journal of King Saud University - "
            "Computer and Information Sciences, 36, 102068.",
            "Al-Selwi et al. (2024)",
        ),
        (
            "Akşehir, Z. D., & Kılıç, E. (2024). Analyzing the critical steps "
            "in deep learning-based stock forecasting: A literature review. "
            "PeerJ Computer Science, 10, e2312.",
            "Akşehir & Kılıç (2024)",
        ),
        (
            "Kim, W., Jeon, J., Jang, M., Kim, S., Lee, H., Yoo, S., & "
            "Ahn, J. (2024). Developing a dynamic feature selection system "
            "(DFSS) for stock market prediction. Applied Sciences, 14(16), 7314.",
            "Kim et al. (2024)",
        ),
        (
            "Caparrini, A., Arroyo, J., & Escayola Mansilla, J. (2024). "
            "S&P 500 stock selection using machine learning classifiers: A look "
            "into the changing role of factors. Research in International "
            "Business and Finance, 70, 102336.",
            "Caparrini et al. (2024)",
        ),
        (
            "Pabuccu, H., & Barbu, A. (2024). Feature selection with annealing "
            "for forecasting financial time series. Financial Innovation, 10, 87.",
            "Pabuccu & Barbu (2024)",
        ),
        (
            "Mostafavi, S. M., & Hooman, A. R. (2025). Key technical indicators "
            "for stock market prediction. Machine Learning with Applications, "
            "20, 100631.",
            "Mostafavi & Hooman (2025)",
        ),
    ]
    return pd.DataFrame(
        [
            {"Reference": citation, "DOI / source": REFERENCE_URLS[key]}
            for citation, key in entries
        ]
    )


def citation_links(keys: list[str]) -> str:
    """Return compact Markdown links for a literature-support caption."""

    return " · ".join(f"[{key}]({REFERENCE_URLS[key]})" for key in keys)
