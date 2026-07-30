from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_CONTEXT_DIR = PROJECT_ROOT / "data" / "context"
DATA_NEWS_DIR = PROJECT_ROOT / "data" / "news"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

DEFAULT_RAW_DATA_PATH = DATA_RAW_DIR / "TSLA_raw_2016_2026.csv"
DATASET_PATHS = {
    "TSLA": DATA_RAW_DIR / "TSLA_raw_2016_2026.csv",
    "AAPL": DATA_RAW_DIR / "AAPL_raw_2016_2026.csv",
    "NVDA": DATA_RAW_DIR / "NVDA_raw_2016_2026.csv",
}
MARKET_PROXY_PATHS = {
    "QQQ": DATA_CONTEXT_DIR / "QQQ_raw_2016_2026.csv",
}
CONTEXT_SERIES_PATHS = {
    series: DATA_CONTEXT_DIR / f"{series}.csv"
    for series in [
        "SP500",
        "NASDAQCOM",
        "DFF",
        "DGS10",
        "DGS2",
        "CPIAUCSL",
        "UNRATE",
        "VIXCLS",
        "UMCSENT",
    ]
}
DATASET_SOURCE_URLS = {
    "TSLA": "https://www.nasdaq.com/market-activity/stocks/tsla/historical",
    "AAPL": "https://www.nasdaq.com/market-activity/stocks/aapl/historical",
    "NVDA": "https://www.nasdaq.com/market-activity/stocks/nvda/historical",
}
SOURCE_MANIFEST_PATH = DATA_CONTEXT_DIR / "source_manifest.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "TSLA_cleaned_preprocessed.csv"
FEATURES_DATA_PATH = DATA_PROCESSED_DIR / "TSLA_features.csv"
ML_RESULTS_PATH = OUTPUT_TABLES_DIR / "ml_classification_results.csv"
ML_BASELINE_COMPARISON_PATH = OUTPUT_TABLES_DIR / "ml_baseline_comparison.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_TABLES_DIR / "feature_importance.csv"
WALK_FORWARD_RESULTS_PATH = OUTPUT_TABLES_DIR / "walk_forward_validation.csv"
EXPERIMENT_CONFIG_PATH = OUTPUT_TABLES_DIR / "experiment_configuration.csv"
BACKTEST_METRICS_PATH = OUTPUT_TABLES_DIR / "backtesting_metrics_comparison.csv"
BACKTEST_EQUITY_PATH = OUTPUT_TABLES_DIR / "backtesting_equity_curves.csv"
SENSITIVITY_PATH = OUTPUT_TABLES_DIR / "sma_ema_single_window_sensitivity.csv"
VALIDATION_REPORT_PATH = OUTPUT_TABLES_DIR / "validation_report.csv"
SENTIMENT_VALIDATION_REPORT_PATH = (
    OUTPUT_TABLES_DIR / "v7_sentiment_validation_report.csv"
)
COMBINED_VALIDATION_REPORT_PATH = (
    OUTPUT_TABLES_DIR / "v7_combined_validation_report.csv"
)
MODEL_PATH = OUTPUT_MODELS_DIR / "tsla_direction_model.joblib"
DATASET_SUMMARY_PATH = OUTPUT_TABLES_DIR / "multi_asset_dataset_summary.csv"
PREPROCESSING_AUDIT_PATH = OUTPUT_TABLES_DIR / "preprocessing_audit.csv"
FEATURE_SELECTION_PATH = OUTPUT_TABLES_DIR / "feature_selection_results.csv"
FEATURE_ABLATION_PATH = OUTPUT_TABLES_DIR / "feature_group_ablation.csv"
TUNING_RESULTS_PATH = OUTPUT_TABLES_DIR / "hyperparameter_tuning_results.csv"
CROSS_ASSET_BENCHMARK_PATH = OUTPUT_TABLES_DIR / "cross_asset_model_benchmark.csv"
CROSS_ASSET_SUMMARY_PATH = OUTPUT_TABLES_DIR / "cross_asset_model_summary.csv"
FEATURE_STABILITY_PATH = OUTPUT_TABLES_DIR / "cross_asset_feature_stability.csv"
OOF_PREDICTIONS_PATH = OUTPUT_TABLES_DIR / "walk_forward_oof_predictions.csv"
OOF_BACKTEST_METRICS_PATH = OUTPUT_TABLES_DIR / "walk_forward_oof_backtest_metrics.csv"
OOF_BACKTEST_EQUITY_PATH = OUTPUT_TABLES_DIR / "walk_forward_oof_backtest_equity.csv"
OOF_SIGNAL_DIAGNOSTICS_PATH = (
    OUTPUT_TABLES_DIR / "walk_forward_oof_signal_diagnostics.csv"
)
OOF_NESTED_TUNING_PATH = (
    OUTPUT_TABLES_DIR / "walk_forward_nested_tuning_results.csv"
)
OOF_LSTM_SEED_RESULTS_PATH = (
    OUTPUT_TABLES_DIR / "walk_forward_lstm_seed_results.csv"
)
OOF_LSTM_SEED_PREDICTIONS_PATH = (
    OUTPUT_TABLES_DIR / "walk_forward_lstm_seed_predictions.csv"
)
MAIN_OOF_CLASSIFICATION_UNCERTAINTY_PATH = (
    OUTPUT_TABLES_DIR / "main_oof_classification_uncertainty.csv"
)
MAIN_OOF_RETURN_SEPARATION_UNCERTAINTY_PATH = (
    OUTPUT_TABLES_DIR / "main_oof_return_separation_uncertainty.csv"
)
MAIN_OOF_STRATEGY_BENCHMARK_UNCERTAINTY_PATH = (
    OUTPUT_TABLES_DIR / "main_oof_strategy_benchmark_uncertainty.csv"
)
FIXED_MODEL_PREDICTIONS_PATH = (
    OUTPUT_TABLES_DIR / "fixed_holdout_model_predictions.csv"
)
TEMPORAL_SPLIT_MANIFEST_PATH = OUTPUT_TABLES_DIR / "temporal_split_manifest.csv"
HELDOUT_BACKTEST_EXECUTION_PATH = (
    OUTPUT_TABLES_DIR / "fixed_holdout_backtest_execution.csv"
)
OOF_BACKTEST_EXECUTION_PATH = (
    OUTPUT_TABLES_DIR / "walk_forward_oof_backtest_execution.csv"
)
HELDOUT_TRADE_LEDGER_PATH = OUTPUT_TABLES_DIR / "fixed_holdout_trade_ledger.csv"
OOF_TRADE_LEDGER_PATH = OUTPUT_TABLES_DIR / "walk_forward_oof_trade_ledger.csv"
CONTEXT_AUDIT_PATH = OUTPUT_TABLES_DIR / "context_availability_audit.csv"
FEATURE_FAMILY_ABLATION_PATH = OUTPUT_TABLES_DIR / "feature_family_ablation.csv"
LSTM_RESULTS_PATH = OUTPUT_TABLES_DIR / "lstm_classification_results.csv"
LSTM_TRAINING_PATH = OUTPUT_TABLES_DIR / "lstm_training_history.csv"
LSTM_MODEL_PATH = OUTPUT_MODELS_DIR / "tsla_lstm_direction.keras"
LSTM_PREPROCESSOR_PATH = OUTPUT_MODELS_DIR / "tsla_lstm_preprocessor.joblib"
LSTM_METADATA_PATH = OUTPUT_MODELS_DIR / "tsla_lstm_metadata.json"
NEWS_SESSION_CALENDAR_PATH = DATA_NEWS_DIR / "xnys_sessions_2021_2024.csv"
NEWS_ARTICLE_SCORES_PATH = DATA_NEWS_DIR / "article_sentiment_scores.csv"
DAILY_NEWS_FEATURES_PATH = DATA_NEWS_DIR / "daily_news_features.csv"
NEWS_SOURCE_MANIFEST_PATH = DATA_NEWS_DIR / "source_manifest.csv"
NEWS_COVERAGE_AUDIT_PATH = OUTPUT_TABLES_DIR / "news_coverage_audit.csv"
NEWS_DEDUP_AUDIT_PATH = OUTPUT_TABLES_DIR / "news_dedup_audit.csv"
SENTIMENT_OOF_PATH = OUTPUT_TABLES_DIR / "v7_sentiment_oof_predictions.csv"
SENTIMENT_ABLATION_PATH = OUTPUT_TABLES_DIR / "v7_f0_f5_ablation.csv"
SENTIMENT_THRESHOLDS_PATH = OUTPUT_TABLES_DIR / "v7_fold_target_thresholds.csv"
SENTIMENT_BOOTSTRAP_PATH = OUTPUT_TABLES_DIR / "v7_bootstrap_confidence_intervals.csv"
SENTIMENT_PLACEBO_PATH = OUTPUT_TABLES_DIR / "v7_placebo_results.csv"
SENTIMENT_PLACEBO_DISTRIBUTION_PATH = (
    OUTPUT_TABLES_DIR / "v7_placebo_null_distribution.csv"
)
SENTIMENT_SUMMARY_PATH = OUTPUT_TABLES_DIR / "v7_sentiment_study_summary.csv"

FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_MODEL_REVISION = "1b7521e9fc2e2a12e93de7e3e19373b0505d353f"
FNSPID_DATASET_ID = "benstaf/FNSPID-filtered-nasdaq-100"
FNSPID_DATASET_REVISION = "c5cebb37e76d64d27d79ca72ede8f93b84e1a8aa"
SENTIMENT_STUDY_START = "2022-06-03"
SENTIMENT_STUDY_END = "2023-12-16"

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
RAW_DATASET_FIELDS = ["Date", "Close/Last", "Volume", "Open", "High", "Low"]
TARGET_RETURN_COLUMN = "Weekly_Trend_Return"
TARGET_ABS_RETURN_COLUMN = "Weekly_Trend_Abs_Return"
TARGET_DIRECTION_COLUMN = "Weekly_Trend_Direction"
TARGET_EXECUTION_DATE_COLUMN = "Target_Execution_Date"
TARGET_VALUATION_DATE_COLUMN = "Target_Valuation_Date"
TARGET_DATE_COLUMN = "Target_Date"
TARGET_HORIZON_SESSIONS = 5
TARGET_TRAILING_SESSIONS = 5
# The classification label uses Close[t+1:t+5]. Five origin rows are therefore
# purged between training and validation/test anchors. The separate downstream
# decision layer executes at Open[t+1] and closes/reconsiders the same sleeve at
# Open[t+6]; the two horizons are aligned but not mathematically identical.
TARGET_PURGE_ROWS = TARGET_HORIZON_SESSIONS
SUPPLEMENTARY_TARGET_RETURN_COLUMN = "Next_Close_to_Close_Return"
SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN = "Next_Close_to_Close_Abs_Return"
SUPPLEMENTARY_TARGET_DIRECTION_COLUMN = "Next_Close_to_Close_Direction"
SUPPLEMENTARY_TARGET_PURGE_ROWS = 1
PREPROCESSED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Price",
    "Volume",
    "Daily_Return",
    "Cumulative_Return",
]

# Frozen EMA-aware Technical specification carried forward from the revised
# RQ2 feature-family comparison into RQ3, RQ4, and the exploratory Sandbox.
# These six channels are pre-specified; they are not selected from the legacy
# broad candidate pool with MI, correlation filtering, or a Top-k control.
PRESPECIFIED_OOF_FEATURES = [
    "Daily_Return",
    "Momentum_5",
    "EMA8_13_State",
    "Volatility_20",
    "Drawdown",
    "Volume_Ratio",
]

# Backward-compatible public alias for callers and historical tests.
SELECTED_FEATURES = PRESPECIFIED_OOF_FEATURES

CANDIDATE_FEATURES = [
    "Daily_Return",
    "Lag_Return_1",
    "Lag_Return_3",
    "Lag_Return_5",
    "Momentum_5",
    "Momentum_10",
    "SMA10_Ratio",
    "SMA20_Ratio",
    "SMA50_Ratio",
    "EMA8_Ratio",
    "EMA13_Ratio",
    "EMA21_Ratio",
    "EMA8_13_State",
    "MACD",
    "MACD_Histogram",
    "RSI_14",
    "Volatility_10",
    "Volatility_20",
    "ATR_14",
    "Bollinger_Band_Width",
    "Volume_Change",
    "Volume_Ratio",
    "OBV_Change",
    "High_Low_Range",
    "Open_Close_Change",
    "Drawdown",
    "QQQ_Return",
    "QQQ_Volatility_20",
    "SP500_Return",
    "SP500_Volatility_20",
    "NASDAQ_Return",
    "NASDAQ_Volatility_20",
    "Fed_Funds_Rate",
    "Treasury_10Y_Yield",
    "Yield_Curve_10Y2Y",
    "CPI_YoY",
    "Unemployment_Rate",
    "VIX_Level",
    "VIX_Change_5",
    "Consumer_Sentiment",
    "Consumer_Sentiment_Change",
]

FEATURE_GROUPS = {
    "Return and momentum": {
        "features": [
            "Daily_Return",
            "Lag_Return_1",
            "Lag_Return_3",
            "Lag_Return_5",
            "Momentum_5",
            "Momentum_10",
        ],
        "purpose": "Capture recent returns and short-term price continuation or reversal.",
    },
    "Trend": {
        "features": [
            "SMA10_Ratio",
            "SMA20_Ratio",
            "SMA50_Ratio",
            "EMA8_Ratio",
            "EMA13_Ratio",
            "EMA21_Ratio",
            "EMA8_13_State",
            "MACD",
            "MACD_Histogram",
        ],
        "purpose": "Represent price position, moving-average structure, and trend changes.",
    },
    "Oscillator": {
        "features": ["RSI_14"],
        "purpose": "Measure the balance of recent gains and losses.",
    },
    "Volatility": {
        "features": ["Volatility_10", "Volatility_20", "ATR_14", "Bollinger_Band_Width"],
        "purpose": "Measure recent return variation, trading range, and volatility expansion.",
    },
    "Volume": {
        "features": ["Volume_Change", "Volume_Ratio", "OBV_Change"],
        "purpose": "Represent changes in trading activity and relative volume.",
    },
    "Range and risk": {
        "features": ["High_Low_Range", "Open_Close_Change", "Drawdown"],
        "purpose": "Capture intraday range and downside condition relative to the historical peak.",
    },
    "Market context": {
        "features": [
            "QQQ_Return",
            "QQQ_Volatility_20",
            "SP500_Return",
            "SP500_Volatility_20",
            "NASDAQ_Return",
            "NASDAQ_Volatility_20",
        ],
        "purpose": (
            "Represent broad-market and technology-market conditions with QQQ, "
            "S&P 500, and Nasdaq Composite information available by day t."
        ),
    },
    "Macroeconomic context": {
        "features": [
            "Fed_Funds_Rate",
            "Treasury_10Y_Yield",
            "Yield_Curve_10Y2Y",
            "CPI_YoY",
            "Unemployment_Rate",
        ],
        "purpose": (
            "Represent monetary policy, long-term rates, the yield curve, "
            "inflation, and labour-market conditions using release-aware joins."
        ),
    },
    "Market-implied risk and survey-based sentiment proxies": {
        "features": [
            "VIX_Level",
            "VIX_Change_5",
            "Consumer_Sentiment",
            "Consumer_Sentiment_Change",
        ],
        "purpose": (
            "Represent market fear and survey-based consumer sentiment without "
            "claiming ticker-specific NLP news sentiment."
        ),
    },
    "Target": {
        "features": [
            TARGET_RETURN_COLUMN,
            TARGET_DIRECTION_COLUMN,
            TARGET_ABS_RETURN_COLUMN,
            SUPPLEMENTARY_TARGET_RETURN_COLUMN,
            SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
            SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
        ],
        "purpose": (
            "Define the five-session forward-versus-trailing mean-close return "
            "and direction used by the v8 classification study. Execution and "
            "valuation dates identify the separate Open[t+1] to Open[t+6] "
            "downstream decision interval. Close-to-close labels remain only for "
            "the bounded supplementary news study. None are input features."
        ),
    },
}

NEWS_ATTENTION_FEATURES = [
    "Log_News_Count",
    "Unique_Story_Count",
    "Unique_Source_Count",
    "Abnormal_News_Volume_20",
    "No_News",
]

NEWS_SENTIMENT_FEATURES = [
    "FinBERT_Sentiment_Balance",
    "FinBERT_Positive_Share",
    "FinBERT_Negative_Share",
    "FinBERT_Neutral_Share",
    "FinBERT_Disagreement",
    "FinBERT_Max_Negative",
    "FinBERT_Sentiment_3D",
    "FinBERT_Sentiment_Surprise_20",
]

SENTIMENT_PROXY_FEATURES = [
    "VIX_Level",
    "VIX_Change_5",
    "Consumer_Sentiment",
    "Consumer_Sentiment_Change",
]

SENTIMENT_BASE_FEATURES = [
    feature
    for group, details in FEATURE_GROUPS.items()
    if group
    not in {
        "Market-implied risk and survey-based sentiment proxies",
        "Target",
    }
    for feature in details["features"]
]

SENTIMENT_FEATURE_SETS = {
    "F0": SENTIMENT_BASE_FEATURES,
    "F1": [*SENTIMENT_BASE_FEATURES, *SENTIMENT_PROXY_FEATURES],
    "F2": [*SENTIMENT_BASE_FEATURES, *NEWS_ATTENTION_FEATURES],
    "F3": [*SENTIMENT_BASE_FEATURES, *NEWS_SENTIMENT_FEATURES],
    "F4": [
        *SENTIMENT_BASE_FEATURES,
        *NEWS_ATTENTION_FEATURES,
        *NEWS_SENTIMENT_FEATURES,
    ],
    "F5": [
        *SENTIMENT_BASE_FEATURES,
        *SENTIMENT_PROXY_FEATURES,
        *NEWS_ATTENTION_FEATURES,
        *NEWS_SENTIMENT_FEATURES,
    ],
}

SMA_WINDOWS = [5, 10, 20, 50, 100]
EMA_WINDOWS = [8, 13, 21, 50]

MAIN_HEATMAP_FEATURES = [
    "Price",
    "Volume",
    "Daily_Return",
    "Lag_Return_1",
    "SMA10_Ratio",
    "SMA20_Ratio",
    "EMA8_Ratio",
    "EMA13_Ratio",
    "Volatility_10",
    "Volatility_20",
    "Volume_Change",
    "High_Low_Range",
    "Drawdown",
    "RSI_14",
    "MACD_Histogram",
    "SP500_Return",
    "NASDAQ_Return",
    "Treasury_10Y_Yield",
    "VIX_Level",
    "Consumer_Sentiment",
    TARGET_DIRECTION_COLUMN,
]

TABULAR_MODEL_NAMES = [
    "Logistic Regression (Baseline)",
    "Random Forest Classifier",
    "XGBoost Classifier",
]
LSTM_MODEL_NAME = "LSTM Classifier"
ALL_MODEL_NAMES = [*TABULAR_MODEL_NAMES, LSTM_MODEL_NAME]
MAJORITY_MODEL_NAME = "Majority Baseline"
MODEL_LEVEL_NAMES = [MAJORITY_MODEL_NAME, *ALL_MODEL_NAMES]

DEFAULT_LSTM_LOOKBACK = 20
DEFAULT_LSTM_UNITS = 16
DEFAULT_LSTM_DROPOUT = 0.20
DEFAULT_LSTM_EPOCHS = 60
DEFAULT_LSTM_BATCH_SIZE = 32
DEFAULT_LSTM_SEEDS = [42, 43, 44]


def ensure_project_directories() -> None:
    for directory in [
        DATA_RAW_DIR,
        DATA_CONTEXT_DIR,
        DATA_NEWS_DIR,
        DATA_PROCESSED_DIR,
        OUTPUT_TABLES_DIR,
        OUTPUT_FIGURES_DIR,
        OUTPUT_MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
