from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    NEWS_ATTENTION_FEATURES,
    NEWS_SENTIMENT_FEATURES,
    MODEL_LEVEL_NAMES,
    REQUIRED_COLUMNS,
    SELECTED_FEATURES,
    SENTIMENT_FEATURE_SETS,
    SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
    SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
    SUPPLEMENTARY_TARGET_RETURN_COLUMN,
    TARGET_ABS_RETURN_COLUMN,
    TARGET_DIRECTION_COLUMN,
    TARGET_EXECUTION_DATE_COLUMN,
    TARGET_HORIZON_SESSIONS,
    TARGET_PURGE_ROWS,
    TARGET_RETURN_COLUMN,
    TARGET_TRAILING_SESSIONS,
    TARGET_VALUATION_DATE_COLUMN,
)
from .ml import MLResult
from .selection import FeatureSelectionResult
from .temporal import TemporalSplitPlan


TABULAR_TUNED_MODELS = {
    "Logistic Regression (Baseline)",
    "Random Forest Classifier",
    "XGBoost Classifier",
}


def _row(test_id: str, area: str, check: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "Test ID": test_id,
        "Area": area,
        "Check": check,
        "Expected Result": "Pass",
        "Actual Result": "Pass" if passed else "Fail",
        "Status": "Pass" if passed else "Fail",
        "Evidence": evidence,
    }


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _row(
            "D01",
            "Data",
            "Required columns are present",
            all(column in df.columns for column in REQUIRED_COLUMNS),
            ", ".join(df.columns),
        ),
        _row(
            "D02",
            "Data",
            "Dates are chronological",
            bool(df["Date"].is_monotonic_increasing),
            f"{df['Date'].min()} to {df['Date'].max()}",
        ),
        _row(
            "D03",
            "Data",
            "No duplicate dates remain",
            not bool(df["Date"].duplicated().any()),
            f"Duplicate count: {int(df['Date'].duplicated().sum())}",
        ),
        _row(
            "D04",
            "Data",
            "Required fields have no missing values",
            int(df[REQUIRED_COLUMNS].isna().sum().sum()) == 0,
            f"Missing required cells: {int(df[REQUIRED_COLUMNS].isna().sum().sum())}",
        ),
        _row(
            "D05",
            "Data",
            "Price values are positive",
            bool((df[["Open", "High", "Low", "Close"]] > 0).all().all()),
            "OHLC minimum values checked",
        ),
        _row(
            "D06",
            "Data",
            "High is not below Low",
            bool((df["High"] >= df["Low"]).all()),
            "High/Low consistency check",
        ),
        _row(
            "D07",
            "Data",
            "Volume is non-negative",
            bool((df["Volume"] >= 0).all()),
            "Volume consistency check",
        ),
    ]
    return pd.DataFrame(rows)


def validate_features(
    engineered: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    features = feature_names or SELECTED_FEATURES
    selected_exist = all(feature in engineered.columns for feature in features)
    target_columns = {
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        TARGET_ABS_RETURN_COLUMN,
        SUPPLEMENTARY_TARGET_RETURN_COLUMN,
        SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
        SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
    }
    no_target_leakage = not any(
        feature in target_columns for feature in features
    )
    finite_count = int(
        np.isinf(engineered[features].select_dtypes(include=np.number)).sum().sum()
    )
    trailing_mean = (
        engineered["Close"].rolling(TARGET_TRAILING_SESSIONS).mean()
    )
    future_mean = (
        engineered["Close"]
        .shift(-1)
        .rolling(TARGET_HORIZON_SESSIONS)
        .mean()
        .shift(-(TARGET_HORIZON_SESSIONS - 1))
    )
    expected_target = future_mean / trailing_mean - 1.0
    target_mask = engineered[TARGET_RETURN_COLUMN].notna() & expected_target.notna()
    target_aligned = bool(
        target_mask.any()
        and np.allclose(
            engineered.loc[target_mask, TARGET_RETURN_COLUMN],
            expected_target.loc[target_mask],
        )
    )
    target_dates_aligned = bool(
        pd.to_datetime(engineered[TARGET_EXECUTION_DATE_COLUMN]).equals(
            pd.to_datetime(engineered["Date"].shift(-1))
        )
        and pd.to_datetime(engineered[TARGET_VALUATION_DATE_COLUMN]).equals(
            pd.to_datetime(
                engineered["Date"].shift(-(TARGET_HORIZON_SESSIONS + 1))
            )
        )
    )
    rows = [
        _row(
            "F01",
            "Features",
            "All selected features exist",
            selected_exist,
            f"Selected feature count: {len(features)}",
        ),
        _row(
            "F02",
            "Features",
            "Targets are excluded from inputs",
            no_target_leakage,
            "Primary and supplementary outcome fields are labels only",
        ),
        _row(
            "F03",
            "Features",
            "No infinite values in selected features",
            finite_count == 0,
            f"Infinite cells: {finite_count}",
        ),
        _row(
            "F04",
            "Features",
            "Prediction target contains both classes",
            engineered[TARGET_DIRECTION_COLUMN].dropna().nunique() == 2,
            (
                "Classes: "
                f"{sorted(engineered[TARGET_DIRECTION_COLUMN].dropna().unique().tolist())}"
            ),
        ),
        _row(
            "F05",
            "Features",
            "Prediction target equals the five-session future-versus-trailing mean-close return",
            target_aligned,
            f"Aligned labelled rows: {int(target_mask.sum())}",
        ),
        _row(
            "F06",
            "Features",
            "Downstream execution and valuation dates are shifted by one and six trading rows",
            target_dates_aligned,
            "Checked row-based Target_Execution_Date and Target_Valuation_Date",
        ),
    ]
    return pd.DataFrame(rows)


def validate_context(
    context_audit: pd.DataFrame,
    engineered: pd.DataFrame,
) -> pd.DataFrame:
    external_features = [
        "QQQ_Return",
        "SP500_Return",
        "NASDAQ_Return",
        "Fed_Funds_Rate",
        "Treasury_10Y_Yield",
        "Yield_Curve_10Y2Y",
        "CPI_YoY",
        "Unemployment_Rate",
        "VIX_Level",
        "Consumer_Sentiment",
    ]
    available = [column for column in external_features if column in engineered.columns]
    violations = (
        int(context_audit["Future Availability Violations"].sum())
        if not context_audit.empty
        else -1
    )
    complete_rows = int(engineered[available].dropna().shape[0]) if available else 0
    return pd.DataFrame(
        [
            _row(
                "C01",
                "External Context",
                "No context value is merged before its availability date",
                violations == 0,
                f"Future availability violations: {violations}",
            ),
            _row(
                "C02",
                "External Context",
                "Market, macroeconomic and sentiment-proxy inputs are present",
                len(available) == len(external_features),
                f"Available features: {len(available)}/{len(external_features)}",
            ),
            _row(
                "C03",
                "External Context",
                "External inputs provide sufficient complete observations",
                complete_rows >= 100,
                f"Complete external-context rows: {complete_rows}",
            ),
        ]
    )


def validate_ml_result(ml_result: MLResult | None) -> pd.DataFrame:
    if ml_result is None:
        return pd.DataFrame(
            [
                _row(
                    "M01",
                    "Machine Learning",
                    "Model result is available",
                    False,
                    "Model could not be trained",
                )
            ]
        )
    metrics_finite = all(
        pd.isna(value) or np.isfinite(value) for value in ml_result.metrics.values()
    )
    expected_models = {
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }
    if ml_result.model_name == "LSTM Classifier":
        expected_models.add("LSTM")
    rows = [
        _row(
            "M01",
            "Machine Learning",
            "Chronological train/test split",
            ml_result.train_end_date < ml_result.test_start_date,
            f"Train end: {ml_result.train_end_date.date()}, test start: {ml_result.test_start_date.date()}",
        ),
        _row(
            "M02",
            "Machine Learning",
            "Classification metrics are finite",
            metrics_finite,
            str(ml_result.metrics),
        ),
        _row(
            "M03",
            "Machine Learning",
            "Predictions align with test rows",
            len(ml_result.results) == ml_result.test_rows,
            f"Predictions: {len(ml_result.results)}, test rows: {ml_result.test_rows}",
        ),
        _row(
            "M04",
            "Machine Learning",
            "Feature importance is available",
            not ml_result.feature_importance.empty,
            f"Rows: {len(ml_result.feature_importance)}",
        ),
        _row(
            "M05",
            "Machine Learning",
            "Training and held-out date ranges do not overlap",
            ml_result.train_end_date < ml_result.test_start_date,
            (
                f"Train: {ml_result.train_start_date.date()} to {ml_result.train_end_date.date()}; "
                f"test: {ml_result.test_start_date.date()} to {ml_result.test_end_date.date()}"
            ),
        ),
        _row(
            "M06",
            "Machine Learning",
            "Required baseline and model comparators are available",
            set(ml_result.baseline_comparison.get("Model", [])).issuperset(
                expected_models
            ),
            f"Rows: {len(ml_result.baseline_comparison)}",
        ),
    ]
    return pd.DataFrame(rows)


def validate_temporal_plan_alignment(
    selection: FeatureSelectionResult,
    ml_result: MLResult | None,
    split_plan: TemporalSplitPlan,
) -> pd.DataFrame:
    """Verify that selection and final modelling share one untouched holdout."""

    model_available = ml_result is not None
    identifiers_match = bool(
        model_available
        and selection.plan_id == split_plan.plan_id == ml_result.split_plan_id
        and selection.plan_hash == split_plan.plan_hash == ml_result.split_plan_hash
        and len(split_plan.plan_hash) == 64
    )
    selection_stops_before_holdout = bool(
        selection.train_end_date == split_plan.train_end_date
        and selection.gap_start_date == split_plan.gap_start_date
        and selection.test_start_date == split_plan.test_start_date
        and selection.train_end_date < split_plan.gap_start_date
        and split_plan.gap_end_date < split_plan.test_start_date
    )
    expected_model_train_rows = (
        len(split_plan.train_dates)
        - max(int(getattr(ml_result, "lookback", 1)) - 1, 0)
        if model_available
        else len(split_plan.train_dates)
    )
    model_uses_exact_boundaries = bool(
        model_available
        and ml_result.train_start_date == split_plan.train_start_date
        and ml_result.train_end_date == split_plan.train_end_date
        and ml_result.purge_start_date == split_plan.gap_start_date
        and ml_result.purge_end_date == split_plan.gap_end_date
        and ml_result.test_start_date == split_plan.test_start_date
        and ml_result.test_end_date == split_plan.test_end_date
        and ml_result.train_rows == expected_model_train_rows
        and ml_result.test_rows == len(split_plan.test_dates)
    )
    return pd.DataFrame(
        [
            _row(
                "T01",
                "Temporal Plan",
                "Feature selection and final model share one plan ID and SHA-256",
                identifiers_match,
                (
                    f"Plan: {split_plan.plan_id}; selection: {selection.plan_id}; "
                    f"model: {getattr(ml_result, 'split_plan_id', None)}"
                ),
            ),
            _row(
                "T02",
                "Temporal Plan",
                "Feature selection stops before the purge and untouched test period",
                selection_stops_before_holdout,
                (
                    f"Selection end: {selection.train_end_date.date()}; "
                    f"purge: {split_plan.gap_start_date.date()}; "
                    f"test start: {split_plan.test_start_date.date()}"
                ),
            ),
            _row(
                "T03",
                "Temporal Plan",
                "Final model uses the exact canonical train, purge and test boundaries",
                model_uses_exact_boundaries,
                (
                    f"Train rows: {getattr(ml_result, 'train_rows', None)}/"
                    f"{expected_model_train_rows} expected after lookback; test rows: "
                    f"{getattr(ml_result, 'test_rows', None)}/{len(split_plan.test_dates)}"
                ),
            ),
            _row(
                "T04",
                "Temporal Plan",
                "Canonical purge covers the full five-session label horizon",
                split_plan.gap == TARGET_PURGE_ROWS
                and len(split_plan.gap_dates) == TARGET_PURGE_ROWS,
                (
                    f"Gap rows: {split_plan.gap}; required: {TARGET_PURGE_ROWS}; "
                    f"purge dates: {split_plan.gap_start_date.date()} to "
                    f"{split_plan.gap_end_date.date()}"
                ),
            ),
        ]
    )


def validate_backtest(
    metrics: pd.DataFrame,
    equity: pd.DataFrame,
    strategy_frame: pd.DataFrame,
) -> pd.DataFrame:
    rule_models = {
        "Buy-and-Hold",
        "Price-above-SMA10",
        "Price-above-SMA20",
        "SMA10/SMA20 Crossover",
        "Price-above-EMA8",
        "Price-above-EMA13",
        "Price-above-EMA21",
        "EMA8/EMA13 Crossover",
    }
    actual_models = set(metrics["Model"]) if not metrics.empty else set()
    signal_models = (
        set(strategy_frame["Signal Model"].dropna().astype(str))
        if "Signal Model" in strategy_frame.columns
        else set()
    )
    if signal_models:
        signal_metric_models = actual_models.difference(rule_models)
        signals_represented = len(signal_metric_models) == len(signal_models) and all(
            any(metric_name.startswith(model_name) for metric_name in signal_metric_models)
            for model_name in signal_models
        )
    else:
        signal_metric_models = {"ML Prediction Strategy"}
        signals_represented = "ML Prediction Strategy" in actual_models
    expected_models_present = rule_models.issubset(actual_models) and signals_represented
    positions = [column for column in strategy_frame.columns if column.endswith("_Position")]
    positions_binary = all(
        set(strategy_frame[column].dropna().unique()).issubset({0, 1}) for column in positions
    ) if positions else False
    numeric_metrics = metrics.select_dtypes(include=np.number)
    allowed_missing = {"Trade-Level Win Rate"}
    required_numeric = numeric_metrics.drop(
        columns=list(allowed_missing.intersection(numeric_metrics.columns))
    )
    optional_numeric = numeric_metrics[
        list(allowed_missing.intersection(numeric_metrics.columns))
    ]
    optional_values = optional_numeric.to_numpy(dtype=float)
    optional_reported = optional_values[~np.isnan(optional_values)]
    finite_metrics = (
        not metrics.empty
        and np.isfinite(required_numeric.to_numpy()).all()
        and np.isfinite(optional_reported).all()
    )
    positive_equity = not equity.empty and bool((equity["Equity"] > 0).all())
    initial_drawdown_correct = False
    if not equity.empty:
        first_rows = equity.sort_values("Date").groupby("Model", as_index=False).first()
        expected_first_drawdown = np.minimum(first_rows["Equity"] - 1.0, 0.0)
        initial_drawdown_correct = bool(
            np.allclose(first_rows["Drawdown"], expected_first_drawdown)
        )
    rule_definitions = {
        "SMA10_Position": strategy_frame.get("Signal Price", pd.Series(dtype=float))
        > strategy_frame.get("SMA10", pd.Series(dtype=float)),
        "SMA20_Position": strategy_frame.get("Signal Price", pd.Series(dtype=float))
        > strategy_frame.get("SMA20", pd.Series(dtype=float)),
        "SMA10_SMA20_Crossover_Position": strategy_frame.get(
            "SMA10", pd.Series(dtype=float)
        )
        > strategy_frame.get("SMA20", pd.Series(dtype=float)),
        "EMA8_Position": strategy_frame.get("Signal Price", pd.Series(dtype=float))
        > strategy_frame.get("EMA8", pd.Series(dtype=float)),
        "EMA13_Position": strategy_frame.get("Signal Price", pd.Series(dtype=float))
        > strategy_frame.get("EMA13", pd.Series(dtype=float)),
        "EMA21_Position": strategy_frame.get("Signal Price", pd.Series(dtype=float))
        > strategy_frame.get("EMA21", pd.Series(dtype=float)),
        "EMA8_EMA13_Crossover_Position": strategy_frame.get(
            "EMA8", pd.Series(dtype=float)
        )
        > strategy_frame.get("EMA13", pd.Series(dtype=float)),
    }
    rule_identities_correct = bool(
        not strategy_frame.empty
        and all(
            column in strategy_frame.columns
            and strategy_frame[column].reset_index(drop=True).equals(
                expected.astype(int).reset_index(drop=True)
            )
            for column, expected in rule_definitions.items()
        )
    )
    target_interval_identity = bool(
        not strategy_frame.empty
        and TARGET_RETURN_COLUMN in strategy_frame.columns
        and TARGET_EXECUTION_DATE_COLUMN in strategy_frame.columns
        and TARGET_VALUATION_DATE_COLUMN in strategy_frame.columns
        and np.allclose(
            strategy_frame[TARGET_RETURN_COLUMN],
            strategy_frame["Open-to-Open Return"],
            rtol=0.0,
            atol=1e-12,
        )
        and pd.to_datetime(strategy_frame[TARGET_EXECUTION_DATE_COLUMN])
        .reset_index(drop=True)
        .equals(
            pd.to_datetime(strategy_frame["Execution Date"]).reset_index(drop=True)
        )
        and pd.to_datetime(strategy_frame[TARGET_VALUATION_DATE_COLUMN])
        .reset_index(drop=True)
        .equals(
            pd.to_datetime(strategy_frame["Valuation Date"]).reset_index(drop=True)
        )
    )
    rows = [
        _row(
            "B01",
            "Backtesting",
            "Expected comparison models are present",
            expected_models_present,
            f"Models: {sorted(actual_models)}",
        ),
        _row(
            "B02",
            "Backtesting",
            "Positions are binary",
            positions_binary,
            f"Position columns: {positions}",
        ),
        _row(
            "B03",
            "Backtesting",
            "Reported metrics are finite (undefined closed-trade win rate may be N/A)",
            finite_metrics,
            f"Metric rows: {len(metrics)}",
        ),
        _row(
            "B04",
            "Backtesting",
            "Equity curves remain positive",
            positive_equity,
            f"Equity rows: {len(equity)}",
        ),
        _row(
            "B05",
            "Backtesting",
            "Drawdown uses initial capital as the first high-water mark",
            initial_drawdown_correct,
            f"Models checked: {len(first_rows) if not equity.empty else 0}",
        ),
        _row(
            "B06",
            "Backtesting",
            "SMA and EMA positions match their documented price-above and crossover rules",
            rule_identities_correct,
            (
                "Checked Price-above-SMA10/20, SMA10/SMA20 crossover, "
                "Price-above-EMA8/13/21 and EMA8/EMA13 crossover"
            ),
        ),
        _row(
            "B07",
            "Backtesting",
            "Prediction label and realised gross return use the identical execution interval",
            target_interval_identity,
            (
                "Checked target return plus execution/valuation dates against "
                "Open-to-Open Return"
            ),
        ),
    ]
    return pd.DataFrame(rows)


def _allclose_on_overlap(
    observed: pd.Series,
    expected: pd.Series,
    *,
    atol: float = 1e-8,
) -> tuple[bool, int]:
    """Compare two numeric series only where both contain an auditable value."""
    left = pd.to_numeric(observed, errors="coerce")
    right = pd.to_numeric(expected, errors="coerce")
    mask = left.notna() & right.notna()
    compared = int(mask.sum())
    return (
        bool(
            compared > 0
            and np.allclose(
                left.loc[mask].to_numpy(),
                right.loc[mask].to_numpy(),
                rtol=1e-6,
                atol=atol,
            )
        ),
        compared,
    )


def _rolling_news_features_are_causal(daily_news: pd.DataFrame) -> tuple[bool, str]:
    """Reconstruct the documented rolling features using current/past rows only."""
    required = {
        "Asset",
        "Date",
        "Unique_Story_Count",
        "FinBERT_Sentiment_Balance",
        "Abnormal_News_Volume_20",
        "FinBERT_Sentiment_3D",
        "FinBERT_Sentiment_Surprise_20",
    }
    if not required.issubset(daily_news.columns):
        missing = sorted(required.difference(daily_news.columns))
        return False, f"Cannot reconstruct causal features; missing: {missing}"

    matches: dict[str, bool] = {
        "Abnormal_News_Volume_20": True,
        "FinBERT_Sentiment_3D": True,
        "FinBERT_Sentiment_Surprise_20": True,
    }
    compared: dict[str, int] = {key: 0 for key in matches}
    for _, asset_frame in daily_news.groupby("Asset", sort=False):
        asset_frame = asset_frame.sort_values("Date")
        count = pd.to_numeric(asset_frame["Unique_Story_Count"], errors="coerce")
        balance = pd.to_numeric(
            asset_frame["FinBERT_Sentiment_Balance"], errors="coerce"
        )

        # The formal aggregation uses preceding-session baselines.  Supporting
        # a short warm-up (min_periods 1/5/10/20) keeps this validator compatible
        # with a source extract that includes or excludes pre-window warm-up rows.
        abnormal_match = False
        abnormal_compared = 0
        preceding_count = count.shift(1)
        for min_periods in (1, 5, 10, 20):
            history = preceding_count.rolling(20, min_periods=min_periods)
            mean = history.mean()
            for ddof in (0, 1):
                std = history.std(ddof=ddof)
                expected = (count - mean) / std.replace(0, np.nan)
                ok, n_rows = _allclose_on_overlap(
                    asset_frame["Abnormal_News_Volume_20"], expected
                )
                if ok and n_rows > abnormal_compared:
                    abnormal_match = True
                    abnormal_compared = n_rows
        matches["Abnormal_News_Volume_20"] &= abnormal_match
        compared["Abnormal_News_Volume_20"] += abnormal_compared

        surprise_match = False
        surprise_compared = 0
        preceding_balance = balance.shift(1)
        for min_periods in (1, 5, 10, 20):
            expected = balance - preceding_balance.rolling(
                20, min_periods=min_periods
            ).mean()
            ok, n_rows = _allclose_on_overlap(
                asset_frame["FinBERT_Sentiment_Surprise_20"], expected
            )
            if ok and n_rows > surprise_compared:
                surprise_match = True
                surprise_compared = n_rows
        matches["FinBERT_Sentiment_Surprise_20"] &= surprise_match
        compared["FinBERT_Sentiment_Surprise_20"] += surprise_compared

        three_day_match = False
        three_day_compared = 0
        for adjust in (False, True):
            expected = balance.ewm(span=3, adjust=adjust, min_periods=1).mean()
            ok, n_rows = _allclose_on_overlap(
                asset_frame["FinBERT_Sentiment_3D"], expected
            )
            if ok and n_rows > three_day_compared:
                three_day_match = True
                three_day_compared = n_rows
        matches["FinBERT_Sentiment_3D"] &= three_day_match
        compared["FinBERT_Sentiment_3D"] += three_day_compared

    passed = bool(matches and all(matches.values()))
    evidence = "; ".join(
        f"{feature}: {'matched' if matches[feature] else 'not matched'} "
        f"({compared[feature]} rows)"
        for feature in matches
    )
    return passed, evidence


def _audit_dates_are_causal(
    daily_news: pd.DataFrame,
    causality_audit: pd.DataFrame | None,
) -> tuple[bool | None, str]:
    """Check optional availability/history-end audit fields against feature dates."""
    frames = [frame for frame in (daily_news, causality_audit) if frame is not None]
    aliases = {
        "feature_as_of",
        "rolling_history_end",
        "attention_history_end",
        "sentiment_history_end",
        "source_max_session",
        "max_source_session",
        "news_available_date",
    }
    comparisons = 0
    violations = 0
    columns_checked: list[str] = []
    for frame in frames:
        if "Date" not in frame.columns:
            continue
        feature_dates = pd.to_datetime(frame["Date"], errors="coerce")
        for column in frame.columns:
            normalised = column.lower().replace(" ", "_")
            if normalised not in aliases:
                continue
            audit_dates = pd.to_datetime(frame[column], errors="coerce")
            mask = feature_dates.notna() & audit_dates.notna()
            comparisons += int(mask.sum())
            violations += int((audit_dates.loc[mask] > feature_dates.loc[mask]).sum())
            columns_checked.append(column)
    if not columns_checked:
        return None, "No optional history-end/availability audit date fields supplied"
    return (
        violations == 0 and comparisons > 0,
        (
            f"Audit columns: {sorted(set(columns_checked))}; "
            f"comparisons: {comparisons}; future-date violations: {violations}"
        ),
    )


def validate_sentiment_daily_news(
    daily_news: pd.DataFrame,
    causality_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate the v7 daily attention and FinBERT feature panel."""
    required_keys = {"Asset", "Date"}
    keys_present = required_keys.issubset(daily_news.columns)
    duplicate_count = (
        int(daily_news.duplicated(["Asset", "Date"]).sum()) if keys_present else -1
    )

    required_features = [*NEWS_ATTENTION_FEATURES, *NEWS_SENTIMENT_FEATURES]
    missing_features = [
        feature for feature in required_features if feature not in daily_news.columns
    ]
    missing_cells = (
        int(daily_news[required_features].isna().sum().sum())
        if not missing_features
        else -1
    )

    probability_columns = [
        "FinBERT_Positive_Share",
        "FinBERT_Negative_Share",
        "FinBERT_Neutral_Share",
    ]
    probabilities_present = all(
        column in daily_news.columns for column in probability_columns
    )
    probability_passed = False
    probability_evidence = "Probability/share columns are missing"
    if probabilities_present and "No_News" in daily_news.columns:
        probabilities = daily_news[probability_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        non_missing = probabilities.notna()
        bounded = bool(
            (
                probabilities.where(non_missing).ge(0)
                & probabilities.where(non_missing).le(1)
            )
            .where(non_missing, True)
            .all()
            .all()
        )
        news_mask = daily_news["No_News"].eq(0) & probabilities.notna().all(axis=1)
        news_sums = probabilities.loc[news_mask].sum(axis=1)
        sums_equal_one = bool(
            news_mask.any()
            and np.allclose(news_sums.to_numpy(), 1.0, rtol=0, atol=1e-6)
        )
        probability_passed = bounded and sums_equal_one
        max_sum_error = (
            float((news_sums - 1.0).abs().max()) if not news_sums.empty else np.nan
        )
        probability_evidence = (
            f"Bounded: {bounded}; scored news rows: {int(news_mask.sum())}; "
            f"maximum sum error: {max_sum_error}"
        )

    no_news_passed = False
    no_news_evidence = "No_News or neutral share is missing"
    if "No_News" in daily_news.columns and probabilities_present:
        no_news = pd.to_numeric(daily_news["No_News"], errors="coerce")
        neutral = pd.to_numeric(
            daily_news["FinBERT_Neutral_Share"], errors="coerce"
        )
        probabilities = daily_news[probability_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        binary = bool(no_news.dropna().isin([0, 1]).all() and no_news.notna().all())
        no_news_mask = no_news.eq(1)
        no_news_probabilities_zero = bool(
            no_news_mask.any()
            and probabilities.loc[no_news_mask].fillna(0).eq(0).all().all()
        )
        no_news_not_neutral = bool(
            no_news_mask.any()
            and no_news_probabilities_zero
            and not no_news.equals(neutral)
        )
        count_consistent = True
        if "Unique_Story_Count" in daily_news.columns:
            story_count = pd.to_numeric(
                daily_news["Unique_Story_Count"], errors="coerce"
            )
            count_consistent = bool(
                story_count.notna().all()
                and no_news.eq(1).equals(story_count.eq(0))
            )
        no_news_passed = binary and no_news_not_neutral and count_consistent
        no_news_evidence = (
            f"Binary: {binary}; no-news rows: {int(no_news_mask.sum())}; "
            f"neutral=1 on no-news rows: "
            f"{int(neutral.loc[no_news_mask].eq(1).sum())}; "
            f"all no-news shares zero: {no_news_probabilities_zero}; "
            f"story-count consistency: {count_consistent}"
        )

    numeric_causal, numeric_evidence = _rolling_news_features_are_causal(daily_news)
    audit_causal, audit_evidence = _audit_dates_are_causal(
        daily_news, causality_audit
    )
    causal_passed = numeric_causal and audit_causal is not False
    causal_evidence = f"{numeric_evidence}; {audit_evidence}"

    return pd.DataFrame(
        [
            _row(
                "N01",
                "V7 Daily News",
                "Daily news rows are unique by Asset and Date",
                keys_present and duplicate_count == 0,
                f"Keys present: {keys_present}; duplicate rows: {duplicate_count}",
            ),
            _row(
                "N02",
                "V7 Daily News",
                "Attention and FinBERT sentiment columns are complete",
                not missing_features and missing_cells == 0,
                (
                    f"Missing columns: {missing_features}; "
                    f"missing feature cells: {missing_cells}"
                ),
            ),
            _row(
                "N03",
                "V7 Daily News",
                "FinBERT daily shares are bounded and sum to one on news days",
                probability_passed,
                probability_evidence,
            ),
            _row(
                "N04",
                "V7 Daily News",
                "No_News is a separate absence flag, not a neutral label",
                no_news_passed,
                no_news_evidence,
            ),
            _row(
                "N05",
                "V7 Daily News",
                "Rolling attention and sentiment features use causal history",
                causal_passed,
                causal_evidence,
            ),
        ]
    )


def validate_sentiment_oof(
    oof_predictions: pd.DataFrame,
    fold_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the v7 paired OOF ledger and fold-specific target thresholds."""
    oof_keys = ["Asset", "Target", "Feature Set", "Model", "Date"]
    oof_required = [*oof_keys, "Fold", "Train End"]
    missing_oof = [
        column for column in oof_required if column not in oof_predictions.columns
    ]
    duplicate_oof = (
        int(oof_predictions.duplicated(oof_keys).sum()) if not missing_oof else -1
    )

    chronological = False
    chronological_evidence = "OOF date fields are missing"
    if not missing_oof:
        dates = pd.to_datetime(oof_predictions["Date"], errors="coerce")
        train_end = pd.to_datetime(oof_predictions["Train End"], errors="coerce")
        valid_dates = dates.notna() & train_end.notna()
        violations = int((train_end.loc[valid_dates] >= dates.loc[valid_dates]).sum())
        chronological = bool(valid_dates.all() and violations == 0)
        chronological_evidence = (
            f"Rows checked: {int(valid_dates.sum())}; "
            f"Train End >= OOF Date violations: {violations}"
        )

    threshold_required = [
        "Asset",
        "Target",
        "Fold",
        "Train End",
        "Purged Anchor Date",
        "Test Start",
        "Test End",
        "Large-Move Threshold",
    ]
    missing_threshold = [
        column for column in threshold_required if column not in fold_thresholds.columns
    ]
    threshold_passed = False
    threshold_evidence = f"Missing threshold columns: {missing_threshold}"
    if not missing_threshold:
        threshold_frame = fold_thresholds.copy()
        for column in [
            "Train End",
            "Purged Anchor Date",
            "Test Start",
            "Test End",
        ]:
            threshold_frame[column] = pd.to_datetime(
                threshold_frame[column], errors="coerce"
            )
        large_move = threshold_frame["Target"].eq("large_move")
        threshold_values = pd.to_numeric(
            threshold_frame.loc[large_move, "Large-Move Threshold"],
            errors="coerce",
        )
        thresholds_valid = bool(
            large_move.any()
            and threshold_values.notna().all()
            and np.isfinite(threshold_values).all()
            and threshold_values.ge(0).all()
        )
        ordered = (
            (threshold_frame["Train End"] < threshold_frame["Purged Anchor Date"])
            & (
                threshold_frame["Purged Anchor Date"]
                < threshold_frame["Test Start"]
            )
            & (threshold_frame["Test Start"] <= threshold_frame["Test End"])
        )
        unique_folds = not threshold_frame.duplicated(
            ["Asset", "Target", "Fold"]
        ).any()
        oof_thresholds_match = True
        if "Large-Move Threshold" in oof_predictions.columns:
            large_oof = oof_predictions.loc[
                oof_predictions["Target"].eq("large_move"),
                ["Asset", "Target", "Fold", "Large-Move Threshold"],
            ].copy()
            expected = threshold_frame.loc[
                large_move,
                ["Asset", "Target", "Fold", "Large-Move Threshold"],
            ]
            merged = large_oof.merge(
                expected,
                on=["Asset", "Target", "Fold"],
                how="left",
                suffixes=("_OOF", "_Ledger"),
                validate="many_to_one",
            )
            oof_thresholds_match = bool(
                len(merged) == len(large_oof)
                and merged["Large-Move Threshold_Ledger"].notna().all()
                and np.allclose(
                    merged["Large-Move Threshold_OOF"],
                    merged["Large-Move Threshold_Ledger"],
                    rtol=0,
                    atol=1e-12,
                )
            )
        threshold_passed = (
            thresholds_valid
            and bool(ordered.all())
            and unique_folds
            and oof_thresholds_match
        )
        threshold_evidence = (
            f"Large-move folds: {int(large_move.sum())}; "
            f"non-empty finite thresholds: {thresholds_valid}; "
            f"purge-order violations: {int((~ordered).sum())}; "
            f"unique fold rows: {unique_folds}; "
            f"OOF thresholds match ledger: {oof_thresholds_match}"
        )

    paired_passed = False
    paired_evidence = "OOF experiment columns are missing"
    if not missing_oof:
        expected_sets = set(SENTIMENT_FEATURE_SETS)
        unpaired_groups = 0
        group_count = 0
        for _, group in oof_predictions.groupby(
            ["Asset", "Target", "Model"], sort=False
        ):
            group_count += 1
            actual_sets = set(group["Feature Set"].astype(str))
            if actual_sets != expected_sets:
                unpaired_groups += 1
                continue
            signatures = {
                feature_set: tuple(
                    sorted(
                        zip(
                            values["Fold"].astype(str),
                            pd.to_datetime(values["Date"]).astype(str),
                        )
                    )
                )
                for feature_set, values in group.groupby("Feature Set")
            }
            if len(set(signatures.values())) != 1:
                unpaired_groups += 1
        paired_passed = group_count > 0 and unpaired_groups == 0
        paired_evidence = (
            f"Asset-target-model groups: {group_count}; "
            f"groups missing exact F0-F5 date pairing: {unpaired_groups}"
        )

    return pd.DataFrame(
        [
            _row(
                "O01",
                "V7 OOF",
                "OOF rows are unique by experiment and Date",
                not missing_oof and duplicate_oof == 0,
                f"Missing columns: {missing_oof}; duplicate rows: {duplicate_oof}",
            ),
            _row(
                "O02",
                "V7 OOF",
                "Every OOF prediction follows its training period",
                chronological,
                chronological_evidence,
            ),
            _row(
                "O03",
                "V7 OOF",
                "Large-move thresholds are fold-fitted with a valid purge gap",
                threshold_passed,
                threshold_evidence,
            ),
            _row(
                "O04",
                "V7 OOF",
                "F0-F5 use completely paired OOF dates",
                paired_passed,
                paired_evidence,
            ),
        ]
    )


def validate_sentiment_study(
    daily_news: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    fold_thresholds: pd.DataFrame,
    causality_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one internal-validation ledger for the complete v7 sub-study."""
    return combine_validation_reports(
        validate_sentiment_daily_news(daily_news, causality_audit),
        validate_sentiment_oof(oof_predictions, fold_thresholds),
    )


def validate_oof_robustness_evidence(
    oof_predictions: pd.DataFrame,
    nested_tuning: pd.DataFrame,
    lstm_seed_results: pd.DataFrame,
    lstm_seed_predictions: pd.DataFrame,
    classification_uncertainty: pd.DataFrame,
    return_separation_uncertainty: pd.DataFrame,
    strategy_benchmark_uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    """Validate nested tuning, seed ensembling and main-study inference tables."""

    tuning_groups = (
        nested_tuning.groupby(["Model", "Outer Fold"], dropna=False)
        if not nested_tuning.empty
        and {"Model", "Outer Fold"}.issubset(nested_tuning.columns)
        else None
    )
    tuning_coverage = bool(
        tuning_groups is not None
        and {"Outer Train End", "Outer Test Start"}.issubset(
            nested_tuning.columns
        )
        and set(nested_tuning["Model"]) == TABULAR_TUNED_MODELS
        and nested_tuning["Outer Fold"].nunique() == 5
        and len(nested_tuning) == 60
        and tuning_groups.size().eq(4).all()
        and tuning_groups["Selected"].sum().eq(1).all()
    )
    tuning_protocol = bool(
        tuning_coverage
        and nested_tuning["Inner Splits"].eq(3).all()
        and nested_tuning["Purge Gap"].eq(TARGET_PURGE_ROWS).all()
        and nested_tuning["Selection Metric"].eq("Macro F1").all()
        and (
            pd.to_datetime(nested_tuning["Outer Train End"], errors="coerce")
            < pd.to_datetime(nested_tuning["Outer Test Start"], errors="coerce")
        ).all()
    )

    seed_coverage = bool(
        not lstm_seed_results.empty
        and {"Model", "Fold", "Seed"}.issubset(lstm_seed_results.columns)
        and set(lstm_seed_results["Seed"].astype(int)) == {42, 43, 44}
        and lstm_seed_results["Fold"].nunique() == 5
        and len(lstm_seed_results) == 15
        and lstm_seed_results.groupby("Fold")["Seed"].nunique().eq(3).all()
    )
    ensemble_identity = False
    ensemble_evidence = "Required seed or ensemble columns are missing."
    required_seed_prediction_columns = {
        "Date",
        "Fold",
        "Seed",
        "Predicted Probability",
    }
    if (
        seed_coverage
        and not lstm_seed_predictions.empty
        and required_seed_prediction_columns.issubset(
            lstm_seed_predictions.columns
        )
        and {
            "Model",
            "Date",
            "Predicted Probability",
            "Probability Threshold",
            "Signal",
        }.issubset(oof_predictions.columns)
    ):
        seed_frame = lstm_seed_predictions.copy()
        seed_frame["Date"] = pd.to_datetime(seed_frame["Date"], errors="coerce")
        seed_mean = (
            seed_frame.groupby("Date", as_index=False)["Predicted Probability"]
            .mean()
            .rename(columns={"Predicted Probability": "Seed Mean Probability"})
        )
        ensemble = oof_predictions.loc[
            oof_predictions["Model"].eq("LSTM Classifier"),
            ["Date", "Predicted Probability"],
        ].copy()
        ensemble["Date"] = pd.to_datetime(ensemble["Date"], errors="coerce")
        comparison = ensemble.merge(seed_mean, on="Date", how="inner")
        ensemble_identity = bool(
            len(comparison) == len(ensemble)
            and len(seed_frame) == len(ensemble) * 3
            and np.allclose(
                comparison["Predicted Probability"],
                comparison["Seed Mean Probability"],
                rtol=0.0,
                atol=1e-12,
            )
        )
        threshold = pd.to_numeric(
            oof_predictions.loc[
                oof_predictions["Model"].eq("LSTM Classifier"),
                "Probability Threshold",
            ],
            errors="coerce",
        )
        signal = pd.to_numeric(
            oof_predictions.loc[
                oof_predictions["Model"].eq("LSTM Classifier"),
                "Signal",
            ],
            errors="coerce",
        )
        ensemble_identity &= bool(
            len(threshold) == len(ensemble)
            and threshold.notna().all()
            and signal.notna().all()
            and np.array_equal(
                signal.astype(int).to_numpy(),
                (
                    pd.to_numeric(
                        ensemble["Predicted Probability"],
                        errors="coerce",
                    ).to_numpy()
                    >= threshold.to_numpy()
                ).astype(int),
            )
        )
        max_error = (
            float(
                np.max(
                    np.abs(
                        comparison["Predicted Probability"]
                        - comparison["Seed Mean Probability"]
                    )
                )
            )
            if not comparison.empty
            else np.nan
        )
        ensemble_evidence = (
            f"ensemble rows={len(ensemble)}; seed rows={len(seed_frame)}; "
            f"maximum mean-probability error={max_error:.3e}"
        )

    expected_models = set(MODEL_LEVEL_NAMES)
    if "Analysis" in classification_uncertainty.columns:
        absolute = classification_uncertainty.loc[
            classification_uncertainty["Analysis"].eq("Absolute OOF metric")
        ]
        paired = classification_uncertainty.loc[
            classification_uncertainty["Analysis"].eq(
                "Paired difference vs Majority"
            )
        ]
    else:
        absolute = pd.DataFrame()
        paired = pd.DataFrame()
    classification_coverage = bool(
        not classification_uncertainty.empty
        and {
            "Analysis",
            "Model",
            "Metric",
            "CI Lower",
            "CI Upper",
        }.issubset(classification_uncertainty.columns)
        and set(classification_uncertainty["Model"]) == expected_models
        and len(classification_uncertainty) == 18
        and len(absolute) == 10
        and len(paired) == 8
        and not absolute.duplicated(["Model", "Metric"]).any()
        and not paired.duplicated(["Model", "Metric"]).any()
        and set(absolute["Metric"]) == {"Accuracy", "Macro F1"}
        and set(paired["Model"]) == expected_models.difference(
            {"Majority Baseline"}
        )
    )
    return_coverage = bool(
        not return_separation_uncertainty.empty
        and set(return_separation_uncertainty["Model"]) == expected_models
        and len(return_separation_uncertainty) == 5
        and not return_separation_uncertainty["Model"].duplicated().any()
    )
    strategy_coverage = bool(
        not strategy_benchmark_uncertainty.empty
        and set(strategy_benchmark_uncertainty["Model"]) == expected_models
        and strategy_benchmark_uncertainty["Benchmark"].nunique() == 8
        and len(strategy_benchmark_uncertainty) == 40
        and not strategy_benchmark_uncertainty.duplicated(
            ["Model", "Benchmark"]
        ).any()
    )

    inference_tables = [
        classification_uncertainty,
        return_separation_uncertainty,
        strategy_benchmark_uncertainty,
    ]
    inference_protocol = True
    protocol_evidence: list[str] = []
    for table in inference_tables:
        required = {
            "CI Lower",
            "CI Upper",
            "Requested Replicates",
            "Valid Replicates",
            "Block Length",
            "Fold Count",
            "Random State",
        }
        if table.empty or not required.issubset(table.columns):
            inference_protocol = False
            protocol_evidence.append("missing required schema")
            continue
        numeric = table[["CI Lower", "CI Upper"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        passed = bool(
            np.isfinite(numeric.to_numpy()).all()
            and (numeric["CI Lower"] <= numeric["CI Upper"]).all()
            and table["Requested Replicates"].eq(2_000).all()
            and table["Valid Replicates"].eq(2_000).all()
            and table["Block Length"].eq(20).all()
            and table["Fold Count"].eq(5).all()
            and table["Random State"].eq(42).all()
        )
        inference_protocol &= passed
        protocol_evidence.append(f"rows={len(table)}; protocol_pass={passed}")

    return pd.DataFrame(
        [
            _row(
                "I01",
                "Main OOF robustness",
                "Three tabular models have four nested candidates in each of five outer folds",
                tuning_coverage,
                f"rows={len(nested_tuning)}; expected=60",
            ),
            _row(
                "I02",
                "Main OOF robustness",
                "Nested tuning uses three inner chronological splits, gap=5 and Macro-F1 only",
                tuning_protocol,
                "Inner tuning is restricted to each outer training period.",
            ),
            _row(
                "I03",
                "Main OOF robustness",
                "LSTM OOF uses seeds 42, 43 and 44 in every outer fold",
                seed_coverage,
                f"seed rows={len(lstm_seed_results)}; expected=15",
            ),
            _row(
                "I04",
                "Main OOF robustness",
                "LSTM ensemble probability equals the arithmetic mean of three seed probabilities",
                ensemble_identity,
                ensemble_evidence,
            ),
            _row(
                "I05",
                "Main OOF inference",
                "Classification uncertainty covers all absolute and Majority-paired comparisons",
                classification_coverage,
                f"rows={len(classification_uncertainty)}; expected=18",
            ),
            _row(
                "I06",
                "Main OOF inference",
                "Gross-return separation uncertainty covers all five model levels",
                return_coverage,
                f"rows={len(return_separation_uncertainty)}; expected=5",
            ),
            _row(
                "I07",
                "Main OOF inference",
                "Net-return uncertainty covers five model levels against eight benchmarks",
                strategy_coverage,
                f"rows={len(strategy_benchmark_uncertainty)}; expected=40",
            ),
            _row(
                "I08",
                "Main OOF inference",
                "All intervals use 2,000 shared fold-aware replicates, 20-day blocks and five folds",
                inference_protocol,
                "; ".join(protocol_evidence),
            ),
        ]
    )


def combine_validation_reports(*reports: pd.DataFrame) -> pd.DataFrame:
    non_empty = [report for report in reports if report is not None and not report.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)
