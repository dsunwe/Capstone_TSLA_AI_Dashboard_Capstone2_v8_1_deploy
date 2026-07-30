from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    DATA_RAW_DIR,
    DATASET_PATHS,
    DATASET_SOURCE_URLS,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_LOOKBACK,
    DEFAULT_LSTM_UNITS,
    DEFAULT_RAW_DATA_PATH,
    LSTM_MODEL_NAME,
    PRESPECIFIED_OOF_FEATURES,
    TARGET_DIRECTION_COLUMN,
)
from src.context import ContextResult, build_context_frame
from src.data import (
    date_summary,
    generate_demo_data,
    numeric_summary,
    preprocessing_audit_table,
    preprocess_stock_data,
    split_continuity_audit,
)
from src.deployment import live_lstm_notice, sandbox_model_names
from src.features import add_features
from src.literature import (
    citation_links,
    method_rationale_table,
    references_table,
)
from src.metrics import format_metrics_for_display
from src.ml import MLResult, run_classification
from src.research import (
    FeatureSelectionResult,
    prespecified_feature_evidence_from_training,
)
from src.research_contract import load_research_contract
from src.v8_evidence import load_validation_tables, validation_summary
from src.v8_sections import (
    render_cross_asset as render_v8_cross_asset,
    render_rq2 as render_v8_rq2,
    render_rq3 as render_v8_rq3,
    render_rq4_backtesting as render_v8_rq4,
)
from src.v8_sandbox import FiveSessionSandboxResult, run_five_session_sandbox
from src.sandbox import (
    sandbox_error_message,
    validate_cleaned_sandbox_data,
    validate_engineered_sandbox_data,
)
from src.validation import (
    combine_validation_reports,
    validate_context,
    validate_dataset,
    validate_features,
)
from src.visualization import (
    correlation_heatmap,
    line_chart,
    model_line_chart,
    price_moving_average_chart,
    return_distribution,
)


st.set_page_config(
    page_title="Capstone II TSLA-Centred ML Backtesting Dashboard",
    page_icon="📈",
    layout="wide",
)

RESEARCH_CONTRACT = load_research_contract()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_preprocess(raw: pd.DataFrame):
    return preprocess_stock_data(raw)




@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_context(trading_dates: pd.Series) -> ContextResult:
    return build_context_frame(trading_dates)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_features(data: pd.DataFrame, context_data: pd.DataFrame):
    return add_features(data, context_data=context_data)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_feature_selection(
    engineered: pd.DataFrame,
) -> FeatureSelectionResult:
    return prespecified_feature_evidence_from_training(
        engineered,
        feature_names=list(PRESPECIFIED_OOF_FEATURES),
    )


@st.cache_resource(
    show_spinner="Training the selected model...",
    max_entries=2,
    ttl=3600,
)
def cached_ml(
    engineered: pd.DataFrame,
    model_name: str,
    threshold: float,
    n_estimators: int,
    max_depth: int,
    feature_names: tuple[str, ...],
    tune_model: bool,
    lstm_lookback: int,
    lstm_units: int,
    lstm_dropout: float,
    lstm_epochs: int,
    lstm_batch_size: int,
) -> MLResult | None:
    return run_classification(
        engineered,
        model_name=model_name,
        train_size=0.80,
        probability_threshold=threshold,
        n_estimators=n_estimators,
        max_depth=max_depth,
        feature_names=list(feature_names),
        tune=tune_model,
        lstm_lookback=lstm_lookback,
        lstm_units=lstm_units,
        lstm_dropout=lstm_dropout,
        lstm_epochs=lstm_epochs,
        lstm_batch_size=lstm_batch_size,
    )
















def infer_asset_name(filename: str) -> str:
    upper_name = Path(filename).stem.upper()
    for asset in DATASET_PATHS:
        if asset in upper_name:
            return asset
    return "UPLOADED"


def stop_for_pipeline_error(stage: str, error: Exception) -> None:
    """Show an actionable Sandbox error and stop the current rerun cleanly."""

    message = sandbox_error_message(stage, error)
    st.sidebar.error("Sandbox input could not be analysed.")
    st.error(message)
    st.info(
        "Use a daily CSV with Date, Open, High, Low, Close (or Adj Close), "
        "and Volume columns. The canonical feature and temporal plan needs a "
        "sufficiently long, varied price history."
    )
    st.stop()


def load_data_from_sidebar() -> tuple[pd.DataFrame, str, str, dict, str]:
    st.sidebar.header("Data Source")
    uploaded = st.sidebar.file_uploader("Upload compatible OHLCV CSV", type=["csv"])
    local_files = [DATASET_PATHS[asset] for asset in DATASET_PATHS]
    options = ["Use deterministic demo data"] + [path.name for path in local_files]
    default_index = options.index(DEFAULT_RAW_DATA_PATH.name)
    local_choice = st.sidebar.selectbox("Bundled or local CSV", options, index=default_index)
    try:
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_BYTES:
                raise ValueError(
                    "Uploaded CSV exceeds the 10 MB deployment Sandbox limit."
                )
            raw = pd.read_csv(uploaded)
            if len(raw) > MAX_UPLOAD_ROWS:
                raise ValueError(
                    "Uploaded CSV exceeds the 10,000-row deployment Sandbox limit."
                )
            cleaned, profile = cached_preprocess(raw)
            return (
                cleaned,
                uploaded.name,
                infer_asset_name(uploaded.name),
                profile,
                "uploaded",
            )
        if local_choice != "Use deterministic demo data":
            raw = pd.read_csv(DATA_RAW_DIR / local_choice)
            cleaned, profile = cached_preprocess(raw)
            return (
                cleaned,
                local_choice,
                infer_asset_name(local_choice),
                profile,
                "bundled",
            )
        raw = generate_demo_data()
        cleaned, profile = cached_preprocess(raw)
        return (
            cleaned,
            "Deterministic synthetic compatibility data",
            "DEMO",
            profile,
            "demo",
        )
    except Exception as exc:
        stop_for_pipeline_error("CSV loading and OHLCV cleaning", exc)


def metric_row(metrics: dict[str, float]) -> None:
    preferred = ["Accuracy", "Balanced Accuracy", "Macro F1", "ROC-AUC"]
    shown = {key: metrics[key] for key in preferred if key in metrics}
    columns = st.columns(len(shown))
    for column, (label, value) in zip(columns, shown.items()):
        display = "N/A" if pd.isna(value) else f"{value * 100:,.2f}%"
        column.metric(label, display)


def data_fingerprint(
    engineered: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> int:
    columns = ["Date", "Price", *(feature_names or [])]
    identity = engineered[[column for column in columns if column in engineered.columns]]
    return int(pd.util.hash_pandas_object(identity, index=False).sum())






def show_figure(fig) -> None:
    if fig is None:
        st.info("No chart is available for the selected data.")
        return
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_method_literature(
    section: str,
    *,
    expanded: bool = False,
) -> None:
    """Show the literature-to-design boundary for one current section."""

    with st.expander("Literature and design support", expanded=expanded):
        st.dataframe(
            method_rationale_table(section),
            width="stretch",
            hide_index=True,
        )
        links = {
            "rq1": [
                "vom Brocke et al. (2020)",
                "Abbasi et al. (2024)",
                "Cerqueira et al. (2020)",
                "Wu et al. (2020)",
            ],
            "eda": [
                "Gu et al. (2020)",
                "Kumbure et al. (2022)",
                "Yang et al. (2022)",
                "Htun et al. (2023)",
                "Mostafavi & Hooman (2025)",
            ],
        }
        st.markdown(citation_links(links[section]))
        st.caption(
            "Published work motivates the design category; exact windows, asset "
            "choices, target formula, and cost settings remain pre-specified "
            "project decisions evaluated by this study."
        )










def render_data_and_preprocessing(
    data: pd.DataFrame,
    source_name: str,
    asset: str,
    profile: dict,
    context_result: ContextResult,
) -> None:
    st.header("1. Data and Study Protocol")
    st.subheader("RQ1 · Completed Research Dashboard")
    st.success(
        "RQ1 is achieved: this project delivers a working eight-section "
        "Streamlit Dashboard linking data preparation, RQ2 feature evaluation, "
        "RQ3 LR/RF/XGBoost/LSTM comparison, cross-asset replication, and RQ4 "
        "downstream evaluation."
    )
    st.caption(
        "Pages 1–6 display reviewed saved results produced by the project "
        "experiment pipelines. Only Page 7 can start optional exploratory training."
    )
    st.write(
        "The same OHLCV schema and cleaning rules are applied to TSLA, AAPL, "
        "NVDA, and uploaded CSV files. The compact protocol below replaces the "
        "research-framing page that was removed from the frontend."
    )
    protocol = pd.DataFrame(
        [
            [
                "Classification target",
                "1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]",
            ],
            ["Information cutoff", "After Close[t]"],
            ["RQ3 common input", "20 consecutive sessions × 6 technical channels"],
            ["RQ4 execution", "Signal after Close[t] → trade at Open[t+1]"],
            ["RQ4 holding design", "Five 20% sleeves; same sleeve reconsidered at Open[t+6]"],
            ["RQ4 friction", "10 bps transaction cost + 5 bps slippage, one way"],
        ],
        columns=["Protocol element", "Pre-specified rule"],
    )
    st.dataframe(protocol, width="stretch", hide_index=True)
    st.caption(
        "The classification target and the Open[t+1]→Open[t+6] decision return "
        "share a five-session horizon, but they are not mathematically identical."
    )
    cols = st.columns(5)
    cols[0].metric("Asset", asset)
    cols[1].metric("Clean Rows", f"{len(data):,}")
    cols[2].metric("Start", f"{data['Date'].min():%Y-%m-%d}")
    cols[3].metric("End", f"{data['Date'].max():%Y-%m-%d}")
    cols[4].metric("Rows Removed", profile["rows_removed"])
    st.caption(f"Current source: {source_name}")
    if asset in DATASET_SOURCE_URLS:
        st.caption(f"Source page: {DATASET_SOURCE_URLS[asset]}")

    st.subheader("Price Field and Split-Continuity Evidence")
    st.write(
        f"The source field `{profile.get('source_price_field', 'Unknown')}` is normalised "
        "to `Close` and then used as `Price` for returns, indicators and drawdown."
    )
    if profile.get("uses_explicit_adjusted_close", False):
        st.success("This input supplied an explicit adjusted-close field.")
    elif asset in DATASET_PATHS:
        split_audit = split_continuity_audit(data, asset)
        if not split_audit.empty:
            split_display = split_audit.copy()
            for column in ["Observed One-Day Return", "Unadjusted Mechanical Return"]:
                split_display[column] = split_display[column].map(
                    lambda value: "N/A" if pd.isna(value) else f"{value * 100:,.2f}%"
                )
            st.dataframe(split_display, width="stretch", hide_index=True)
        st.info(
            "The bundled Nasdaq file has no separate Adjusted Close column or "
            "corporate-action metadata. The known split dates show no mechanical split "
            "crash, so the series is empirically consistent with split-adjusted price "
            "continuity. This is not evidence of total-return adjustment, dividend "
            "reinvestment, or complete corporate-action coverage."
        )
    else:
        st.warning(
            "This uploaded/demo input does not identify an explicit adjusted-close "
            "field. Verify split and other corporate-action handling before interpreting "
            "long-horizon returns or drawdowns."
        )

    with st.expander("Detailed raw-to-clean audit", expanded=False):
        st.dataframe(preprocessing_audit_table(profile), width="stretch", hide_index=True)
        raw_col, clean_col = st.columns(2)
        with raw_col:
            st.subheader("Raw CSV Preview")
            st.dataframe(profile["raw_preview"], width="stretch", hide_index=True)
        with clean_col:
            st.subheader("Cleaned Preview")
            st.dataframe(profile["cleaned_preview"], width="stretch", hide_index=True)
        st.subheader("Date and Data Types")
        left, right = st.columns(2)
        with left:
            st.dataframe(date_summary(data), width="stretch", hide_index=True)
        with right:
            dtypes = data.dtypes.astype(str).rename("Data Type").reset_index()
            dtypes = dtypes.rename(columns={"index": "Column"})
            st.dataframe(dtypes, width="stretch", hide_index=True)
        missing = data.isna().sum().rename("Missing Values").reset_index()
        st.subheader("Missing Values After Cleaning")
        st.dataframe(
            missing.rename(columns={"index": "Column"}),
            width="stretch",
            hide_index=True,
        )
    with st.expander("External context availability audit", expanded=False):
        st.write(
            "Each context observation is merged backward only after its defined "
            "availability date. A zero violation count means no row received a "
            "context value dated after that trading day."
        )
        st.dataframe(context_result.audit, width="stretch", hide_index=True)
        st.dataframe(context_result.manifest, width="stretch", hide_index=True)
    render_method_literature("rq1")


def render_eda(engineered: pd.DataFrame, asset: str) -> None:
    st.header("2. Exploratory Data Analysis")
    st.write(
        f"EDA for {asset} checks price behaviour, returns, volatility, and the "
        "six pre-specified technical inputs used downstream."
    )
    summary_columns = [
        column
        for column in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            *PRESPECIFIED_OOF_FEATURES,
            TARGET_DIRECTION_COLUMN,
        ]
        if column in engineered.columns
    ]
    with st.expander("Numeric summary", expanded=False):
        st.dataframe(
            numeric_summary(engineered[summary_columns]),
            width="stretch",
        )
    show_figure(
        price_moving_average_chart(
            engineered,
            title=f"{asset} Price with MA10 and MA20",
        )
    )
    st.caption(
        "The Price + MA10/MA20 chart is retained from Capstone I as a direct "
        "continuity feature. The v8 comparison uses EMA8/13 and SMA10/20 "
        "states as the two transparent moving-average baselines."
    )
    show_figure(return_distribution(engineered))
    show_figure(
        line_chart(
            engineered,
            x="Date",
            y_columns=["Volatility_20"],
            title=f"{asset} 20-Day Rolling Volatility",
            y_label="Volatility",
        )
    )
    st.subheader("Selected-Input Correlation")
    heatmap_columns = [
        column
        for column in [*PRESPECIFIED_OOF_FEATURES, TARGET_DIRECTION_COLUMN]
        if column in engineered.columns
    ]
    show_figure(correlation_heatmap(engineered, heatmap_columns))
    st.caption(
        "The heatmap is limited to the six revised RQ2 inputs and the five-session "
        "classification label rather than the larger exploratory candidate pool."
    )
    render_method_literature("eda")


def rq2_technical_feature_specification() -> pd.DataFrame:
    """Return the protocol-ordered six-channel RQ2 Technical specification."""

    roles = {
        "Daily_Return": "One-session price change",
        "Momentum_5": "Five-session price momentum",
        "EMA8_13_State": "Persistent state: 1 when EMA8(t) > EMA13(t)",
        "Volatility_20": "Twenty-session realised volatility",
        "Drawdown": "Current decline from the running price peak",
        "Volume_Ratio": "Current volume relative to its recent level",
    }
    return pd.DataFrame(
        {
            "Feature": list(PRESPECIFIED_OOF_FEATURES),
            "RQ2 Feature Family": "Technical — EMA-aware compact",
            "Input Role": [
                roles[feature] for feature in PRESPECIFIED_OOF_FEATURES
            ],
            "Downstream Status": "Used by RQ3, RQ4 and the Sandbox",
        }
    )


















def _format_five_session_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Format exploratory five-session results without changing stored values."""

    display = metrics.copy()
    for column in [
        "Gross Total Return",
        "Net Total Return",
        "Annualized Net Return",
        "Maximum Drawdown",
        "Average Exposure",
    ]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: (
                    "N/A"
                    if pd.isna(value)
                    else f"{float(value) * 100:,.2f}%"
                )
            )
    if "Cost Drag (pp)" in display.columns:
        display["Cost Drag (pp)"] = display["Cost Drag (pp)"].map(
            lambda value: (
                "N/A" if pd.isna(value) else f"{float(value):,.2f} pp"
            )
        )
    for column in ["Path Start", "Path End"]:
        if column in display.columns:
            display[column] = pd.to_datetime(
                display[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    return display


def _active_five_session_sandbox_result(
    engineered: pd.DataFrame,
    ml_result: MLResult | None,
    settings: dict,
) -> FiveSessionSandboxResult:
    """Convert the current exploratory Sandbox output into the v8 decision layer."""

    if ml_result is None or ml_result.results.empty:
        raise ValueError(
            "The active Sandbox model has no prediction rows for backtesting."
        )
    prediction_columns = [
        column
        for column in ["Date", "Predicted Probability", "Signal"]
        if column in ml_result.results.columns
    ]
    predictions = ml_result.results[prediction_columns].copy()
    return run_five_session_sandbox(
        engineered[["Date", "Open", "Close"]].copy(),
        predictions,
        prediction_date_column="Date",
        probability_column=(
            "Predicted Probability"
            if "Predicted Probability" in predictions.columns
            else None
        ),
        signal_column="Signal" if "Signal" in predictions.columns else None,
        probability_threshold=settings["threshold"],
        transaction_cost_bps=settings["transaction_cost_bps"],
        slippage_bps=settings["slippage_bps"],
        prediction_strategy_name=ml_result.model_name,
    )


def _five_session_equity_chart(result: FiveSessionSandboxResult, title: str):
    chart = result.equity_rows[
        ["Open Date", "Strategy", "Net NAV"]
    ].rename(
        columns={
            "Open Date": "Date",
            "Strategy": "Model",
            "Net NAV": "Equity",
        }
    )
    return model_line_chart(chart, "Equity", title)


def render_sandbox_five_session_backtest(
    engineered: pd.DataFrame,
    ml_result: MLResult | None,
    settings: dict,
    asset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the optional user-data Sandbox without changing the reviewed results."""

    st.header("7. Analyze Your Data")
    st.subheader("Exploratory Five-Session Sandbox")
    st.write(
        "The current sidebar model is converted into a long-or-cash signal after "
        "Close[t]. Execution occurs at Open[t+1]. Five equal, self-financing 20% "
        "sleeves allow a new decision each session, and each sleeve is reconsidered "
        "at its corresponding Open[t+6]. Costs are charged only on actual 0→1 or "
        "1→0 changes; a continuous 1→1 position is not charged again."
    )
    st.info(
        "This page is an exploratory reconstruction using the active CSV, model, "
        "threshold and cost controls. It is separate from the reviewed RQ3 and RQ4 "
        "results on Pages 4–6."
    )
    st.caption(
        "The exploratory model uses the same six feature roles explained with "
        "literature support on Page 3; Sandbox outputs never replace formal results."
    )
    if ml_result is not None:
        metric_row(ml_result.metrics)
        with st.expander("Current-model classification diagnostics", expanded=False):
            st.write(
                f"**Model:** {ml_result.model_name} · "
                f"**Training rows:** {ml_result.train_rows:,} · "
                f"**Held-out rows:** {ml_result.test_rows:,}"
            )
            st.dataframe(
                format_metrics_for_display(ml_result.baseline_comparison),
                width="stretch",
                hide_index=True,
            )
            st.dataframe(
                ml_result.feature_importance,
                width="stretch",
                hide_index=True,
            )
    try:
        result = _active_five_session_sandbox_result(
            engineered,
            ml_result,
            settings,
        )
    except Exception as exc:
        st.error(sandbox_error_message("five-session backtesting", exc))
        return pd.DataFrame(), pd.DataFrame()

    prediction_name = ml_result.model_name if ml_result is not None else ""
    selected = result.metrics.loc[
        result.metrics["Strategy"].eq(prediction_name)
    ]
    if not selected.empty:
        row = selected.iloc[0]
        cards = st.columns(4)
        cards[0].metric(
            "Net Return",
            f"{float(row['Net Total Return']) * 100:,.2f}%",
        )
        cards[1].metric(
            "Maximum Drawdown",
            f"{float(row['Maximum Drawdown']) * 100:,.2f}%",
        )
        cards[2].metric(
            "Exposure",
            f"{float(row['Average Exposure']) * 100:,.2f}%",
        )
        cards[3].metric(
            "Cost Drag",
            f"{float(row['Cost Drag (pp)']):,.2f} pp",
        )

    st.subheader("Active Model and Transparent Baselines")
    st.dataframe(
        _format_five_session_metrics(result.metrics),
        width="stretch",
        hide_index=True,
    )
    show_figure(
        _five_session_equity_chart(
            result,
            f"{asset} Exploratory Five-Session Net NAV",
        )
    )

    st.caption(
        f"One-way friction = {settings['transaction_cost_bps']:.1f} bps "
        f"transaction cost + {settings['slippage_bps']:.1f} bps slippage. "
        "The future-mean classification target and Open[t+1]→Open[t+6] return "
        "share a five-session horizon but are not mathematically identical."
    )
    return result.metrics, result.equity_rows












def render_validation(
    validation_report: pd.DataFrame,
    manifest_verification: dict[str, object] | None = None,
) -> None:
    """Show concise research checks without restoring a separate evidence mode."""

    st.header("8. Testing and Limitations")
    saved = validation_summary()
    stage_table = saved["table"]
    stage_lookup = stage_table.set_index("Module")
    cols = st.columns(4)
    for column, module in zip(cols[:3], ["RQ2", "RQ3", "RQ4"]):
        row = stage_lookup.loc[module]
        column.metric(f"{module} checks", f"{int(row['Passed'])}/{int(row['Total'])}")
    cols[3].metric("Combined checks", f"{saved['passed']}/{saved['total']}")
    st.dataframe(stage_table, width="stretch", hide_index=True)

    with st.expander("Detailed saved-result checks", expanded=False):
        detailed = []
        for module, frame in load_validation_tables().items():
            table = frame.copy()
            table.insert(0, "Module", module)
            detailed.append(table)
        st.dataframe(
            pd.concat(detailed, ignore_index=True),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Current Input Checks")
    current_total = len(validation_report)
    current_passed = int(validation_report["Status"].eq("Pass").sum())
    current_cols = st.columns(2)
    current_cols[0].metric(
        "Current-input checks",
        f"{current_passed}/{current_total}",
    )
    current_cols[1].metric(
        "Current-input pass rate",
        f"{current_passed / max(current_total, 1) * 100:.1f}%",
    )
    st.caption(
        "Current-input checks cover the bundled or uploaded data and the six "
        "pre-specified features. They do not alter the reviewed RQ2–RQ4 results."
    )
    st.dataframe(validation_report, width="stretch", hide_index=True)

    with st.expander("Interpretation boundaries", expanded=False):
        for boundary in RESEARCH_CONTRACT.get("claim_boundaries", []):
            st.markdown(f"- {boundary}")

    with st.expander("Literature traceability and complete references", expanded=False):
        coverage = []
        for section, label in [
            ("rq1", "RQ1 / protocol"),
            ("eda", "EDA"),
            ("rq2", "RQ2"),
            ("rq3", "RQ3"),
            ("cross_asset", "Cross-asset"),
            ("rq4", "RQ4"),
            ("validation", "Validation"),
        ]:
            table = method_rationale_table(section).copy()
            table.insert(0, "Dashboard part", label)
            coverage.append(table)
        st.dataframe(
            pd.concat(coverage, ignore_index=True),
            width="stretch",
            hide_index=True,
        )
        st.dataframe(references_table(), width="stretch", hide_index=True)
        st.markdown(
            citation_links(
                [
                    "vom Brocke et al. (2020)",
                    "Cerqueira et al. (2020)",
                    "Peng et al. (2021)",
                    "Kumbure et al. (2022)",
                    "Htun et al. (2023)",
                    "Mostafavi & Hooman (2025)",
                    "Yun et al. (2021)",
                    "Vuong et al. (2022)",
                    "Al-Selwi et al. (2024)",
                    "Patton & Weller (2020)",
                ]
            )
        )
        st.caption(
            "The reference map supports research design and traceability. It does "
            "not turn internal consistency checks into predictive-validity claims."
        )






NAVIGATION_SECTIONS = (
    "1. Data and Study Protocol",
    "2. Exploratory Data Analysis",
    "3. RQ2 Feature Families",
    "4. RQ3 Model Comparison",
    "5. Cross-Asset Replication",
    "6. RQ4 Downstream Evaluation",
    "7. Analyze Your Data",
    "8. Testing and Limitations",
)


def sandbox_settings_from_sidebar() -> dict:
    """Render controls only on the explicitly exploratory Sandbox page."""

    st.sidebar.header("Exploratory Sandbox")
    available_models = sandbox_model_names()
    live_lstm_available = LSTM_MODEL_NAME in available_models
    model_name = st.sidebar.selectbox(
        "Exploratory classification model",
        available_models,
        index=0,
    )
    capability_notice = live_lstm_notice()
    if capability_notice is not None:
        st.sidebar.info(capability_notice)
    threshold = st.sidebar.slider(
        "Positive probability threshold",
        0.30,
        0.70,
        0.50,
        0.01,
    )
    tune_model = False
    n_estimators = 200
    max_depth = 6
    lstm_lookback = DEFAULT_LSTM_LOOKBACK
    lstm_units = DEFAULT_LSTM_UNITS
    lstm_dropout = DEFAULT_LSTM_DROPOUT
    lstm_epochs = DEFAULT_LSTM_EPOCHS
    lstm_batch_size = DEFAULT_LSTM_BATCH_SIZE

    with st.sidebar.expander("Advanced model controls", expanded=False):
        if live_lstm_available:
            tune_model = st.toggle(
                "Training-only hyperparameter tuning",
                value=False,
            )
        else:
            st.caption(
                "Cloud-safe profile: live hyperparameter tuning is disabled "
                "to keep the examiner instance responsive."
            )
        if model_name in {"Random Forest Classifier", "XGBoost Classifier"}:
            maximum_estimators = 400 if live_lstm_available else 240
            n_estimators = st.slider(
                "Tree estimators",
                80,
                maximum_estimators,
                200,
                40,
            )
            max_depth = st.slider("Tree max depth", 2, 12, 6, 1)
        if model_name == LSTM_MODEL_NAME:
            lstm_lookback = st.select_slider(
                "Sequence lookback (trading days)",
                options=[10, 20, 40],
                value=DEFAULT_LSTM_LOOKBACK,
            )
            lstm_units = st.select_slider(
                "LSTM units",
                options=[8, 16, 32],
                value=DEFAULT_LSTM_UNITS,
            )
            lstm_dropout = st.slider(
                "LSTM dropout",
                0.0,
                0.4,
                DEFAULT_LSTM_DROPOUT,
                0.1,
            )
            lstm_epochs = st.slider(
                "LSTM maximum epochs",
                10,
                80,
                DEFAULT_LSTM_EPOCHS,
                10,
            )
            lstm_batch_size = st.select_slider(
                "LSTM batch size",
                options=[16, 32, 64],
                value=DEFAULT_LSTM_BATCH_SIZE,
            )

    with st.sidebar.expander("Backtesting assumptions", expanded=True):
        transaction_cost_bps = st.slider(
            "Transaction cost (bps)",
            0.0,
            50.0,
            10.0,
            1.0,
        )
        slippage_bps = st.slider(
            "Slippage (bps)",
            0.0,
            50.0,
            5.0,
            1.0,
        )
    st.sidebar.caption(
        "Sandbox only · six EMA-aware inputs · reviewed RQ3/RQ4 results are unchanged."
    )
    return {
        "model_name": model_name,
        "tune_model": tune_model,
        "threshold": threshold,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "cv_splits": 5,
        "lstm_lookback": lstm_lookback,
        "lstm_units": lstm_units,
        "lstm_dropout": lstm_dropout,
        "lstm_epochs": lstm_epochs,
        "lstm_batch_size": lstm_batch_size,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "top_k": len(PRESPECIFIED_OOF_FEATURES),
    }


def sandbox_model_context(
    engineered: pd.DataFrame,
    settings: dict,
    selected_features: list[str],
) -> tuple:
    """Identify whether a cached interactive result still matches the controls."""

    return (
        len(engineered),
        data_fingerprint(engineered, selected_features),
        settings["model_name"],
        settings["tune_model"],
        settings["threshold"],
        settings["n_estimators"],
        settings["max_depth"],
        settings["lstm_lookback"],
        settings["lstm_units"],
        settings["lstm_dropout"],
        settings["lstm_epochs"],
        settings["lstm_batch_size"],
        tuple(selected_features),
    )


def main() -> None:
    st.title("Capstone II TSLA-Centred ML Backtesting Dashboard")
    st.caption(
        "Five-session target · RQ2 six-feature design · RQ3 common 20×6 model "
        "comparison · three-asset RQ4 decision evaluation."
    )
    st.warning(
        "Educational and research use only. This dashboard is not financial "
        "advice or a live trading system."
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Section",
        NAVIGATION_SECTIONS,
        label_visibility="collapsed",
    )

    if page == "4. RQ3 Model Comparison":
        render_v8_rq3()
        return
    if page == "5. Cross-Asset Replication":
        render_v8_cross_asset()
        return
    if page == "6. RQ4 Downstream Evaluation":
        render_v8_rq4()
        return

    try:
        data, source_name, asset, profile, _source_kind = load_data_from_sidebar()
        validate_cleaned_sandbox_data(data)
        context_result = cached_context(data["Date"])
        engineered = cached_features(data, context_result.frame)
        validate_engineered_sandbox_data(engineered)
        selection = cached_feature_selection(engineered)
        selected_features = list(selection.selected_features)
    except Exception as exc:
        stop_for_pipeline_error("data and six-feature preparation", exc)

    if page == "1. Data and Study Protocol":
        render_data_and_preprocessing(data, source_name, asset, profile, context_result)
        return
    if page == "2. Exploratory Data Analysis":
        render_eda(engineered, asset)
        return
    if page == "3. RQ2 Feature Families":
        render_v8_rq2(selection.table)
        return
    if page == "7. Analyze Your Data":
        settings = sandbox_settings_from_sidebar()
        expected_context = sandbox_model_context(
            engineered,
            settings,
            selected_features,
        )
        if st.sidebar.button("Run Sandbox Model", type="primary"):
            try:
                st.session_state["sandbox_ml_result"] = cached_ml(
                    engineered,
                    settings["model_name"],
                    settings["threshold"],
                    settings["n_estimators"],
                    settings["max_depth"],
                    tuple(selected_features),
                    settings["tune_model"],
                    settings["lstm_lookback"],
                    settings["lstm_units"],
                    settings["lstm_dropout"],
                    settings["lstm_epochs"],
                    settings["lstm_batch_size"],
                )
                st.session_state["sandbox_ml_context"] = expected_context
            except Exception as exc:
                st.error(sandbox_error_message("exploratory model training", exc))
        ml_result = (
            st.session_state.get("sandbox_ml_result")
            if st.session_state.get("sandbox_ml_context") == expected_context
            else None
        )
        if not isinstance(ml_result, MLResult):
            st.header("7. Analyze Your Data")
            st.info(
                "Choose an exploratory model and cost assumptions in the sidebar, "
                "then click **Run Sandbox Model**. Opening Pages 1–6 never trains a model."
            )
            capability_notice = live_lstm_notice()
            if capability_notice is not None:
                st.info(capability_notice)
            st.dataframe(
                rq2_technical_feature_specification(),
                width="stretch",
                hide_index=True,
            )
            return
        render_sandbox_five_session_backtest(
            engineered,
            ml_result,
            settings,
            asset,
        )
        return

    if page == "8. Testing and Limitations":
        validation_report = combine_validation_reports(
            validate_dataset(data),
            validate_context(context_result.audit, engineered),
            validate_features(engineered, selected_features),
        )
        render_validation(validation_report)


if __name__ == "__main__":
    main()
