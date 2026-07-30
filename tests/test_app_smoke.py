from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import PRESPECIFIED_OOF_FEATURES
from src.v8_evidence import ASSETS, MODELS


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

SECTIONS = (
    "1. Data and Study Protocol",
    "2. Exploratory Data Analysis",
    "3. RQ2 Feature Families",
    "4. RQ3 Model Comparison",
    "5. Cross-Asset Replication",
    "6. RQ4 Downstream Evaluation",
    "7. Analyze Your Data",
    "8. Testing and Limitations",
)

EXPECTED_HEADERS = {
    "1. Data and Study Protocol": "1. Data and Study Protocol",
    "2. Exploratory Data Analysis": "2. Exploratory Data Analysis",
    "3. RQ2 Feature Families": "3. RQ2 Feature-Family Comparison",
    "4. RQ3 Model Comparison": "4. RQ3 Common-Information Model Comparison",
    "5. Cross-Asset Replication": "5. Cross-Asset Classification Replication",
    "6. RQ4 Downstream Evaluation": "6. RQ4 Five-Session Downstream Evaluation",
    "7. Analyze Your Data": "7. Analyze Your Data",
    "8. Testing and Limitations": "8. Testing and Limitations",
}


def _section_control(app: AppTest):
    return next(
        radio
        for radio in app.radio
        if tuple(radio.options) == SECTIONS
    )


def _navigate(app: AppTest, section: str) -> AppTest:
    _section_control(app).set_value(section).run()
    assert not app.exception
    return app


def _visible_text(app: AppTest) -> str:
    groups = (
        app.title,
        app.header,
        app.subheader,
        app.caption,
        app.info,
        app.warning,
        app.success,
        app.error,
        app.markdown,
    )
    text = [
        str(element.value)
        for group in groups
        for element in group
        if getattr(element, "value", None) is not None
    ]
    for dataframe in app.dataframe:
        text.extend(dataframe.value.astype(str).to_numpy().ravel())
    return "\n".join(map(str, text))


def _dataframe_with_columns(app: AppTest, required: set[str]):
    return next(
        dataframe.value
        for dataframe in app.dataframe
        if required.issubset(dataframe.value.columns)
    )


def _assert_non_empty(app: AppTest, section: str) -> None:
    assert not app.exception
    assert any(
        header.value == EXPECTED_HEADERS[section]
        for header in app.header
    )
    assert (
        len(app.subheader)
        + len(app.dataframe)
        + len(app.metric)
        + len(app.info)
    ) > 0


def test_dashboard_exposes_the_streamlined_eight_section_workflow():
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()

    assert not app.exception
    assert [title.value for title in app.title] == [
        "Capstone II TSLA-Centred ML Backtesting Dashboard"
    ]
    assert tuple(_section_control(app).options) == SECTIONS
    assert len(_section_control(app).options) == 8
    assert not any(
        selectbox.label == "Evidence mode"
        for selectbox in app.selectbox
    )
    assert not any(
        selectbox.label == "Exploratory classification model"
        for selectbox in app.selectbox
    )
    assert "RQ1 · Completed Research Dashboard" in _visible_text(app)
    assert "RQ1 is achieved" in _visible_text(app)
    assert "Only Page 7 can start optional exploratory training" in _visible_text(app)

    for section in SECTIONS:
        if _section_control(app).value != section:
            _navigate(app, section)
        _assert_non_empty(app, section)
        visible = _visible_text(app)
        assert "Frozen Research Evidence" not in visible
        assert "Supplementary Firm-Associated News Sentiment" not in visible
        assert "Five-Session Single-Window Rule Sensitivity" not in visible

        if section in {
            "1. Data and Study Protocol",
            "3. RQ2 Feature Families",
            "4. RQ3 Model Comparison",
            "5. Cross-Asset Replication",
            "6. RQ4 Downstream Evaluation",
        }:
            assert any(
                expander.label == "Literature and design support"
                for expander in app.expander
            )


def test_reviewed_rq2_rq3_and_rq4_pages_use_saved_results():
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()

    _navigate(app, "3. RQ2 Feature Families")
    six_features = _dataframe_with_columns(
        app,
        {"Feature", "Role", "Training Missing %", "Selected"},
    )
    assert six_features["Feature"].tolist() == list(PRESPECIFIED_OOF_FEATURES)
    assert len(six_features) == 6
    rationale = _dataframe_with_columns(
        app,
        {
            "Feature",
            "Definition",
            "Market state captured",
            "Why included",
            "Literature support",
            "Project-specific boundary",
        },
    )
    assert rationale["Feature"].tolist() == list(PRESPECIFIED_OOF_FEATURES)
    assert rationale["Feature"].is_unique
    assert len(rationale) == 6
    assert "Why These Six Features Were Selected" in _visible_text(app)
    assert "Citations do not prove that any exact window is universally optimal" in _visible_text(app)
    family_results = _dataframe_with_columns(
        app,
        {"Feature Family", "Dev mean Accuracy", "Audit Accuracy"},
    )
    assert len(family_results) == 4
    assert "Feature-Group and Preprocessing Ablation" not in _visible_text(app)
    assert "Examiner-Requested Additive Feature-Family Ablation" not in _visible_text(app)

    _navigate(app, "4. RQ3 Model Comparison")
    rq3_models = _dataframe_with_columns(
        app,
        {"Comparator", "Mean outer-fold BA", "Audit BA"},
    )
    assert set(rq3_models["Comparator"]) == set(MODELS)
    one_se = _dataframe_with_columns(
        app,
        {"Model", "One-SE Eligible", "One-SE primary selection"},
    )
    assert len(one_se) == 4
    assert one_se["One-SE Eligible"].all()
    selected = one_se.loc[one_se["One-SE primary selection"]]
    assert selected["Model"].tolist() == ["Lagged Logistic Regression"]
    assert "Exploratory Current-Model Holdout Check" not in _visible_text(app)
    assert not any(button.label == "Run Walk-Forward Validation" for button in app.button)
    assert not any(
        selectbox.label == "Exploratory classification model"
        for selectbox in app.selectbox
    )

    _navigate(app, "5. Cross-Asset Replication")
    replication = _dataframe_with_columns(
        app,
        {"Asset", "Model", "Dev mean BA", "Audit BA"},
    )
    assert set(replication["Asset"]) == set(ASSETS)
    assert set(replication["Model"]) == set(MODELS)
    assert len(replication) == len(ASSETS) * len(MODELS)
    assert not any(button.label == "Run Cross-Asset Benchmark" for button in app.button)

    _navigate(app, "6. RQ4 Downstream Evaluation")
    rq4_models = _dataframe_with_columns(
        app,
        {
            "Asset",
            "Strategy",
            "Net Total Return",
            "Rank among 4 predictive models",
        },
    )
    assert set(rq4_models["Asset"]) == set(ASSETS)
    assert set(rq4_models["Strategy"]) == set(MODELS)
    assert len(rq4_models) == len(ASSETS) * len(MODELS)
    assert any(selectbox.label == "Evaluation period" for selectbox in app.selectbox)
    assert any(selectbox.label == "Asset" for selectbox in app.selectbox)
    assert any(multiselect.label == "Predictive models" for multiselect in app.multiselect)
    assert not any(button.label == "Run Sandbox Model" for button in app.button)


def test_exploratory_training_is_explicit_and_validation_is_compact():
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()

    _navigate(app, "7. Analyze Your Data")
    assert any(
        selectbox.label == "Exploratory classification model"
        for selectbox in app.selectbox
    )
    assert any(button.label == "Run Sandbox Model" for button in app.button)
    assert "click **Run Sandbox Model**" in _visible_text(app)
    assert not any(
        {"Strategy", "Net Total Return", "Terminal All Cash"}.issubset(
            dataframe.value.columns
        )
        for dataframe in app.dataframe
    )

    _navigate(app, "8. Testing and Limitations")
    metric_values = {metric.label: metric.value for metric in app.metric}
    assert metric_values["RQ2 checks"] == "22/22"
    assert metric_values["RQ3 checks"] == "24/24"
    assert metric_values["RQ4 checks"] == "38/38"
    assert metric_values["Combined checks"] == "84/84"
    assert "Root release integrity" not in metric_values


def test_rq4_period_and_asset_filters_update_saved_results_without_training():
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    _navigate(app, "6. RQ4 Downstream Evaluation")

    period = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Evaluation period"
    )
    assert period.value == "Development OOF"
    default_metrics = {metric.label: metric.value for metric in app.metric}
    assert default_metrics == {
        "Assets in view": "3",
        "Models in view": "4",
        "Holding horizon": "5 sessions",
        "One-way friction": "15 bps",
    }
    period.set_value("Retrospective audit").run()
    assert not app.exception
    visible = _visible_text(app)
    audit_caption = (
        "A retained later-period robustness audit showed weaker stability than "
        "Development and is reported as a study limitation."
    )
    assert visible.count(audit_caption) == 1
    assert "No model–asset combination retained positive Timing Value" not in visible
    assert "only AAPL Lagged LR" not in visible

    asset = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Asset"
    )
    asset.set_value("AAPL").run()
    assert not app.exception
    filtered = _dataframe_with_columns(
        app,
        {
            "Asset",
            "Strategy",
            "Net Total Return",
            "Rank among 4 predictive models",
        },
    )
    assert filtered["Asset"].tolist() == ["AAPL"] * 4
    assert set(filtered["Strategy"]) == set(MODELS)
    filtered_metrics = {metric.label: metric.value for metric in app.metric}
    assert filtered_metrics["Assets in view"] == "1"
    assert filtered_metrics["Models in view"] == "4"

    model_filter = next(
        multiselect
        for multiselect in app.multiselect
        if multiselect.label == "Predictive models"
    )
    model_filter.set_value(["Compact LSTM"]).run()
    assert not app.exception
    one_model_metrics = {metric.label: metric.value for metric in app.metric}
    assert one_model_metrics["Assets in view"] == "1"
    assert one_model_metrics["Models in view"] == "1"
    assert not any(button.label == "Run Sandbox Model" for button in app.button)
