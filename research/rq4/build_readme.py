"""Build an evidence-linked README from the executed RQ4 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODELS = [
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
]
ASSETS = ["TSLA", "AAPL", "NVDA"]


def percentage(value: float) -> str:
    return f"{value * 100:,.2f}%"


def percentage_points(value: float) -> str:
    """Format a decimal cumulative-return difference in percentage points."""

    return f"{value * 100:,.2f} pp"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    index_label = frame.index.name or "Index"
    columns = [index_label, *[str(column) for column in frame.columns]]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for index, row in frame.iterrows():
        values = [str(index), *[str(value) for value in row.tolist()]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_matrix(frame: pd.DataFrame, value: str) -> str:
    pivot = (
        frame.pivot(index="Asset", columns="Strategy", values=value)
        .reindex(index=ASSETS, columns=MODELS)
        .rename(columns={"Lagged Logistic Regression": "Lagged LR"})
    )
    formatted = pivot.map(percentage)
    formatted.index.name = "Asset"
    return frame_to_markdown(formatted)


def baseline_matrix(frame: pd.DataFrame) -> str:
    strategies = ["EMA8/EMA13 State", "SMA10/SMA20 State", "Buy-and-Hold"]
    pivot = (
        frame.loc[frame["Strategy"].isin(strategies)]
        .pivot(index="Asset", columns="Strategy", values="Net Total Return")
        .reindex(index=ASSETS, columns=strategies)
    )
    formatted = pivot.map(percentage)
    formatted.index.name = "Asset"
    return frame_to_markdown(formatted)


def timing_matrix(frame: pd.DataFrame, period: str) -> str:
    pivot = (
        frame.loc[frame["Period"].eq(period)]
        .pivot(
            index="Asset",
            columns="Model",
            values="Net Timing Value vs Constant-Exposure Passive",
        )
        .reindex(index=ASSETS, columns=MODELS)
        .rename(columns={"Lagged Logistic Regression": "Lagged LR"})
    )
    formatted = pivot.map(percentage_points)
    formatted.index.name = "Asset"
    return frame_to_markdown(formatted)


def best_model_sentences(frame: pd.DataFrame, period: str) -> list[str]:
    rows = frame.loc[frame["Period"].eq(period)]
    sentences = []
    for asset in ASSETS:
        best = (
            rows.loc[rows["Asset"].eq(asset)]
            .sort_values("Net Total Return", ascending=False)
            .iloc[0]
        )
        sentences.append(
            f"- {asset}: {best['Strategy']} recorded the highest observed "
            f"model return at {percentage(best['Net Total Return'])}."
        )
    return sentences


def main() -> None:
    metrics = pd.read_csv(OUTPUTS / "rq4_headline_metrics.csv")
    matched = pd.read_csv(OUTPUTS / "rq4_exposure_matched_buy_hold.csv")
    model_results = pd.read_csv(OUTPUTS / "rq4_model_results.csv")
    metadata = pd.read_json(
        OUTPUTS / "rq4_experiment_metadata.json",
        typ="series",
    )
    checks_path = OUTPUTS / "rq4_validation_checks.csv"
    validation_text = "Validation has not yet been executed."
    if checks_path.exists():
        checks = pd.read_csv(checks_path)
        passed = int(checks["Status"].eq("Pass").sum())
        total = len(checks)
        validation_text = f"{passed}/{total} independent validation checks passed."

    development = metrics.loc[metrics["Period"].eq("Development OOF")]
    audit = metrics.loc[metrics["Period"].eq("Retrospective audit")]
    model_development = model_results.loc[
        model_results["Period"].eq("Development OOF")
    ]
    model_audit = model_results.loc[
        model_results["Period"].eq("Retrospective audit")
    ]

    lines = [
        "# RQ4 Five-Session Cross-Asset Downstream Evaluation",
        "",
        "## Purpose",
        "",
        "This experiment evaluates whether the frozen predictions from Lagged "
        "Logistic Regression, Random Forest, XGBoost and Compact LSTM can be "
        "converted into a consistent five-session downstream decision process. "
        "The primary emphasis is the machine-learning comparison across TSLA, "
        "AAPL and NVDA; Buy-and-Hold and two moving-average rules are transparent "
        "reference points.",
        "",
        "The results are descriptive historical backtests. No RQ4 bootstrap, "
        "p-value or statistical-superiority claim is used. No static-versus-"
        "quarterly-rolling-retraining extension is included.",
        "",
        "## Frozen Evaluation Design",
        "",
        "- RQ3 target: `1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`.",
        "- Four models use the same six feature streams, 20-session lookback, "
        "0.50 threshold and nested temporal protocol.",
        "- TSLA reuses the frozen v8.3 RQ3 predictions; AAPL and NVDA are "
        "asset-specific replications under the same protocol.",
        "- Each signal observed after `Close[t]` is executed at `Open[t+1]`.",
        "- Five independent self-financing sleeves each start with 20% of "
        "capital and retain a decision through `Open[t+6]`.",
        "- Continuous `1→1` states roll without an artificial exit and re-entry.",
        "- One-way friction is 10 bps transaction cost plus 5 bps slippage.",
        "- Average exposure is NAV-weighted across the five sleeves.",
        "- Static exposure-matched Buy-and-Hold uses an ex-post initial allocation "
        "whose mean value-weighted exposure equals the corresponding model path.",
        "- A separate constant-exposure passive path matches the model's average "
        "exposure each interval and provides the main descriptive Timing Value.",
        "",
        "The five-session implementation aligns the decision horizon with RQ3, "
        "but it is not mathematically identical to the classification target "
        "because the target uses the mean of five future closing prices.",
        "",
        "## Development OOF Results",
        "",
        "### Predictive-model net total return",
        "",
        markdown_matrix(model_development, "Net Total Return"),
        "",
        "### Transparent baseline net total return",
        "",
        baseline_matrix(development),
        "",
        "### Net timing value versus constant-exposure passive reference (pp)",
        "",
        timing_matrix(matched, "Development OOF"),
        "",
        "Observed model leaders in the Development OOF period:",
        "",
        *best_model_sentences(model_development, "Development OOF"),
        "",
        "These are historical rankings only. They did not change any RQ3 model, "
        "threshold, feature or hyperparameter choice.",
        "",
        "## Later-Period Results",
        "",
        "The later period is retained as a concise temporal-stability check. It "
        "is a retrospective audit, not an untouched external validation.",
        "",
        "### Predictive-model net total return",
        "",
        markdown_matrix(model_audit, "Net Total Return"),
        "",
        "### Transparent baseline net total return",
        "",
        baseline_matrix(audit),
        "",
        "### Net timing value versus constant-exposure passive reference (pp)",
        "",
        timing_matrix(matched, "Retrospective audit"),
        "",
        "Observed model leaders in the later period:",
        "",
        *best_model_sentences(model_audit, "Retrospective audit"),
        "",
        "The raw later-period ledger contains 407 predictions per asset-model. "
        "The final 2026-07-15 origin is excluded from every economic strategy "
        "because the price snapshot ends on 2026-07-22 and therefore lacks "
        "`Open[t+6]`; 406 complete signals remain.",
        "",
        "## Validation Status",
        "",
        validation_text,
        "",
        f"- Raw model prediction rows: {int(metadata['raw_model_prediction_rows']):,}.",
        f"- Asset-level signal origins: {int(metadata['asset_signal_rows']):,}.",
        "- Development economic signals: 1,250 per asset.",
        "- Later-period economic signals: 406 per asset.",
        "- Quarterly rolling retraining used: no.",
        "- RQ4 bootstrap used: no.",
        "",
        "## Main Files",
        "",
        "- `notebooks/v8_4_rq4_five_session_cross_asset.ipynb`: executed "
        "reader-facing analysis.",
        "- `outputs/rq4_model_results.csv`: four-model return, risk, exposure and "
        "timing-value results.",
        "- `outputs/rq4_headline_metrics.csv`: four models, two moving-average "
        "baselines and Full Buy-and-Hold.",
        "- `outputs/rq4_exposure_matched_buy_hold.csv`: 24 model-specific "
        "static Buy-and-Hold matches plus constant-exposure passive references.",
        "- `outputs/rq4_signal_ledger.csv`: signal-to-execution and terminal-date "
        "mapping.",
        "- `outputs/rq4_equity_curves.csv`: complete gross/net NAV paths.",
        "- `outputs/rq4_sleeve_trade_ledger.csv`: phase-level state changes and "
        "cost factors.",
        "- `outputs/rq4_validation_checks.csv`: independent validation evidence.",
        "- `protocol.json`: frozen scope, rules and claim boundaries.",
        "",
        "## Interpretation Boundary",
        "",
        "The evaluated assets are three large US technology-related equities and "
        "the inputs are price-return series rather than a complete total-return "
        "investment universe. Results therefore show how the AI pipeline behaved "
        "under one frozen downstream mapping; they do not establish future or "
        "live profitability.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
