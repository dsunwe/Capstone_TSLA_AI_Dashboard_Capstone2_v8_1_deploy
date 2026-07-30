"""Build the reader-facing RQ4 five-session cross-asset notebook.

The builder consumes only the formal CSV, JSON, and PNG artefacts produced by
``run_rq4_experiment.py``.  It does not import the experiment engine, retrain a
model, regenerate a strategy path, run a bootstrap, or add a rolling-retraining
comparison.

Run this file after the formal RQ4 experiment has completed:

    python build_notebook.py

The generated notebook is intentionally left unexecuted so it can be executed
later with a normal Jupyter kernel from either the experiment directory or its
``notebooks`` subdirectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOKS / "v8_4_rq4_five_session_cross_asset.ipynb"

ASSETS = ["TSLA", "AAPL", "NVDA"]
PERIODS = ["Development OOF", "Retrospective audit"]
MODELS = [
    "Lagged Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Compact LSTM",
]
BASELINES = [
    "EMA8/EMA13 State",
    "SMA10/SMA20 State",
    "Buy-and-Hold",
]
SHORT_NAMES = {
    "Lagged Logistic Regression": "Lagged LR",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "Compact LSTM": "Compact LSTM",
    "EMA8/EMA13 State": "EMA8/13",
    "SMA10/SMA20 State": "SMA10/20",
    "Buy-and-Hold": "Buy-and-Hold",
}

FORMAL_TABLES = {
    "headline": OUTPUTS / "rq4_headline_metrics.csv",
    "models": OUTPUTS / "rq4_model_results.csv",
    "matched": OUTPUTS / "rq4_exposure_matched_buy_hold.csv",
    "signals": OUTPUTS / "rq4_signal_ledger.csv",
}
FORMAL_FIGURES = {
    "development_models": FIGURES / "rq4_development_model_returns.png",
    "development_timing": FIGURES / "rq4_development_timing_value.png",
    "development_all": FIGURES / "rq4_development_strategy_comparison.png",
    "later_models": FIGURES / "rq4_later_period_model_returns.png",
}


def percentage(value: float) -> str:
    """Format a decimal return or exposure as a percentage."""

    return f"{100.0 * float(value):,.2f}%"


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    """Fail early if a formal result table no longer matches its contract."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{source.name} is missing columns: {missing}")


def load_formal_results() -> tuple[
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and minimally validate the frozen report inputs."""

    required_paths = [
        ROOT / "protocol.json",
        OUTPUTS / "rq4_experiment_metadata.json",
        *FORMAL_TABLES.values(),
        *FORMAL_FIGURES.values(),
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Formal RQ4 artefacts are incomplete. Run the experiment before "
            f"building the notebook. Missing: {missing}"
        )

    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (OUTPUTS / "rq4_experiment_metadata.json").read_text(encoding="utf-8")
    )
    headline = pd.read_csv(FORMAL_TABLES["headline"])
    models = pd.read_csv(FORMAL_TABLES["models"])
    matched = pd.read_csv(FORMAL_TABLES["matched"])
    signals = pd.read_csv(FORMAL_TABLES["signals"])

    require_columns(
        headline,
        {
            "Asset",
            "Period",
            "Strategy",
            "Family",
            "Net Total Return",
            "Annualised Net Return",
            "Maximum Drawdown",
            "Average Exposure",
        },
        FORMAL_TABLES["headline"],
    )
    require_columns(
        models,
        {
            "Asset",
            "Period",
            "Strategy",
            "Net Total Return",
            "Annualised Net Return",
            "Maximum Drawdown",
            "Average Exposure",
            "Matched Net Total Return",
            "Net Return Difference vs Static Matched B&H",
            "Constant-Exposure Passive Net Total Return",
            "Net Timing Value vs Constant-Exposure Passive",
            "Return Rank within Asset-Period",
        },
        FORMAL_TABLES["models"],
    )
    require_columns(
        matched,
        {
            "Asset",
            "Period",
            "Model",
            "Model Average Exposure",
            "Matched Average Exposure",
            "Matched Net Total Return",
            "Net Return Difference vs Static Matched B&H",
            "Constant-Exposure Passive Net Total Return",
            "Net Timing Value vs Constant-Exposure Passive",
        },
        FORMAL_TABLES["matched"],
    )
    require_columns(
        signals,
        {
            "Asset",
            "Period",
            "Date",
            "Execution Date",
            "Scheduled End Date",
            "Five-Session Executable",
        },
        FORMAL_TABLES["signals"],
    )

    expected_metric_rows = len(ASSETS) * len(PERIODS) * (
        len(MODELS) + len(BASELINES)
    )
    expected_model_rows = len(ASSETS) * len(PERIODS) * len(MODELS)
    if len(headline) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} headline rows, found {len(headline)}."
        )
    if len(models) != expected_model_rows or len(matched) != expected_model_rows:
        raise RuntimeError(
            "Expected 24 model and exposure-match rows "
            f"but found {len(models)} and {len(matched)}."
        )
    if set(headline["Asset"]) != set(ASSETS):
        raise RuntimeError("Headline table does not contain the three frozen assets.")
    if set(headline["Period"]) != set(PERIODS):
        raise RuntimeError("Headline table does not contain both frozen periods.")
    if set(models["Strategy"]) != set(MODELS):
        raise RuntimeError("Model table does not contain the four frozen models.")
    if bool(metadata.get("bootstrap_used", True)):
        raise RuntimeError("This descriptive notebook expects bootstrap_used=False.")
    if bool(metadata.get("quarterly_rolling_retraining_used", True)):
        raise RuntimeError(
            "This notebook excludes quarterly rolling retraining by design."
        )
    if bool(metadata.get("return_metrics_used_for_model_training_or_selection", True)):
        raise RuntimeError("Return-informed model training or selection is prohibited.")
    if not bool(metadata.get("five_self_financing_sleeves", False)):
        raise RuntimeError("The formal result is not the declared five-sleeve run.")

    return protocol, metadata, headline, models, matched, signals


def build_tldr(models: pd.DataFrame) -> str:
    """Create a concise, evidence-backed summary from the frozen tables."""

    development = models.loc[models["Period"].eq("Development OOF")].copy()
    later = models.loc[models["Period"].eq("Retrospective audit")].copy()

    winner_lines: list[str] = []
    for asset in ASSETS:
        current = development.loc[development["Asset"].eq(asset)]
        winner = current.loc[current["Net Total Return"].idxmax()]
        winner_lines.append(
            f"**{asset}: {SHORT_NAMES[winner['Strategy']]} "
            f"({percentage(winner['Net Total Return'])})**"
        )

    positive_timing = int(
        development["Net Timing Value vs Constant-Exposure Passive"].gt(0).sum()
    )
    positive_later = int(later["Net Total Return"].gt(0).sum())
    return (
        "## tl;dr\n\n"
        "- This RQ4 notebook evaluates **four frozen predictive models across "
        "TSLA, AAPL, and NVDA** under one five-session, five-sleeve "
        "prediction-to-decision mapping. Development OOF is the primary "
        "descriptive result.\n"
        "- The highest observed Development OOF model return for each asset "
        f"was {'; '.join(winner_lines)}. These are within-asset historical "
        "rankings, not claims of universal model superiority.\n"
        f"- **{positive_timing} of 12** Development asset-model combinations "
        "had positive Net Timing Value relative to their own "
        "constant-exposure passive reference.\n"
        f"- In the concise later-period check, **{positive_later} of 12** "
        "asset-model combinations had a positive net return. The later "
        "period is retained to show temporal stability, but it is not the "
        "headline comparison.\n"
        "- RQ4 is a **descriptive downstream evaluation**. It uses no RQ4 "
        "bootstrap or significance claim and includes no quarterly rolling "
        "retraining experiment."
    )


def main() -> None:
    """Assemble and write the unexecuted notebook."""

    protocol, metadata, headline, models, matched, signals = load_formal_results()
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    research_question = str(protocol["research_question"])
    protocol_id = str(protocol["protocol_id"])
    cells = [
        nbformat.v4.new_markdown_cell(
            "# RQ4: Five-Session Cross-Asset Downstream Evaluation\n\n"
            "Reader-facing evidence notebook for an Artificial Intelligence "
            "and Computer Science capstone. The notebook reads the frozen "
            "RQ4 CSV/JSON tables and official PNG figures only; it does not "
            "train, tune, or rerun any predictive model or backtest."
        ),
        nbformat.v4.new_markdown_cell(build_tldr(models)),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            f"**Research question.** {research_question}\n\n"
            "RQ3 produces a probability after Close $t$ for the class "
            "$1[\\operatorname{mean}(Close_{t+1:t+5}) > "
            "\\operatorname{mean}(Close_{t-4:t})]$. RQ4 converts each frozen "
            "probability into a common downstream decision using a fixed "
            "0.50 threshold. A long signal is implemented at Open $t+1$ and "
            "assigned to one of five self-financing sleeves. That sleeve is "
            "next eligible for a decision at Open $t+6$, producing a "
            "staggered five-session implementation without leverage or "
            "daily cross-sleeve rebalancing.\n\n"
            "The four predictive models are Lagged Logistic Regression, "
            "Random Forest, XGBoost, and Compact LSTM. EMA8/13 state, "
            "SMA10/20 state, and full Buy-and-Hold are transparent reference "
            "strategies. RQ4 returns do not reopen RQ3 model selection.\n\n"
            "### Exposure-Matched References\n\n"
            "Each model is compared with two passive stock/cash references. "
            "First, a static Buy-and-Hold allocation is solved ex post so its "
            "mean value-weighted exposure matches the model; its result is "
            "reported as a return difference, not pure timing value. Second, "
            "a synthetic constant-exposure passive path compounds the same "
            "average exposure each interval. **Net Timing Value** is the "
            "model net return minus this constant-exposure return. Both "
            "cumulative-return differences are reported in percentage points "
            "(pp), while returns and exposures are reported as percentages. "
            "They are descriptive devices and neither is used to tune a model, "
            "threshold, or strategy.\n\n"
            "### Key Assumptions\n\n"
            "- Development OOF predictions are the primary results. The "
            "later period is shown briefly as a retrospective temporal-"
            "stability check, not a prospectively untouched test.\n"
            "- TSLA reuses frozen RQ3 probabilities. AAPL and NVDA use the "
            "same target, six feature channels, 20-session representation, "
            "walk-forward protocol, and fixed threshold, but are fitted "
            "independently; this is cross-asset replication, not zero-shot "
            "transfer from TSLA.\n"
            "- A signal is known after Close $t$; execution occurs at Open "
            "$t+1$; the assigned sleeve's five open-to-open intervals end at "
            "Open $t+6$.\n"
            "- One-way friction is 10 bps transaction cost plus 5 bps adverse "
            "slippage. Cash return, dividends, tax, and additional market "
            "impact are excluded.\n"
            "- The five-session implementation better aligns the decision "
            "horizon with RQ3, but the target uses the mean of five future "
            "closes rather than the fifth-session endpoint return; the two "
            "constructs are related, not mathematically identical.\n"
            "- Results are descriptive historical observations. No RQ4 "
            "bootstrap, p-value, statistical-significance claim, or quarterly "
            "rolling-retraining comparison is included."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ASSETS = ['TSLA', 'AAPL', 'NVDA']\n"
            "PERIODS = ['Development OOF', 'Retrospective audit']\n"
            "MODELS = [\n"
            "    'Lagged Logistic Regression', 'Random Forest',\n"
            "    'XGBoost', 'Compact LSTM'\n"
            "]\n"
            "BASELINES = ['EMA8/EMA13 State', 'SMA10/SMA20 State', 'Buy-and-Hold']\n\n"
            "def locate_experiment_root():\n"
            "    cwd = Path.cwd().resolve()\n"
            "    candidates = [cwd, cwd.parent, cwd / 'v8_4_rq4_five_session_cross_asset']\n"
            "    for candidate in candidates:\n"
            "        if (candidate / 'protocol.json').is_file() and (candidate / 'outputs/rq4_headline_metrics.csv').is_file():\n"
            "            return candidate\n"
            "    raise FileNotFoundError('Run this notebook from the experiment directory, its notebooks directory, or the parent workspace.')\n\n"
            "ROOT = locate_experiment_root()\n"
            "protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))\n"
            "metadata = json.loads((ROOT / 'outputs/rq4_experiment_metadata.json').read_text(encoding='utf-8'))\n"
            "headline = pd.read_csv(ROOT / 'outputs/rq4_headline_metrics.csv', parse_dates=['Path Start', 'Path End'])\n"
            "model_results = pd.read_csv(ROOT / 'outputs/rq4_model_results.csv', parse_dates=['Path Start', 'Path End'])\n"
            "matched = pd.read_csv(ROOT / 'outputs/rq4_exposure_matched_buy_hold.csv', parse_dates=['Path Start', 'Path End'])\n"
            "signals = pd.read_csv(\n"
            "    ROOT / 'outputs/rq4_signal_ledger.csv',\n"
            "    parse_dates=['Date', 'Target_Date', 'Execution Date', 'Scheduled End Date']\n"
            ")\n\n"
            "assert metadata['bootstrap_used'] is False\n"
            "assert metadata['quarterly_rolling_retraining_used'] is False\n"
            "assert metadata['return_metrics_used_for_model_training_or_selection'] is False\n"
            "assert metadata['five_self_financing_sleeves'] is True\n"
            "assert set(headline['Asset']) == set(ASSETS)\n"
            "assert set(headline['Period']) == set(PERIODS)\n"
            "assert len(headline) == 42 and len(model_results) == 24 and len(matched) == 24\n"
            "print('Protocol:', metadata['protocol_id'])\n"
            "print('Loaded formal result rows:', {'headline': len(headline), 'models': len(model_results), 'exposure_matches': len(matched), 'signals': len(signals)})"
        ),
        nbformat.v4.new_markdown_cell(
            "## Data\n\n"
            "The source tables below report the frozen prediction-origin "
            "coverage and the number of five-session decisions that can be "
            "executed through Open $t+6$. The final later-period origin for "
            "each asset is retained in the signal ledger but excluded from "
            "the strategy path when its scheduled endpoint is unavailable."
        ),
        nbformat.v4.new_code_cell(
            "coverage = (\n"
            "    signals.groupby(['Asset', 'Period'], sort=False)\n"
            "    .agg(\n"
            "        Prediction_Origins=('Date', 'size'),\n"
            "        Executable_Five_Session_Decisions=('Five-Session Executable', 'sum'),\n"
            "        Signal_Start=('Date', 'min'),\n"
            "        Signal_End=('Date', 'max'),\n"
            "        First_Execution=('Execution Date', 'min'),\n"
            "        Last_Scheduled_End=('Scheduled End Date', 'max'),\n"
            "    )\n"
            "    .reset_index()\n"
            ")\n"
            "display(coverage)\n"
            "assert coverage.loc[coverage.Period.eq('Development OOF'), 'Executable_Five_Session_Decisions'].eq(1250).all()\n"
            "assert coverage.loc[coverage.Period.eq('Retrospective audit'), 'Executable_Five_Session_Decisions'].eq(406).all()"
        ),
        nbformat.v4.new_code_cell(
            "source_inventory = pd.DataFrame([\n"
            "    {'Formal artefact': 'rq4_headline_metrics.csv', 'Purpose': 'All seven strategies by asset and period', 'Rows': len(headline)},\n"
            "    {'Formal artefact': 'rq4_model_results.csv', 'Purpose': 'Four-model headline and matched-return fields', 'Rows': len(model_results)},\n"
            "    {'Formal artefact': 'rq4_exposure_matched_buy_hold.csv', 'Purpose': 'Exposure matching and Timing Value', 'Rows': len(matched)},\n"
            "    {'Formal artefact': 'rq4_signal_ledger.csv', 'Purpose': 'Frozen prediction-to-execution mapping', 'Rows': len(signals)},\n"
            "])\n"
            "display(source_inventory)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n"
            "### Development OOF: Four Predictive Models Across Three Assets\n\n"
            "Development OOF is the principal RQ4 comparison. The table keeps "
            "return, drawdown, exposure, and exposure-matched Timing Value "
            "together so a high raw return is not interpreted without its "
            "market participation."
        ),
        nbformat.v4.new_code_cell(
            "def format_percentage_units(frame, percent_columns, percentage_point_columns=()):\n"
            "    result = frame.copy()\n"
            "    for column in percent_columns:\n"
            "        result[column] = result[column].map(lambda value: f'{100 * value:,.2f}%')\n"
            "    for column in percentage_point_columns:\n"
            "        result[column] = result[column].map(lambda value: f'{100 * value:,.2f} pp')\n"
            "    return result\n\n"
            "development_models = model_results.loc[\n"
            "    model_results['Period'].eq('Development OOF'),\n"
            "    ['Asset', 'Strategy', 'Net Total Return', 'Annualised Net Return',\n"
            "     'Maximum Drawdown', 'Average Exposure', 'Matched Net Total Return',\n"
            "     'Net Return Difference vs Static Matched B&H',\n"
            "     'Constant-Exposure Passive Net Total Return',\n"
            "     'Net Timing Value vs Constant-Exposure Passive',\n"
            "     'Return Rank within Asset-Period']\n"
            "].sort_values(['Asset', 'Return Rank within Asset-Period'])\n"
            "display(format_percentage_units(\n"
            "    development_models,\n"
            "    ['Net Total Return', 'Annualised Net Return', 'Maximum Drawdown',\n"
            "     'Average Exposure', 'Matched Net Total Return',\n"
            "     'Constant-Exposure Passive Net Total Return'],\n"
            "    ['Net Return Difference vs Static Matched B&H',\n"
            "     'Net Timing Value vs Constant-Exposure Passive']\n"
            "))"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'figures/rq4_development_model_returns.png'), width=980))"
        ),
        nbformat.v4.new_markdown_cell(
            "### Development OOF: Exposure-Matched Comparisons\n\n"
            "A positive value means that the model's observed dynamic "
            "long/cash path returned more than the constant-exposure passive "
            "reference. The static matched Buy-and-Hold difference is shown "
            "separately. Neither establishes statistical or prospective "
            "superiority."
        ),
        nbformat.v4.new_code_cell(
            "development_matched = matched.loc[\n"
            "    matched['Period'].eq('Development OOF'),\n"
            "    ['Asset', 'Model', 'Model Average Exposure',\n"
            "     'Matched Average Exposure', 'Model Net Total Return',\n"
            "     'Matched Net Total Return',\n"
            "     'Net Return Difference vs Static Matched B&H',\n"
            "     'Constant-Exposure Passive Net Total Return',\n"
            "     'Net Timing Value vs Constant-Exposure Passive']\n"
            "].sort_values(['Asset', 'Net Timing Value vs Constant-Exposure Passive'], ascending=[True, False])\n"
            "display(format_percentage_units(\n"
            "    development_matched,\n"
            "    ['Model Average Exposure', 'Matched Average Exposure',\n"
            "     'Model Net Total Return', 'Matched Net Total Return',\n"
            "     'Constant-Exposure Passive Net Total Return'],\n"
            "    ['Net Return Difference vs Static Matched B&H',\n"
            "     'Net Timing Value vs Constant-Exposure Passive']\n"
            "))\n"
            "exposure_error = (matched['Model Average Exposure'] - matched['Matched Average Exposure']).abs().max()\n"
            "assert exposure_error < 1e-9\n"
            "print(f'Maximum absolute exposure-matching error: {exposure_error:.3e}')"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'figures/rq4_development_timing_value.png'), width=980))"
        ),
        nbformat.v4.new_markdown_cell(
            "### Development OOF: Transparent Reference Strategies\n\n"
            "Buy-and-Hold, EMA8/13 state, and SMA10/20 state provide market "
            "and rule-based context. They are reference points rather than "
            "additional candidates for return-informed model selection."
        ),
        nbformat.v4.new_code_cell(
            "development_all = headline.loc[\n"
            "    headline['Period'].eq('Development OOF'),\n"
            "    ['Asset', 'Strategy', 'Family', 'Net Total Return',\n"
            "     'Annualised Net Return', 'Maximum Drawdown', 'Average Exposure']\n"
            "].sort_values(['Asset', 'Net Total Return'], ascending=[True, False])\n"
            "display(format_percentage_units(\n"
            "    development_all,\n"
            "    ['Net Total Return', 'Annualised Net Return',\n"
            "     'Maximum Drawdown', 'Average Exposure']\n"
            "))"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'figures/rq4_development_strategy_comparison.png'), width=1180))"
        ),
        nbformat.v4.new_markdown_cell(
            "### Concise Later-Period Check\n\n"
            "The later period is retained because temporal stability is "
            "relevant to an AI system, but it is not expanded into a second "
            "headline study. These observations are retrospective and do not "
            "trigger model reselection, threshold changes, or rolling "
            "retraining."
        ),
        nbformat.v4.new_code_cell(
            "later_models = model_results.loc[\n"
            "    model_results['Period'].eq('Retrospective audit'),\n"
            "    ['Asset', 'Strategy', 'Net Total Return', 'Maximum Drawdown',\n"
            "     'Average Exposure',\n"
            "     'Net Timing Value vs Constant-Exposure Passive',\n"
            "     'Return Rank within Asset-Period']\n"
            "].sort_values(['Asset', 'Return Rank within Asset-Period'])\n"
            "display(format_percentage_units(\n"
            "    later_models,\n"
            "    ['Net Total Return', 'Maximum Drawdown',\n"
            "     'Average Exposure'],\n"
            "    ['Net Timing Value vs Constant-Exposure Passive']\n"
            "))"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(ROOT / 'figures/rq4_later_period_model_returns.png'), width=980))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. **Machine learning remains central to RQ4.** All four frozen "
            "model families are evaluated across all three assets; the "
            "moving-average and passive paths serve as interpretable "
            "baselines.\n"
            "2. **The downstream mapping now matches the five-session research "
            "horizon more closely.** Five staggered sleeves convert daily "
            "predictions into overlapping five-session decisions without "
            "leverage, although the endpoint return is not identical to the "
            "future-mean-price target.\n"
            "3. **Exposure-matched comparison separates participation from "
            "observed timing.** The static matched Buy-and-Hold difference "
            "and the constant-exposure Timing Value show that the conclusion "
            "depends on benchmark construction, while both remain ex-post "
            "descriptive quantities.\n"
            "4. **Development OOF is the main result; later-period outcomes "
            "remain visible.** Changes in later rankings indicate temporal "
            "instability rather than invalidating the AI pipeline or "
            "justifying selective removal.\n"
            "5. **Claims stay within a Computer Science capstone boundary.** "
            "The notebook demonstrates a reproducible prediction-to-decision "
            "application layer. It does not claim statistical significance, "
            "live profitability, universal investment superiority, or the "
            "benefit of quarterly rolling retraining."
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
            "rq4": {
                "protocol_id": protocol_id,
                "mode": "descriptive downstream evaluation",
                "primary_period": "Development OOF",
                "formal_inputs_only": True,
                "bootstrap_used": bool(metadata["bootstrap_used"]),
                "quarterly_rolling_retraining_used": bool(
                    metadata["quarterly_rolling_retraining_used"]
                ),
            },
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(
        json.dumps(
            {
                "status": "built_unexecuted",
                "notebook": str(NOTEBOOK_PATH.relative_to(ROOT)),
                "cells": len(notebook.cells),
                "code_cells": sum(
                    cell.cell_type == "code" for cell in notebook.cells
                ),
                "protocol_id": protocol_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
