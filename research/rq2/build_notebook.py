"""Build and code-cell validate the reader-facing v8.2 RQ2 notebook."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)
NOTEBOOK = NOTEBOOKS / "v8_2_rq2_ema_aware_diagnostic.ipynb"


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def points(value: float) -> str:
    return f"{value * 100:.2f} pp"


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    formatters: dict[str, callable],
) -> str:
    headers = ["Design", *columns[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                formatters[column](value)
                if column in formatters
                else str(value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    family = pd.read_csv(OUTPUTS / "rq2_feature_family_summary.csv")
    technical = pd.read_csv(OUTPUTS / "rq2_technical_design_summary.csv")
    family_bootstrap = pd.read_csv(
        OUTPUTS / "rq2_feature_family_paired_block_bootstrap.csv"
    )
    technical_bootstrap = pd.read_csv(
        OUTPUTS / "rq2_technical_design_paired_block_bootstrap.csv"
    )
    baseline_bootstrap = pd.read_csv(
        OUTPUTS / "rq2_primary_vs_baseline_paired_block_bootstrap.csv"
    )
    family_phase = pd.read_csv(
        OUTPUTS / "rq2_feature_family_phase_offset_metrics.csv"
    )
    technical_phase_summary = pd.read_csv(
        OUTPUTS / "rq2_technical_design_phase_offset_summary.csv"
    )

    primary = family.loc[
        family["Feature Family"].eq("Technical — EMA-aware compact")
    ].iloc[0]
    persistence = baseline_bootstrap.loc[
        baseline_bootstrap["Group B"].eq("Weekly-trend persistence")
    ].iloc[0]
    ema_core_dev = technical_bootstrap.loc[
        technical_bootstrap["Period"].eq("Development OOF")
        & technical_bootstrap["Group B"].eq(
            "Core-5 without moving-average state"
        )
    ].iloc[0]
    ema_core_audit = technical_bootstrap.loc[
        technical_bootstrap["Period"].eq("Retrospective audit")
        & technical_bootstrap["Group B"].eq(
            "Core-5 without moving-average state"
        )
    ].iloc[0]
    technical_phase = family_phase.loc[
        family_phase["Feature Family"].eq(
            "Technical — EMA-aware compact"
        )
    ]
    family_phase_winners = (
        family_phase.sort_values(
            ["Phase Offset", "Accuracy"],
            ascending=[True, False],
        )
        .groupby("Phase Offset", as_index=False)
        .first()
    )
    technical_phase_wins = int(
        family_phase_winners["Feature Family"]
        .eq("Technical — EMA-aware compact")
        .sum()
    )
    ema_phase_pooled = technical_phase_summary.loc[
        technical_phase_summary["Technical Design"].eq(
            "Primary EMA-aware compact"
        ),
        "Pooled_Accuracy",
    ].iloc[0]
    sma_phase_pooled = technical_phase_summary.loc[
        technical_phase_summary["Technical Design"].eq(
            "SMA10/20-state sensitivity"
        ),
        "Pooled_Accuracy",
    ].iloc[0]

    family_table = markdown_table(
        family.sort_values("Audit_Accuracy", ascending=False),
        [
            "Feature Family",
            "Development_Mean_Accuracy",
            "Development_Accuracy_SD",
            "Audit_Accuracy",
            "Audit_Balanced_Accuracy",
            "Audit_Macro_F1",
        ],
        {
            "Development_Mean_Accuracy": percent,
            "Development_Accuracy_SD": points,
            "Audit_Accuracy": percent,
            "Audit_Balanced_Accuracy": percent,
            "Audit_Macro_F1": lambda value: f"{value:.4f}",
        },
    )
    technical_table = markdown_table(
        technical.sort_values(
            "Development_Mean_Accuracy",
            ascending=False,
        ),
        [
            "Technical Design",
            "Development_Mean_Accuracy",
            "Development_Accuracy_SD",
            "Audit_Accuracy",
            "Audit_Balanced_Accuracy",
            "Audit_Macro_F1",
        ],
        {
            "Development_Mean_Accuracy": percent,
            "Development_Accuracy_SD": points,
            "Audit_Accuracy": percent,
            "Audit_Balanced_Accuracy": percent,
            "Audit_Macro_F1": lambda value: f"{value:.4f}",
        },
    )

    cells = [
        markdown(
            f"""
# v8.2 RQ2 EMA-aware Feature-Family Diagnostic

## tl;dr

The focused RQ2 revision is complete. Under an identical regularised Logistic
Regression framework, the six-feature EMA-aware Technical family achieved
**{percent(primary['Development_Mean_Accuracy'])}** mean development OOF
Accuracy and **{percent(primary['Audit_Accuracy'])}** retrospective audit
Accuracy (Balanced Accuracy **{percent(primary['Audit_Balanced_Accuracy'])}**).

It exceeded the weekly-trend persistence baseline by
**{points(persistence['Point Difference'])}**, with a 20-day paired
moving-block bootstrap 95% interval of
**[{points(persistence['CI Lower 95%'])}, {points(persistence['CI Upper 95%'])}]**.

The EMA state itself provided only a small increment over Core-5:
**{points(ema_core_dev['Point Difference'])}** in development OOF and
**{points(ema_core_audit['Point Difference'])}** in the later audit. Both
intervals include or touch zero. Therefore EMA8/13 is retained because it gives
a theoretically interpretable bridge to RQ4 and slightly improves fold
stability, not because it has been shown to produce a large or statistically
secure Accuracy gain.

Across the five every-fifth-date phase checks, Technical achieved the highest
family Accuracy in **{technical_phase_wins}/5** phases. Its phase Accuracy
ranged from **{percent(technical_phase['Accuracy'].min())}** to
**{percent(technical_phase['Accuracy'].max())}**, so the aggregate 63.88%
should not be presented as uniformly stable over time.
"""
        ),
        markdown(
            r"""
## Context & Methods

RQ2 asks:

> Under an identical regularised Logistic Regression framework, how does
> retrospective walk-forward classification performance differ across four
> pre-specified standalone feature families—EMA-aware technical, cross-market,
> macroeconomic, and broad sentiment/risk proxies—when predicting Tesla's
> five-session forward-versus-trailing mean-close direction?

The target is

$$
y_t=\mathbb{1}\left[
\frac{1}{5}\sum_{i=1}^{5}Close_{t+i}
>
\frac{1}{5}\sum_{i=0}^{4}Close_{t-i}
\right].
$$

The signal information cutoff is after `Close[t]`. Development uses five
chronological folds with 250 test rows and a five-row purge. The common model
uses training-fold median imputation, training-fold standardisation,
`C=0.5`, balanced class weights, and a fixed 0.50 threshold.

### Key assumptions and boundaries

- `EMA8_13_State = 1[EMA8[t] > EMA13[t]]` is a persistent state, not a
  one-day crossover event.
- `SMA10_20_State` is a sensitivity design; raw moving-average price levels are
  not stacked into the main family.
- The 2024–2026 period is a retrospective temporal audit because the historical
  results had already been inspected during redesign.
- The target is a weekly mean-price classification task, not a trading win rate.
- RQ3 model comparisons and RQ4 economic backtests are intentionally excluded.
"""
        ),
        markdown("## Data"),
        code(
            """
from pathlib import Path
import pandas as pd

try:
    from IPython.display import display, Image
except ImportError:
    def display(value):
        print(value)
    def Image(filename):
        return f'[Image: {filename}]'

working_directory = Path.cwd().resolve()
if (working_directory / 'outputs').exists():
    ROOT = working_directory
elif (working_directory.parent / 'outputs').exists():
    ROOT = working_directory.parent
else:
    raise FileNotFoundError(
        'Run from the v8.2 project root or its notebooks directory.'
    )
OUTPUTS = ROOT / 'outputs'
FIGURES = ROOT / 'figures'

family = pd.read_csv(OUTPUTS / 'rq2_feature_family_summary.csv')
technical = pd.read_csv(OUTPUTS / 'rq2_technical_design_summary.csv')
family_bootstrap = pd.read_csv(
    OUTPUTS / 'rq2_feature_family_paired_block_bootstrap.csv'
)
technical_bootstrap = pd.read_csv(
    OUTPUTS / 'rq2_technical_design_paired_block_bootstrap.csv'
)
family_phase = pd.read_csv(
    OUTPUTS / 'rq2_feature_family_phase_offset_metrics.csv'
)
validation = pd.read_csv(OUTPUTS / 'rq2_validation_checks.csv')
"""
        ),
        markdown("## Results"),
        markdown(
            f"""
### 1. Standalone feature-family comparison

{family_table}

Technical produced the highest point estimate in both development and the
later retrospective audit. Macroeconomic features alone remained close to
chance. Cross-market and broad sentiment/risk proxies showed moderate
information, but their later ordering differed from the development ordering.
"""
        ),
        code(
            """
display(
    family[
        [
            'Feature Family',
            'Development_Mean_Accuracy',
            'Development_Accuracy_SD',
            'Audit_Accuracy',
            'Audit_Balanced_Accuracy',
            'Audit_Macro_F1',
            'Audit_ROC_AUC',
        ]
    ].sort_values('Audit_Accuracy', ascending=False)
)
display(Image(filename=str(FIGURES / 'rq2_feature_family_audit.png')))
"""
        ),
        markdown(
            """
### 2. Pairwise uncertainty

Technical exceeded Macroeconomic in the audit with an interval above zero. Its
comparison with Broad sentiment remained statistically unresolved. The lower
bound against Cross-market touched exactly zero, so that contrast should be
treated as borderline rather than conclusive.
"""
        ),
        code(
            """
uncertainty = family_bootstrap.copy()
for column in ['Point Difference', 'CI Lower 95%', 'CI Upper 95%']:
    uncertainty[column] = uncertainty[column] * 100
display(uncertainty)
"""
        ),
        markdown(
            f"""
### 3. Moving-average expression sensitivity

{technical_table}

The EMA-aware design had the smallest fold standard deviation, but its
development Accuracy was 0.08 percentage points below the legacy SMA50-ratio
comparator. The EMA and legacy designs both reached 63.88% audit Accuracy.
This supports selecting EMA for conceptual alignment and stability, not for a
claim of dominant predictive power. In the stricter phase-specific refit,
the SMA sensitivity pooled to **{percent(sma_phase_pooled)}**, compared with
**{percent(ema_phase_pooled)}** for EMA. This further confirms that EMA is not
uniformly superior across specifications.
"""
        ),
        code(
            """
display(technical)
display(technical_bootstrap)
display(Image(filename=str(FIGURES / 'rq2_technical_design_comparison.png')))
"""
        ),
        markdown(
            """
### 4. Five phase-offset robustness checks

Each phase retains every fifth eligible date and refits its own Logistic
Regression using phase-specific development rows. Future five-day windows do
not overlap within a phase, although adjacent phase labels still share a
comparison window and therefore are not fully independent trials.
"""
        ),
        code(
            """
phase_accuracy = family_phase.pivot(
    index='Phase Offset',
    columns='Feature Family',
    values='Accuracy',
)
display(phase_accuracy)
"""
        ),
        markdown("## Takeaways"),
        markdown(
            """
1. Use the EMA-aware six-feature Technical family as the RQ2 main
   specification.
2. Report SMA10/20, Core-5, and legacy SMA50 as sensitivity or ablation
   evidence.
3. State that EMA8/13 improved interpretability and reduced fold variability,
   while its incremental Accuracy contribution was modest and statistically
   unresolved.
4. Carry the same `EMA8 > EMA13` definition into RQ4 as the transparent rule
   comparator. RQ4—not RQ2—will test whether the state can support economic
   returns.
5. Preserve the retrospective claim boundary until a genuinely new future
   period is observed.
"""
        ),
        code(
            """
display(validation)
assert validation['Status'].eq('Pass').all()
"""
        ),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "cells": cells,
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )

    namespace: dict[str, object] = {"__name__": "notebook_validation"}
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    errors: list[str] = []
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    compile(source, f"<notebook-cell-{index}>", "exec"),
                    namespace,
                    namespace,
                )
        except Exception as exc:  # pragma: no cover - handoff safeguard
            errors.append(f"cell {index}: {type(exc).__name__}: {exc}")

    validation_result = {
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "nbformat": notebook["nbformat"],
        "cells": len(cells),
        "code_cells": len(code_cells),
        "code_cells_executed_in_shared_namespace": len(code_cells) - len(errors),
        "stored_kernel_outputs": False,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "note": (
            "Code cells were executed sequentially in a shared Python namespace. "
            "Kernel-generated notebook outputs were not stored because nbformat, "
            "nbclient, and Jupyter are not installed in this build environment."
        ),
    }
    (OUTPUTS / "rq2_notebook_validation.json").write_text(
        json.dumps(validation_result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(validation_result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
