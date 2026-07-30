"""Build and execute the reader-facing v8.3 RQ3 evidence notebook."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import matplotlib
import nbformat
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS / "v8_3_rq3_model_comparison.ipynb"


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    summary = pd.read_csv(OUTPUTS / "rq3_performance_summary.csv")
    selection = pd.read_csv(OUTPUTS / "rq3_one_se_selection.csv")
    bootstrap = pd.read_csv(OUTPUTS / "rq3_paired_block_bootstrap.csv")
    selected = selection.loc[selection["Selected for RQ4"], "Model"].iloc[0]
    best = selection.sort_values(
        "Mean_Balanced_Accuracy",
        ascending=False,
    ).iloc[0]
    selected_row = summary.loc[summary["Comparator"].eq(selected)].iloc[0]
    lstm_row = summary.loc[
        summary["Comparator"].eq("Compact LSTM")
    ].iloc[0]
    lstm_audit_bootstrap = bootstrap.loc[
        bootstrap["Comparison Family"].eq("Primary complex-versus-linear")
        & bootstrap["Period"].eq("Retrospective audit")
        & bootstrap["Metric"].eq("Balanced Accuracy")
        & bootstrap["Comparator A"].eq("Compact LSTM")
    ].iloc[0]

    cells = [
        nbformat.v4.new_markdown_cell(
            "# RQ3 Common-Information Model Comparison\n\n"
            "Executed evidence notebook for the focused TSLA v8.3 RQ3 "
            "retrospective experiment."
        ),
        nbformat.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"- Compact LSTM had the highest mean outer-fold Balanced "
            f"Accuracy ({percentage(best['Mean_Balanced_Accuracy'])}), "
            f"but all four models fell within one standard error.\n"
            f"- The pre-specified simplicity rule therefore selected "
            f"**{selected}** for RQ4.\n"
            f"- In the retrospective audit, {selected} achieved "
            f"{percentage(selected_row['Audit Accuracy'])} Accuracy and "
            f"{percentage(selected_row['Audit Balanced Accuracy'])} "
            f"Balanced Accuracy. Compact LSTM achieved "
            f"{percentage(lstm_row['Audit Accuracy'])} and "
            f"{percentage(lstm_row['Audit Balanced Accuracy'])}, "
            f"respectively.\n"
            f"- The audit LSTM-minus-LR Balanced Accuracy difference was "
            f"{100*lstm_audit_bootstrap['Point Difference']:.2f} pp "
            f"(95% block-bootstrap CI "
            f"{100*lstm_audit_bootstrap['CI Lower 95%']:.2f} to "
            f"{100*lstm_audit_bootstrap['CI Upper 95%']:.2f} pp; "
            f"Holm-adjusted p="
            f"{lstm_audit_bootstrap['Holm-adjusted p']:.3f}).\n"
            "- These are retrospective classification results, not trading "
            "win rates or prospective confirmation."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "All four main models receive the same frozen RQ2 Technical "
            "feature streams over sessions t-19 through t. LR, RF, and "
            "XGBoost flatten the 20×6 tensor to 120 ordered inputs; LSTM "
            "retains the tensor. The target is "
            "`1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]`.\n\n"
            "The 20-session window was fixed before the final comparison as "
            "a compact approximation of four trading weeks. In a related "
            "but different one-step-ahead daily equity-volatility task, "
            "Petrozziello et al. (2022) used walk-forward validation and "
            "selected a 20-day LSTM lookback from a pre-defined grid. This "
            "provides task-adjacent precedent rather than evidence that 20 "
            "sessions is optimal for TSLA direction classification or every "
            "model family. "
            "The adapted one-standard-error rule follows the parsimony "
            "heuristic discussed by Chen and Yang (2021), whose documented "
            "limitations are why it is treated as conservative selection "
            "rather than an equivalence test.\n\n"
            "### Key Assumptions\n\n"
            "- Features are available after close t.\n"
            "- Five origin rows are purged at every train/evaluation boundary.\n"
            "- Audit observations were previously inspected and are therefore "
            "a retrospective temporal audit.\n"
            "- One-SE model selection uses development outer-fold Balanced "
            "Accuracy only; audit and RQ4 returns are excluded."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n\n"
            "ROOT = Path.cwd()\n"
            "assert (ROOT / 'outputs').exists(), ROOT\n"
            "summary = pd.read_csv(ROOT / 'outputs/rq3_performance_summary.csv')\n"
            "selection = pd.read_csv(ROOT / 'outputs/rq3_one_se_selection.csv')\n"
            "folds = pd.read_csv(ROOT / 'outputs/rq3_fold_metrics.csv')\n"
            "bootstrap = pd.read_csv(ROOT / 'outputs/rq3_paired_block_bootstrap.csv')\n"
            "phases = pd.read_csv(ROOT / 'outputs/rq3_phase_offset_metrics.csv')\n"
            "predictions = pd.read_csv(ROOT / 'outputs/rq3_all_predictions.csv')\n"
            "metadata = json.loads((ROOT / 'outputs/rq3_experiment_metadata.json').read_text())\n"
            "metadata['selected_model'], predictions.shape"
        ),
        nbformat.v4.new_markdown_cell("## Data"),
        nbformat.v4.new_code_cell(
            "sequence_index = pd.read_csv(ROOT / 'outputs/rq3_sequence_index.csv')\n"
            "outer = pd.read_csv(ROOT / 'outputs/rq3_outer_folds.csv')\n"
            "display(sequence_index.head(3))\n"
            "display(outer)"
        ),
        nbformat.v4.new_code_cell(
            "counts = predictions.groupby(['Period','Comparator'])['Date'].nunique().unstack(0)\n"
            "display(counts)\n"
            "assert counts['Development OOF'].eq(1250).all()\n"
            "assert counts['Retrospective audit'].eq(407).all()"
        ),
        nbformat.v4.new_markdown_cell("## Results"),
        nbformat.v4.new_code_cell(
            "main_models = ['Lagged Logistic Regression','Random Forest','XGBoost','Compact LSTM']\n"
            "columns = ['Comparator','Development Mean Fold Accuracy',"
            "'Development Mean Fold Balanced Accuracy','Development Balanced Accuracy SD',"
            "'Audit Accuracy','Audit Balanced Accuracy','Audit Macro F1','Audit MCC']\n"
            "display(summary.loc[summary.Comparator.isin(main_models), columns].round(4))\n"
            "display(selection.round(4))"
        ),
        nbformat.v4.new_code_cell(
            "model_summary = summary.loc[summary.Comparator.isin(main_models)].copy()\n"
            "x = np.arange(len(model_summary)); width = 0.36\n"
            "fig, ax = plt.subplots(figsize=(9,4.8))\n"
            "ax.bar(x-width/2, 100*model_summary['Development Mean Fold Balanced Accuracy'], width, label='Development mean fold')\n"
            "ax.bar(x+width/2, 100*model_summary['Audit Balanced Accuracy'], width, label='Retrospective audit')\n"
            "ax.set_xticks(x, model_summary.Comparator, rotation=15, ha='right')\n"
            "ax.set_ylabel('Balanced Accuracy (%)')\n"
            "ax.set_title('RQ3 model performance with common 20-session information')\n"
            "ax.set_ylim(40,80); ax.grid(axis='y', alpha=.25); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        nbformat.v4.new_code_cell(
            "primary = bootstrap.loc[bootstrap['Comparison Family'].eq('Primary complex-versus-linear'),\n"
            "    ['Period','Metric','Comparator A','Comparator B','Point Difference','CI Lower 95%','CI Upper 95%','Holm-adjusted p']]\n"
            "display(primary.round(4))"
        ),
        nbformat.v4.new_code_cell(
            "phase_main = phases.loc[phases.Comparator.isin(main_models)]\n"
            "phase_ranges = phase_main.groupby(['Period','Comparator']).agg(\n"
            "    Min_BA=('Balanced Accuracy','min'), Max_BA=('Balanced Accuracy','max'), SD_BA=('Balanced Accuracy','std')).reset_index()\n"
            "display(phase_ranges.round(4))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            f"1. **Model complexity did not provide statistically conclusive "
            f"incremental value over lagged LR.** LSTM's audit point estimate "
            f"was slightly higher, but its paired interval versus LR crossed "
            f"zero.\n"
            f"2. **The one-SE rule selected {selected}.** This is a conservative "
            f"development-only choice, not a claim that LR was uniquely best.\n"
            "3. **Transparent baselines remain informative.** The selected LR "
            "was materially above weekly persistence in the audit, while its "
            "advantage over EMA state was not statistically stable across both "
            "periods.\n"
            "4. **RQ4 must use the frozen probabilities and threshold.** It "
            "must not reselect the model after seeing economic returns."
        ),
        nbformat.v4.new_markdown_cell(
            "## References\n\n"
            "Chen, Y., & Yang, Y. (2021). The one standard error rule for "
            "model selection: Does it work? *Stats, 4*(4), 868–892. "
            "https://doi.org/10.3390/stats4040051\n\n"
            "Petrozziello, A., Troiano, L., Serra, A., Jordanov, I., "
            "Storti, G., Tagliaferri, R., & La Rocca, M. (2022). Deep "
            "learning for volatility forecasting in asset management. "
            "*Soft Computing, 26*(17), 8553–8574. "
            "https://doi.org/10.1007/s00500-022-07161-1"
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
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    previous_directory = Path.cwd()
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    execution_count = 0
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if not result.success:
                raise RuntimeError(
                    f"Notebook code cell {execution_count} failed."
                )
            outputs = []
            if captured.stdout:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text=captured.stdout,
                    )
                )
            if captured.stderr:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stderr",
                        text=captured.stderr,
                    )
                )
            for rich_output in captured.outputs:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data=rich_output.data,
                        metadata=rich_output.metadata or {},
                    )
                )
            import matplotlib.pyplot as plt

            for figure_number in plt.get_fignums():
                figure = plt.figure(figure_number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": base64.b64encode(
                                buffer.getvalue()
                            ).decode("ascii")
                        },
                        metadata={},
                    )
                )
                plt.close(figure)
            cell.outputs = outputs
            cell.execution_count = execution_count
    finally:
        os.chdir(previous_directory)
    nbformat.write(notebook, NOTEBOOK_PATH)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    status = {
        "status": "executed",
        "execution_backend": (
            "in-process IPython shared namespace; local Jupyter kernel TCP "
            "startup is unavailable in this sandbox"
        ),
        "notebook": str(NOTEBOOK_PATH.relative_to(ROOT)),
        "cells": len(notebook.cells),
        "code_cells_executed": len(code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
    }
    (OUTPUTS / "rq3_notebook_validation.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
