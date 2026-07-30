"""Generate docs/research_framing.md from the canonical v8 research contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research_contract import load_research_contract, rq_objective_pairs


OUTPUT_PATH = PROJECT_ROOT / "docs" / "research_framing.md"


def _join(values: list[str] | None) -> str:
    return ", ".join(values or ["Not applicable"])


def _objective_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value["text"])
    return str(value)


def render_markdown(contract: dict[str, Any]) -> str:
    """Render the canonical v8 framing without reintroducing legacy RQ claims."""

    release = contract["release"]
    lineage = contract.get("lineage", {})
    questions = contract["research_questions"]
    objectives = contract["objectives"]
    target = contract["target"]
    rq3 = contract["rq3_information_design"]
    decision = contract["decision_layer"]
    validation = contract["validation_summary"]
    validation_parts = validation["components"]

    lines = [
        f"# Canonical Research Contract for Capstone II v{release['version']}",
        "",
        "> This file is generated from `research_contract.json` by",
        "> `scripts/sync_research_framing.py`. Edit the contract, not this file.",
        "",
        "## Contract Metadata",
        "",
        f"- **Release:** {release['version']}",
        f"- **Release name:** {release['name']}",
        f"- **Date:** {release['date']}",
        f"- **Status:** {release['status']}",
        "",
        "## Title",
        "",
        f"**{contract['title']}**",
        "",
        "## Research Lineage",
        "",
        f"- **Code and data base:** {lineage['code_and_data_base']}",
        f"- **Interface contract:** {lineage['interface_contract']}",
        f"- **Integration boundary:** {lineage['integration_boundary']}",
        "",
        "## Aim",
        "",
        contract["aim"],
        "",
        "## Shared Five-Session Target",
        "",
        f"- **Formula:** `{target['formula']}`",
        f"- **Information cut-off:** {target['information_cutoff']}",
        f"- **Target horizon:** {target['horizon_trading_sessions']} trading sessions",
        f"- **Target date:** trading session `t+{target['horizon_trading_sessions']}`",
        f"- **Purged origin rows:** {target['purge_origin_rows']}",
        f"- **Interpretation boundary:** {target['interpretation_boundary']}",
        "",
        "The classification target compares the mean of the next five closes with",
        "the trailing five-close mean available at the signal origin. It is not the",
        "same-session trading win rate and is not a fifth-session endpoint-return label.",
        "",
        "## Strict RQ-to-Objective Alignment",
        "",
        "| Research question | Objective |",
        "|---|---|",
    ]
    for rq_id, objective_id in rq_objective_pairs(contract):
        lines.append(f"| {rq_id} | {objective_id} |")

    lines.extend(["", "## Research Questions, Objectives and Evidence", ""])
    for rq_id, objective_id in rq_objective_pairs(contract):
        question = questions[rq_id]
        lines.extend(
            [
                f"### {rq_id} ↔ {objective_id}",
                "",
                f"**{rq_id}.** {question['text']}",
                "",
                f"**{objective_id}.** {_objective_text(objectives[objective_id])}",
                "",
                f"**Scope boundary.** {question['scope']}",
                "",
                "**Reviewed evidence:**",
                "",
                *[f"- `{path}`" for path in question["evidence"]],
                "",
            ]
        )

    lines.extend(
        [
            "## RQ3 Common-Information Design",
            "",
            f"- **Lookback:** {rq3['lookback_trading_sessions']} trading sessions",
            f"- **Sequence shape:** `{rq3['sequence_shape'][0]} × {rq3['sequence_shape'][1]}`",
            f"- **Feature channels:** {_join(rq3['feature_channels'])}",
            f"- **Common-information rule:** {rq3['common_information_rule']}",
            "",
            "The tabular estimators receive the ordered 120-value flattening. Compact",
            "LSTM retains the same 20 × 6 tensor. RQ3 model selection uses Development",
            "evidence only; the later audit and RQ4 returns do not select the model.",
            "",
            "## RQ4 Five-Sleeve Decision Layer",
            "",
            f"- **Assets:** {_join(decision['assets'])}",
            f"- **Models:** {_join(decision['models'])}",
            f"- **Signal rule:** {decision['signal']}",
            f"- **Execution:** `{decision['execution']}`",
            f"- **Same-sleeve reconsideration/valuation:** `{decision['next_same_sleeve_decision']}`",
            f"- **Self-financing sleeves:** {decision['sleeves']}",
            f"- **Initial weight per sleeve:** {decision['initial_weight_per_sleeve']:.0%}",
            f"- **One-way friction:** {decision['one_way_friction_bps']} bps",
            f"- **Target-to-decision boundary:** {decision['target_alignment_boundary']}",
            "",
            "RQ4 is a descriptive application-layer evaluation. It does not use an",
            "RQ4 bootstrap, p-value, statistical-superiority claim, return-driven",
            "model reselection, or quarterly rolling-retraining comparison.",
            "",
            "## Evidence-Scope Matrix",
            "",
            "| Evidence scope | Purpose | Assets | Models | Evaluation | Claim boundary |",
            "|---|---|---|---|---|---|",
        ]
    )
    for scope_id, scope in contract["evidence_scopes"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{scope_id}`",
                    scope["purpose"],
                    _join(scope.get("assets")),
                    _join(scope.get("models")),
                    scope["evaluation"],
                    scope["claim_boundary"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Saved Analytical Validation",
            "",
            "| Module | Passed |",
            "|---|---:|",
            f"| RQ2 | {validation_parts['RQ2']}/{validation_parts['RQ2']} |",
            f"| RQ3 | {validation_parts['RQ3']}/{validation_parts['RQ3']} |",
            f"| RQ4 | {validation_parts['RQ4']}/{validation_parts['RQ4']} |",
            f"| **Total** | **{validation['analytical_checks_total']}/{validation['analytical_checks_total']}** |",
            "",
            validation["boundary"],
        ]
    )

    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            *[f"- {boundary}" for boundary in contract["claim_boundaries"]],
            "",
            "## Contribution Claim",
            "",
            "The contribution is a reproducible local AI research system that connects",
            "a working eight-section Streamlit interface with the final",
            "five-session target, standalone feature-family comparison, common 20 × 6",
            "model comparison, and three-asset five-sleeve downstream evaluation.",
            "It is not a claim of guaranteed profitability, universal model",
            "superiority, or a deployable live-trading signal.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/research_framing.md is not synchronised.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Markdown output path.",
    )
    args = parser.parse_args()

    rendered = render_markdown(load_research_contract())
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(
                f"{args.output} is not synchronised with research_contract.json.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is synchronised.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
