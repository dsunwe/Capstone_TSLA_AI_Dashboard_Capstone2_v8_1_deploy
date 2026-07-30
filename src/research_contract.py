from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_CONTRACT_PATH = PROJECT_ROOT / "research_contract.json"
EXPECTED_RELEASE_VERSION = "8.1"
EXPECTED_RQ_IDS = ("RQ1", "RQ2", "RQ3", "RQ4")
EXPECTED_OBJECTIVE_IDS = ("O1", "O2", "O3", "O4")


class ResearchContractError(ValueError):
    """Raised when the canonical research contract is incomplete or inconsistent."""


def _require_nonempty_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{field} must be a non-empty string.")


def validate_research_contract(contract: dict[str, Any]) -> None:
    """Validate release metadata, evidence scope and one-to-one RQ alignment."""

    if not isinstance(contract, dict):
        raise ResearchContractError("The research contract must be a JSON object.")

    release = contract.get("release")
    if not isinstance(release, dict):
        raise ResearchContractError("release must be an object.")
    if release.get("version") != EXPECTED_RELEASE_VERSION:
        raise ResearchContractError(
            f"release.version must be {EXPECTED_RELEASE_VERSION!r}."
        )
    for field in ("name", "status"):
        _require_nonempty_text(release.get(field), f"release.{field}")

    _require_nonempty_text(contract.get("title"), "title")
    _require_nonempty_text(contract.get("aim"), "aim")

    questions = contract.get("research_questions")
    objectives = contract.get("objectives")
    alignment = contract.get("alignment")
    if not isinstance(questions, dict) or tuple(questions) != EXPECTED_RQ_IDS:
        raise ResearchContractError(
            f"research_questions must contain {EXPECTED_RQ_IDS} in order."
        )
    if not isinstance(objectives, dict) or tuple(objectives) != EXPECTED_OBJECTIVE_IDS:
        raise ResearchContractError(
            f"objectives must contain {EXPECTED_OBJECTIVE_IDS} in order."
        )
    if not isinstance(alignment, dict) or tuple(alignment) != EXPECTED_RQ_IDS:
        raise ResearchContractError(f"alignment must contain {EXPECTED_RQ_IDS} in order.")

    mapped_objectives: list[str] = []
    for index, rq_id in enumerate(EXPECTED_RQ_IDS):
        objective_id = EXPECTED_OBJECTIVE_IDS[index]
        question = questions[rq_id]
        objective = objectives[objective_id]
        if not isinstance(question, dict):
            raise ResearchContractError(f"{rq_id} must be an object.")

        for field in ("text", "scope"):
            _require_nonempty_text(question.get(field), f"{rq_id}.{field}")
        objective_text = (
            objective.get("text") if isinstance(objective, dict) else objective
        )
        _require_nonempty_text(objective_text, f"{objective_id}.text")
        if question.get("objective", objective_id) != objective_id:
            raise ResearchContractError(
                f"{rq_id}.objective must be {objective_id}, not "
                f"{question.get('objective')!r}."
            )
        if (
            isinstance(objective, dict)
            and objective.get("research_question", rq_id) != rq_id
        ):
            raise ResearchContractError(
                f"{objective_id}.research_question must be {rq_id}, not "
                f"{objective.get('research_question')!r}."
            )
        if alignment.get(rq_id) != objective_id:
            raise ResearchContractError(
                f"alignment[{rq_id!r}] must be {objective_id!r}."
            )
        mapped_objectives.append(objective_id)

    if len(set(mapped_objectives)) != len(EXPECTED_OBJECTIVE_IDS):
        raise ResearchContractError("RQ-to-objective alignment must be one-to-one.")

    target = contract.get("target")
    if not isinstance(target, dict):
        raise ResearchContractError("target must be an object.")
    for field in ("formula", "information_cutoff", "interpretation_boundary"):
        _require_nonempty_text(target.get(field), f"target.{field}")
    if target.get("horizon_trading_sessions") != 5:
        raise ResearchContractError("target.horizon_trading_sessions must be 5.")
    if target.get("purge_origin_rows") != 5:
        raise ResearchContractError("target.purge_origin_rows must be 5.")

    decision = contract.get("decision_layer")
    if not isinstance(decision, dict):
        raise ResearchContractError("decision_layer must be an object.")
    for field in (
        "signal",
        "execution",
        "next_same_sleeve_decision",
        "target_alignment_boundary",
    ):
        _require_nonempty_text(decision.get(field), f"decision_layer.{field}")
    if decision.get("execution") != "Open[t+1]":
        raise ResearchContractError("decision_layer.execution must be Open[t+1].")
    if decision.get("next_same_sleeve_decision") != "Open[t+6]":
        raise ResearchContractError(
            "decision_layer.next_same_sleeve_decision must be Open[t+6]."
        )
    if decision.get("sleeves") != 5:
        raise ResearchContractError("decision_layer.sleeves must be 5.")

    boundaries = contract.get("claim_boundaries")
    if (
        not isinstance(boundaries, list)
        or not boundaries
        or any(not isinstance(item, str) or not item.strip() for item in boundaries)
    ):
        raise ResearchContractError(
            "claim_boundaries must be a non-empty list of strings."
        )


def load_research_contract(
    path: str | Path = RESEARCH_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and validate the canonical Capstone II research contract."""

    contract_path = Path(path)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchContractError(
            f"Research contract not found: {contract_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ResearchContractError(
            f"Research contract is not valid JSON: {exc}"
        ) from exc

    validate_research_contract(contract)
    return contract


def rq_objective_pairs(
    contract: dict[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return the reviewed bijective RQ-to-objective mapping."""

    loaded = load_research_contract() if contract is None else contract
    validate_research_contract(loaded)
    return tuple((rq_id, loaded["alignment"][rq_id]) for rq_id in EXPECTED_RQ_IDS)
