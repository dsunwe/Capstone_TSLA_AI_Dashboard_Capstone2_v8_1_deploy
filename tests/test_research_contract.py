from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest

from src.research_contract import (
    EXPECTED_OBJECTIVE_IDS,
    EXPECTED_RELEASE_VERSION,
    EXPECTED_RQ_IDS,
    PROJECT_ROOT,
    ResearchContractError,
    load_research_contract,
    rq_objective_pairs,
    validate_research_contract,
)


def test_contract_is_release_8_0_with_bijective_alignment():
    contract = load_research_contract()

    assert contract["release"]["version"] == EXPECTED_RELEASE_VERSION
    assert tuple(contract["research_questions"]) == EXPECTED_RQ_IDS
    assert tuple(contract["objectives"]) == EXPECTED_OBJECTIVE_IDS
    assert rq_objective_pairs(contract) == (
        ("RQ1", "O1"),
        ("RQ2", "O2"),
        ("RQ3", "O3"),
        ("RQ4", "O4"),
    )


def test_primary_rqs_match_the_five_session_target_and_reviewed_evidence_scope():
    contract = load_research_contract()
    rq2 = contract["research_questions"]["RQ2"]
    rq3 = contract["research_questions"]["RQ3"]
    rq4 = contract["research_questions"]["RQ4"]
    scopes = contract["evidence_scopes"]

    assert (
        contract["target"]["formula"]
        == "1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])]"
    )
    assert contract["target"]["horizon_trading_sessions"] == 5
    assert contract["target"]["purge_origin_rows"] == 5
    assert contract["target"]["information_cutoff"] == "after Close[t]"
    assert "five-session" in rq2["text"]
    assert "Logistic Regression" in rq2["text"]
    assert scopes["quantitative_feature_family_comparison"]["assets"] == ["TSLA"]
    assert scopes["quantitative_feature_family_comparison"]["models"] == [
        "Regularised Logistic Regression"
    ]

    assert "compact LSTM" in rq3["text"]
    assert contract["rq3_information_design"]["sequence_shape"] == [20, 6]
    assert len(contract["rq3_information_design"]["feature_channels"]) == 6
    assert scopes["quantitative_model_comparison"]["assets"] == ["TSLA"]
    assert scopes["quantitative_model_comparison"]["models"] == [
        "Lagged Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Compact LSTM",
    ]

    assert "TSLA, AAPL and NVDA" in rq4["text"]
    assert scopes["descriptive_downstream_evaluation"]["assets"] == [
        "TSLA",
        "AAPL",
        "NVDA",
    ]
    assert scopes["descriptive_downstream_evaluation"]["models"] == [
        "Lagged Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Compact LSTM",
    ]
    assert "no RQ4 bootstrap" in scopes[
        "descriptive_downstream_evaluation"
    ]["claim_boundary"]
    assert "rolling retraining" in scopes[
        "descriptive_downstream_evaluation"
    ]["claim_boundary"]
    assert contract["decision_layer"]["execution"] == "Open[t+1]"
    assert contract["decision_layer"]["next_same_sleeve_decision"] == "Open[t+6]"
    assert contract["decision_layer"]["sleeves"] == 5
    assert contract["validation_summary"]["analytical_checks_total"] == 84
    assert contract["validation_summary"]["components"] == {
        "RQ2": 22,
        "RQ3": 24,
        "RQ4": 38,
    }


def test_rq4_is_descriptive_without_bootstrap_or_return_driven_reselection():
    contract = load_research_contract()
    rq4_scope = contract["research_questions"]["RQ4"]["scope"]
    boundary_text = " ".join(contract["claim_boundaries"])

    assert "no RQ4 bootstrap" in rq4_scope
    assert "p-value" in rq4_scope
    assert "return-driven model reselection" in rq4_scope
    assert "rolling-retraining comparison" in rq4_scope
    assert "does not use a bootstrap" in boundary_text
    assert (
        "RQ4 return results do not alter RQ2 feature choice, RQ3 model "
        "selection, thresholds, or hyperparameters."
        in contract["claim_boundaries"]
    )


def test_contract_contains_only_the_rq1_rq4_evidence_chain():
    contract = load_research_contract()

    assert "supplementary_study" not in contract
    assert "supplementary_finbert_evidence" not in contract["evidence_scopes"]


def test_contract_rejects_non_bijective_alignment():
    contract = deepcopy(load_research_contract())
    contract["alignment"]["RQ4"] = "O2"

    with pytest.raises(ResearchContractError, match="alignment\\['RQ4'\\]"):
        validate_research_contract(contract)


def test_generated_research_framing_is_synchronised():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "sync_research_framing.py"),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
