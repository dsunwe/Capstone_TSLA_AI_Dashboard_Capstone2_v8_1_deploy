"""Run read-only system checks for the integrated local v8 Dashboard."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v8_evidence import (
    dataset_summary,
    load_contract,
    load_rq4_model_results,
    rq3_selection_facts,
    rq4_summary_facts,
    validation_summary,
    verify_release_manifest,
)


def _check(check_id: str, description: str, passed: bool, evidence: str) -> dict:
    return {
        "Check ID": check_id,
        "Description": description,
        "Status": "Pass" if bool(passed) else "Fail",
        "Evidence": evidence,
    }


def build_checks() -> pd.DataFrame:
    manifest = verify_release_manifest()
    contract = load_contract()
    datasets = dataset_summary()
    validation = validation_summary()
    selection = rq3_selection_facts()
    rq4 = rq4_summary_facts()
    model_results = load_rq4_model_results()
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    sections_source = (ROOT / "src" / "v8_sections.py").read_text(encoding="utf-8")
    literature_source = (ROOT / "src" / "literature.py").read_text(encoding="utf-8")
    main_source = app_source.split("def main() -> None:", maxsplit=1)[1]
    operational_docs = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in [
            "README.md",
            "docs/architecture.md",
            "docs/deployment.md",
            "docs/requirements_traceability.md",
            "docs/testing_and_validation.md",
            "docs/user_guide.md",
        ]
    )
    sections = [
        "1. Data and Study Protocol",
        "2. Exploratory Data Analysis",
        "3. RQ2 Feature Families",
        "4. RQ3 Model Comparison",
        "5. Cross-Asset Replication",
        "6. RQ4 Downstream Evaluation",
        "7. Analyze Your Data",
        "8. Testing and Limitations",
    ]

    rows = [
        _check(
            "SYS-01",
            "Root manifest verifies every packaged artifact.",
            manifest["passed"] and manifest["failed"] == 0,
            f"{manifest['checked']} checked; {manifest['failed']} failed",
        ),
        _check(
            "SYS-02",
            "Release lineage identifies the historical base and current v8.1 eight-section shell.",
            contract["release"]["version"] == "8.1"
            and "v7.5.3" in contract["lineage"]["code_and_data_base"]
            and "v8.1" in contract["lineage"]["code_and_data_base"]
            and "eight visible sections" in contract["lineage"]["interface_contract"],
            contract["lineage"]["interface_contract"],
        ),
        _check(
            "SYS-03",
            "Prediction target and temporal purge both use five sessions.",
            contract["target"]["horizon_trading_sessions"] == 5
            and contract["target"]["purge_origin_rows"] == 5,
            contract["target"]["formula"],
        ),
        _check(
            "SYS-04",
            "Decision layer uses t+1 execution, t+6 same-sleeve decision, and five sleeves.",
            contract["decision_layer"]["execution"] == "Open[t+1]"
            and contract["decision_layer"]["next_same_sleeve_decision"] == "Open[t+6]"
            and contract["decision_layer"]["sleeves"] == 5,
            str(contract["decision_layer"]),
        ),
        _check(
            "SYS-05",
            "Three asset snapshots share 2,513 complete OHLCV sessions.",
            len(datasets) == 3
            and datasets["Rows"].eq(2513).all()
            and datasets["Unique Dates"].eq(2513).all()
            and datasets["Missing OHLCV"].eq(0).all(),
            datasets.to_json(orient="records"),
        ),
        _check(
            "SYS-06",
            "All saved RQ2-RQ4 analytical checks pass.",
            validation["all_passed"]
            and validation["passed"] == validation["total"] == 84,
            f"{validation['passed']}/{validation['total']}",
        ),
        _check(
            "SYS-07",
            "RQ3 one-SE rule retains Lagged LR while all four models remain eligible.",
            selection["eligible_count"] == 4
            and selection["selected_model"] == "Lagged Logistic Regression",
            str(selection),
        ),
        _check(
            "SYS-08",
            "RQ4 contains the complete 3×2×4 predictive-model grid.",
            len(model_results) == 24,
            f"rows={len(model_results)}",
        ),
        _check(
            "SYS-09",
            "Development comparison facts match the reviewed result ledger.",
            rq4["development_positive_timing"] == 7
            and rq4["development_beats_buy_hold"] == 0
            and rq4["development_beats_ema"] == 0
            and rq4["development_beats_sma"] == 2,
            str(rq4),
        ),
        _check(
            "SYS-10",
            "Later-period comparison facts remain available for robustness review.",
            rq4["audit_positive_return"] == 1
            and rq4["audit_positive_timing"] == 0
            and rq4["audit_beats_buy_hold"] == 0
            and rq4["audit_beats_ema"] == 0
            and rq4["audit_beats_sma"] == 0,
            str(rq4),
        ),
        _check(
            "SYS-11",
            "Local setup, launch, formal recomputation, and manifest verification entry points exist.",
            all(
                (ROOT / relative).is_file()
                for relative in [
                    "setup_windows.bat",
                    "run_dashboard.bat",
                    "setup_macos_linux.sh",
                    "run_dashboard.sh",
                    "scripts/reproduce_all.py",
                    "scripts/verify_root_release_manifest.py",
                ]
            ),
            "Windows and macOS/Linux launchers plus isolated research wrapper",
        ),
        _check(
            "SYS-12",
            "The current single-app workflow exposes RQ1, literature-backed RQ2-RQ4 pages, and an explicit Sandbox without stale interface code or documentation.",
            all(main_source.count(f'"{section}"') >= 1 for section in sections)
            and "RQ1 · Completed Research Dashboard" in app_source
            and "feature_rationale_table()" in sections_source
            and "Why These Six Features Were Selected" in sections_source
            and all(feature in literature_source for feature in contract["rq3_information_design"]["feature_channels"])
            and "render_news_sentiment_study" not in app_source
            and "render_research_framing" not in app_source
            and "render_rule_sensitivity" not in sections_source
            and "render_news_boundary" not in sections_source
            and "Frozen Research Evidence" not in app_source + sections_source
            and "evidence_mode" not in app_source + sections_source
            and not any(
                marker in operational_docs
                for marker in [
                    "nine pages",
                    "Nine-page",
                    "Page 10",
                    "Frozen mode",
                    "Frozen Research Evidence",
                ]
            )
            and "file_uploader(" in app_source
            and "Exploratory classification model" in app_source
            and "Run Sandbox Model" in main_source
            and "render_v8_rq3()" in main_source
            and "render_v8_rq4()" in main_source
            and not (ROOT / "src" / "v8_pages.py").exists(),
            "one app.py; eight sections; explicit RQ1; six-feature literature map; saved RQ3/RQ4 routes; button-triggered CSV Sandbox",
        ),
    ]
    return pd.DataFrame(rows)


def main() -> int:
    checks = build_checks()
    print(checks.to_string(index=False))
    passed = int(checks["Status"].eq("Pass").sum())
    print(f"\nIntegrated release validation: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
