import re

from src.config import PRESPECIFIED_OOF_FEATURES
from src.literature import (
    REFERENCE_URLS,
    feature_rationale_table,
    method_rationale_table,
    references_table,
)


def test_rq2_feature_rationale_covers_the_six_inputs_once():
    table = feature_rationale_table()

    assert table["Feature"].tolist() == list(PRESPECIFIED_OOF_FEATURES)
    assert table["Feature"].is_unique
    assert len(table) == 6
    assert table[
        [
            "Definition",
            "Market state captured",
            "Why included",
            "Literature support",
            "Project-specific boundary",
        ]
    ].notna().all().all()
    assert table["Project-specific boundary"].str.len().gt(30).all()


def test_literature_mapping_covers_every_dashboard_method_part():
    sections = ("rq1", "eda", "rq2", "rq3", "cross_asset", "rq4", "validation")

    for section in sections:
        table = method_rationale_table(section)
        assert not table.empty
        assert {
            "Design decision",
            "Reason in this project",
            "Literature support",
            "Boundary",
        }.issubset(table.columns)
        assert table.notna().all().all()


def test_complete_reference_table_has_resolvable_doi_or_source_links():
    table = references_table()

    assert len(table) == len(REFERENCE_URLS)
    assert table["DOI / source"].is_unique
    assert table["DOI / source"].str.startswith("https://").all()
    assert set(table["DOI / source"]) == set(REFERENCE_URLS.values())


def test_all_dashboard_references_are_from_2020_or_later():
    table = references_table()
    key_years = [
        int(re.search(r"\((\d{4})\)", key).group(1))
        for key in REFERENCE_URLS
    ]
    reference_years = [
        int(re.search(r"\((\d{4})\)", reference).group(1))
        for reference in table["Reference"]
    ]

    assert min(key_years) >= 2020
    assert min(reference_years) >= 2020


def test_every_feature_and_method_citation_resolves_to_a_reference():
    support_cells = feature_rationale_table()["Literature support"].tolist()
    for section in (
        "rq1",
        "eda",
        "rq2",
        "rq3",
        "cross_asset",
        "rq4",
        "validation",
    ):
        support_cells.extend(
            method_rationale_table(section)["Literature support"].tolist()
        )

    cited_keys = {
        key.strip()
        for support in support_cells
        for key in support.split(";")
    }
    assert cited_keys <= set(REFERENCE_URLS)


def test_twenty_session_design_has_direct_recent_lstm_precedent():
    key = "Petrozziello et al. (2022)"
    table = method_rationale_table("rq3")
    common_design = table.loc[
        table["Design decision"].eq("Common 20 × 6 information design")
    ].iloc[0]

    assert key in REFERENCE_URLS
    assert key in common_design["Literature support"]
    assert "20-day LSTM look-back" in common_design["Boundary"]
