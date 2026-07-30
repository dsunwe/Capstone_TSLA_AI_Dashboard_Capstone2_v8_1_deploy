import pandas as pd

import src.export as export_module


def test_empty_current_walk_forward_overwrites_stale_export(tmp_path, monkeypatch):
    feature_path = tmp_path / "features.csv"
    walk_forward_path = tmp_path / "walk_forward.csv"
    walk_forward_path.write_text("Fold,Accuracy\n1,0.99\n", encoding="utf-8")
    monkeypatch.setattr(export_module, "FEATURES_DATA_PATH", feature_path)
    monkeypatch.setattr(export_module, "WALK_FORWARD_RESULTS_PATH", walk_forward_path)

    export_module.save_project_outputs(
        engineered=pd.DataFrame({"Date": ["2026-01-01"], "Price": [100.0]}),
        ml_result=None,
        backtest_metrics=pd.DataFrame(),
        backtest_equity=pd.DataFrame(),
        sensitivity=pd.DataFrame(),
        validation_report=pd.DataFrame(),
        walk_forward=pd.DataFrame(),
    )

    assert walk_forward_path.read_text(encoding="utf-8") == "\n"
