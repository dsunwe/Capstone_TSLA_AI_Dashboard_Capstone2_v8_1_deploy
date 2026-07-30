from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_has_no_explicit_download_widgets_or_helpers():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]
    )

    assert "st.download_button" not in source
    assert "dataframe_to_csv_bytes" not in source
    assert "def _download_csv" not in source
    assert "def _download_json" not in source


def test_dataframe_export_is_disabled_globally():
    config = tomllib.loads(
        (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert config["client"]["disableDataExport"] is True
