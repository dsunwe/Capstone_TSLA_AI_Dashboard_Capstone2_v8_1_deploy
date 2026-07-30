from __future__ import annotations

from pathlib import Path

from src.config import LSTM_MODEL_NAME, TABULAR_MODEL_NAMES
from src.deployment import live_lstm_notice, sandbox_model_names


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r")):
            continue
        names.add(
            line.split(";", maxsplit=1)[0]
            .split("==", maxsplit=1)[0]
            .split(">=", maxsplit=1)[0]
            .casefold()
        )
    return names


def test_cloud_requirements_are_lightweight_and_local_profile_is_complete():
    cloud_names = _requirement_names(ROOT / "requirements.txt")
    local_text = (ROOT / "requirements-local.txt").read_text(encoding="utf-8")
    locked_names = _requirement_names(ROOT / "requirements-lock.txt")

    assert "streamlit" in cloud_names
    assert "xgboost-cpu" in cloud_names
    assert "tensorflow" not in cloud_names
    assert "tensorflow-cpu" not in cloud_names
    assert "pytest" not in cloud_names
    assert "-r requirements-lock.txt" in local_text
    assert {"tensorflow", "tensorflow-cpu", "pytest"}.issubset(locked_names)


def test_runtime_hides_only_live_lstm_when_tensorflow_is_absent():
    assert sandbox_model_names(has_tensorflow=False) == tuple(TABULAR_MODEL_NAMES)
    assert LSTM_MODEL_NAME not in sandbox_model_names(has_tensorflow=False)
    assert sandbox_model_names(has_tensorflow=True) == tuple(
        [*TABULAR_MODEL_NAMES, LSTM_MODEL_NAME]
    )
    assert live_lstm_notice(has_tensorflow=False) is not None
    assert live_lstm_notice(has_tensorflow=True) is None


def test_cloud_configuration_has_no_committed_secret_or_legacy_runtime_file():
    config = ROOT / ".streamlit" / "config.toml"
    assert config.is_file()
    assert "maxUploadSize = 10" in config.read_text(encoding="utf-8")
    assert not (ROOT / ".streamlit" / "secrets.toml").exists()
    assert not (ROOT / "runtime.txt").exists()
    for higher_priority_environment_file in [
        "uv.lock",
        "Pipfile",
        "environment.yml",
    ]:
        assert not (ROOT / higher_priority_environment_file).exists()
    workflow = ROOT / ".github" / "workflows" / "deploy-smoke.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in text
    assert "/_stcore/health" in text
