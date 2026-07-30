"""Runtime capability checks for the local and cloud deployment profiles."""

from __future__ import annotations

from importlib.util import find_spec

from .config import ALL_MODEL_NAMES, LSTM_MODEL_NAME


def tensorflow_available() -> bool:
    """Return whether live TensorFlow-backed Sandbox training is available."""

    return find_spec("tensorflow") is not None


def sandbox_model_names(
    *,
    has_tensorflow: bool | None = None,
) -> tuple[str, ...]:
    """Return models that can be trained in the current interactive runtime.

    Formal LSTM evidence on Pages 4-6 is loaded from reviewed result files and
    therefore remains available even when the lightweight cloud runtime omits
    TensorFlow.
    """

    available = tensorflow_available() if has_tensorflow is None else has_tensorflow
    return tuple(
        model
        for model in ALL_MODEL_NAMES
        if available or model != LSTM_MODEL_NAME
    )


def live_lstm_notice(
    *,
    has_tensorflow: bool | None = None,
) -> str | None:
    """Return an explanatory UI notice when live LSTM training is unavailable."""

    available = tensorflow_available() if has_tensorflow is None else has_tensorflow
    if available:
        return None
    return (
        "Live LSTM Sandbox training is disabled in this lightweight runtime "
        "because TensorFlow is not installed. The reviewed Compact LSTM "
        "results and evidence remain fully available on Pages 4–6."
    )
