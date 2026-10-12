"""Regression tests for the epoch whose weights inner LSTM scoring uses.

Run the real Keras EarlyStopping callback on prescribed losses, without
optimising a neural network. Encoding the epoch in the fake model's weights
also verifies that fit_lstm_inner predicts *after* restoring those weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _EpochWeightModel:
    """Small model double that executes the real callback lifecycle."""

    def __init__(self, validation_losses, *, report_validation_loss=True):
        self.validation_losses = list(validation_losses)
        self.report_validation_loss = report_validation_loss
        self.stop_training = False
        self.metrics = []
        self.weight = np.array([0.0], dtype=np.float64)
        self.callback = None
        self.predicted_epochs = []
        self.fit_epochs = 0

    def get_weights(self):
        # Keras expects a snapshot, not a mutable reference to current weights.
        return [self.weight.copy()]

    def set_weights(self, weights):
        self.weight = np.array(weights[0], dtype=np.float64, copy=True)

    def fit(self, x_train, y_train, *, callbacks, epochs, **kwargs):
        del y_train
        assert len(callbacks) == 1
        callback = callbacks[0]
        self.callback = callback
        callback.set_model(self)
        callback.set_params({"epochs": epochs, "steps": 1, "verbose": 0})
        callback.on_train_begin({})
        history = {"loss": []}
        if self.report_validation_loss:
            history["val_loss"] = []

        for epoch, value in enumerate(self.validation_losses[:epochs]):
            self.weight = np.array([float(epoch + 1)], dtype=np.float64)
            logs = {"loss": 1.0}
            history["loss"].append(logs["loss"])
            if self.report_validation_loss:
                logs["val_loss"] = value
                history["val_loss"].append(value)
            callback.on_epoch_end(epoch, logs)
            if self.stop_training:
                break

        callback.on_train_end({})
        self.fit_epochs = len(history["loss"])
        # Check unchanged training semantics as well as callback behaviour.
        assert epochs == 100
        assert kwargs["shuffle"] is False
        assert kwargs["batch_size"] == 32
        assert len(x_train) > 0
        return SimpleNamespace(history=history)

    def predict(self, x_validation, *, verbose=0):
        assert verbose == 0
        epoch = float(self.weight[0])
        self.predicted_epochs.append(epoch)
        # Keep encoded output in [0, 1] even at the 100-epoch budget.
        return np.full((len(x_validation), 1), epoch / 1000.0)


@pytest.fixture
def run_inner(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from research.rq3 import run_rq3_experiment as rq3

    monkeypatch.setattr(rq3, "tensorflow_module", lambda: tf)

    def run(losses, *, report_validation_loss=True):
        model = _EpochWeightModel(
            losses, report_validation_loss=report_validation_loss
        )
        monkeypatch.setattr(rq3, "build_lstm", lambda config, seed: model)
        x_train = np.zeros((4, rq3.LOOKBACK, len(rq3.FEATURES)))
        y_train = np.array([0, 1, 0, 1])
        x_validation = np.zeros((3, rq3.LOOKBACK, len(rq3.FEATURES)))
        y_validation = np.array([0, 1, 0])
        result = rq3.fit_lstm_inner(
            x_train,
            y_train,
            x_validation,
            y_validation,
            {"units": 8, "dropout": 0.1, "learning_rate": 0.001},
            seed=42,
        )
        return result, model

    return run


@pytest.mark.parametrize(
    "losses, expected_epoch, expected_epochs_run, expected_stopped_epoch",
    [
        pytest.param(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.60, 0.55, 0.58],
            7,
            8,
            0,
            id="warmup-minimum-is-not-restorable",
        ),
        pytest.param(
            [1.5] * 5 + [0.70000, 0.69995, 0.69980, 0.69975],
            8,
            9,
            0,
            id="sub-min-delta-decrease-does-not-replace-best",
        ),
        pytest.param(
            [2.0] * 5 + [1.0 - 0.002 * i for i in range(90)] + [0.9] * 5,
            95,
            100,
            0,
            id="max-epochs-still-restores-earlier-best",
        ),
        pytest.param(
            [2.0] * 5 + [0.8, 0.75, 0.7, 0.72],
            8,
            9,
            0,
            id="qualifying-improvement-updates-best",
        ),
        pytest.param(
            [1.5] * 5 + [0.7] * 11,
            6,
            16,
            15,
            id="patience-stop-restores-first-monitored-epoch",
        ),
    ],
)
def test_recorded_epoch_matches_restored_prediction_weights(
    run_inner, losses, expected_epoch, expected_epochs_run, expected_stopped_epoch
):
    (probability, best_epoch, epochs_run, elapsed), model = run_inner(losses)
    callback = model.callback

    assert callback.monitor == "val_loss"
    assert callback.start_from_epoch == 5
    assert callback.patience == 10
    assert abs(callback.min_delta) == pytest.approx(1e-4)
    assert callback.restore_best_weights is True
    assert callback.best_weights is not None
    assert callback.stopped_epoch == expected_stopped_epoch
    assert callback.best_epoch + 1 == expected_epoch
    assert best_epoch == expected_epoch
    assert epochs_run == expected_epochs_run == model.fit_epochs
    assert elapsed >= 0.0
    assert model.predicted_epochs == [float(expected_epoch)]
    assert model.weight[0] == pytest.approx(expected_epoch)
    np.testing.assert_allclose(probability, expected_epoch / 1000.0)


def test_missing_validation_monitor_fails_explicitly(run_inner):
    with pytest.warns(UserWarning, match="val_loss"):
        with pytest.raises(RuntimeError, match="monitor|best|epoch|val_loss|restor"):
            run_inner([0.8] * 8, report_validation_loss=False)


def test_no_monitored_epoch_does_not_silently_report_epoch_one(run_inner):
    with pytest.raises(RuntimeError, match="monitor|best|epoch|val_loss|restor"):
        run_inner([0.2, 0.1, 0.3, 0.4, 0.5])


def test_nonfinite_first_monitored_loss_is_rejected(run_inner):
    # Real Keras can select its first observed value even when it is NaN.
    with pytest.raises(RuntimeError, match="finite|monitor|best|epoch|val_loss"):
        run_inner([1.0] * 5 + [float("nan"), 0.7, 0.6])
