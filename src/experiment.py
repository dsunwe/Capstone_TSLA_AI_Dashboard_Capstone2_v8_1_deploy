from __future__ import annotations

import pandas as pd

from .config import PRESPECIFIED_OOF_FEATURES
from .research_contract import load_research_contract
from .temporal import TemporalSplitPlan


def build_experiment_configuration(
    data: pd.DataFrame,
    *,
    source_name: str,
    model_name: str,
    settings: dict,
    selected_features: list[str] | None = None,
    split_plan: TemporalSplitPlan | None = None,
    feature_protocol: str = "Reviewed six-feature RQ2 Technical specification",
    oof_feature_protocol: str = "Same six RQ2 inputs in every outer fold",
) -> pd.DataFrame:
    """Create a compact, exportable record of one experiment configuration."""
    contract = load_research_contract()
    fixed_features = list(selected_features or PRESPECIFIED_OOF_FEATURES)
    values = {
        "Release": contract["release"]["version"],
        "Release name": contract["release"]["name"],
        "Research contract status": contract["release"]["status"],
        "Data source": source_name,
        "Rows after preprocessing": len(data),
        "Data start": data["Date"].min().date(),
        "Data end": data["Date"].max().date(),
        "Model": model_name,
        "Fixed-holdout feature protocol": feature_protocol,
        "Fixed-holdout selected features": len(fixed_features),
        "Fixed-holdout feature names": ", ".join(fixed_features),
        "Expanding-OOF feature protocol": oof_feature_protocol,
        "Expanding-OOF candidate features": ", ".join(fixed_features),
        "Chronological train fraction": 0.80,
        "Prediction target": (
            "1[mean(Close[t+1:t+5]) > mean(Close[t-4:t])] using information "
            "available through Close[t]"
        ),
        "Boundary purge gap": "5 trading-session anchors",
        "Common temporal plan ID": (
            split_plan.plan_id if split_plan is not None else "Not supplied"
        ),
        "Common temporal plan SHA-256": (
            split_plan.plan_hash if split_plan is not None else "Not supplied"
        ),
        "Common train boundary": (
            f"{split_plan.train_start_date.date()} to {split_plan.train_end_date.date()}"
            if split_plan is not None
            else "Not supplied"
        ),
        "Purged boundary": (
            f"{split_plan.gap_start_date.date()} to {split_plan.gap_end_date.date()}"
            if split_plan is not None
            else "Not supplied"
        ),
        "Common test boundary": (
            f"{split_plan.test_start_date.date()} to {split_plan.test_end_date.date()}"
            if split_plan is not None
            else "Not supplied"
        ),
        "Random shuffle": False,
        "Random seed": 42,
        "Probability threshold": settings["threshold"],
        "Random Forest trees": settings["n_estimators"],
        "Random Forest max depth": settings["max_depth"],
        "Training-only hyperparameter tuning": settings.get("tune_model", False),
        "OOF nested tuning protocol": settings.get(
            "oof_nested_tuning",
            "Not supplied",
        ),
        "OOF LSTM deterministic seeds": settings.get(
            "oof_lstm_seeds",
            "Not supplied",
        ),
        "OOF LSTM ensemble rule": (
            "Arithmetic mean of seed probabilities before applying threshold"
        ),
        "Main-study uncertainty protocol": (
            "Descriptive downstream evaluation; no statistical-significance claim"
        ),
        "Tuning method": (
            "Chronological validation and early stopping"
            if model_name == "LSTM Classifier"
            else (
                "RandomizedSearchCV with TimeSeriesSplit(gap=5), scored by Macro-F1"
                if settings.get("tune_model", False)
                else "Disabled"
            )
        ),
        "LSTM lookback": settings.get("lstm_lookback", "Not applicable"),
        "LSTM units": settings.get("lstm_units", "Not applicable"),
        "LSTM dropout": settings.get("lstm_dropout", "Not applicable"),
        "LSTM maximum epochs": settings.get("lstm_epochs", "Not applicable"),
        "Fixed-holdout LSTM maximum epochs": settings.get(
            "fixed_holdout_lstm_epochs",
            "Not applicable",
        ),
        "LSTM batch size": settings.get("lstm_batch_size", "Not applicable"),
        "Walk-forward folds": settings["cv_splits"],
        "Transaction cost (bps)": settings["transaction_cost_bps"],
        "Slippage (bps)": settings["slippage_bps"],
        "Backtest signal time": "After day-t close",
        "Backtest execution time": "Next trading-session open",
        "Backtest valuation return": (
            "Five staggered 20% sleeves; each sleeve holds from Open[t+1] "
            "to its corresponding Open[t+6]"
        ),
    }
    return pd.DataFrame({"Setting": values.keys(), "Value": values.values()})
