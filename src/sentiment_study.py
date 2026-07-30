from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .config import (
    ALL_MODEL_NAMES,
    DAILY_NEWS_FEATURES_PATH,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_LOOKBACK,
    LSTM_MODEL_NAME,
    NEWS_ATTENTION_FEATURES,
    NEWS_SENTIMENT_FEATURES,
    SENTIMENT_BASE_FEATURES,
    SENTIMENT_FEATURE_SETS,
    SENTIMENT_STUDY_END,
    SENTIMENT_STUDY_START,
    SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
    SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
    SUPPLEMENTARY_TARGET_RETURN_COLUMN,
)
from .ml import _classification_metrics, build_classifier
from .research import prepare_multi_asset_engineered


TARGET_DIRECTION = "direction"
TARGET_LARGE_MOVE = "large_move"
TARGETS = [TARGET_DIRECTION, TARGET_LARGE_MOVE]
MAJORITY_MODEL_NAME = "Majority Baseline"


@dataclass
class SentimentStudyResult:
    summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_thresholds: pd.DataFrame
    study_frames: dict[str, pd.DataFrame]


def load_daily_news_features(
    path=DAILY_NEWS_FEATURES_PATH,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or "Asset" not in frame.columns:
        raise ValueError("Daily news features require Date and Asset columns.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame.duplicated(["Asset", "Date"]).any():
        raise ValueError("Daily news features must be unique by Asset and Date.")
    return frame.sort_values(["Asset", "Date"]).reset_index(drop=True)


def prepare_sentiment_substudy(
    *,
    daily_news: pd.DataFrame | None = None,
    start: str = SENTIMENT_STUDY_START,
    end: str = SENTIMENT_STUDY_END,
) -> dict[str, pd.DataFrame]:
    """Join frozen daily news features to each v6 asset on one common short window."""
    engineered, _ = prepare_multi_asset_engineered()
    news = daily_news.copy() if daily_news is not None else load_daily_news_features()
    news["Date"] = pd.to_datetime(news["Date"], errors="raise")
    required_news = [*NEWS_ATTENTION_FEATURES, *NEWS_SENTIMENT_FEATURES]
    missing_news = [column for column in required_news if column not in news.columns]
    if missing_news:
        raise ValueError(f"Missing daily news features: {missing_news}")

    outputs: dict[str, pd.DataFrame] = {}
    for asset, price_frame in engineered.items():
        asset_news = news.loc[news["Asset"] == asset, ["Date", *required_news]]
        merged = price_frame.merge(
            asset_news,
            on="Date",
            how="inner",
            validate="one_to_one",
        )
        merged = merged.loc[
            (merged["Date"] >= pd.Timestamp(start))
            & (merged["Date"] < pd.Timestamp(end))
        ].copy()
        eligible_required = [
            "Date",
            "Price",
            SUPPLEMENTARY_TARGET_RETURN_COLUMN,
            SUPPLEMENTARY_TARGET_DIRECTION_COLUMN,
            SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN,
            *SENTIMENT_BASE_FEATURES,
        ]
        merged = merged.dropna(subset=list(dict.fromkeys(eligible_required)))
        if len(merged) < 250:
            raise ValueError(
                f"{asset} has only {len(merged)} eligible short-window rows; "
                "at least 250 are required."
            )
        outputs[asset] = merged.sort_values("Date").reset_index(drop=True)

    common_dates = None
    for frame in outputs.values():
        dates = pd.Index(frame["Date"])
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or len(common_dates) < 250:
        raise ValueError("The three assets do not share a sufficiently long news window.")
    return {
        asset: frame.loc[frame["Date"].isin(common_dates)].reset_index(drop=True)
        for asset, frame in outputs.items()
    }


def feature_set_table() -> pd.DataFrame:
    descriptions = {
        "F0": "Technical + market index + macroeconomic context",
        "F1": "F0 + market-implied risk and survey-based sentiment proxies",
        "F2": "F0 + firm-news attention",
        "F3": "F0 + FinBERT textual sentiment",
        "F4": "F0 + attention + textual sentiment",
        "F5": "F0 + proxies + attention + textual sentiment",
    }
    return pd.DataFrame(
        [
            {
                "Feature Set": name,
                "Description": descriptions[name],
                "Feature Count": len(list(dict.fromkeys(features))),
                "Features": ", ".join(list(dict.fromkeys(features))),
            }
            for name, features in SENTIMENT_FEATURE_SETS.items()
        ]
    )


def _fold_labels(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target: str,
    *,
    large_move_quantile: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if target == TARGET_DIRECTION:
        return (
            frame.iloc[train_indices][SUPPLEMENTARY_TARGET_DIRECTION_COLUMN]
            .astype(int)
            .to_numpy(),
            frame.iloc[test_indices][SUPPLEMENTARY_TARGET_DIRECTION_COLUMN]
            .astype(int)
            .to_numpy(),
            None,
        )
    if target != TARGET_LARGE_MOVE:
        raise ValueError(f"Unsupported target: {target}")
    threshold = float(
        frame.iloc[train_indices][SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN].quantile(
            large_move_quantile
        )
    )
    y_train = (
        frame.iloc[train_indices][SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN].to_numpy()
        > threshold
    ).astype(int)
    y_test = (
        frame.iloc[test_indices][SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN].to_numpy()
        > threshold
    ).astype(int)
    return y_train, y_test, threshold


def _tabular_probability(
    frame: pd.DataFrame,
    features: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    *,
    random_state: int,
) -> np.ndarray:
    model = build_classifier(
        model_name,
        random_state=random_state,
        n_estimators=120,
        max_depth=5,
    )
    fitted = clone(model)
    fitted.fit(frame.iloc[train_indices][features], y_train)
    return fitted.predict_proba(frame.iloc[test_indices][features])[:, 1]


def _lstm_probability(
    frame: pd.DataFrame,
    features: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    lookback: int,
    units: int,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    from .lstm import fit_lstm_fold, prepare_sequence_fold

    labelled = frame.copy()
    # prepare_sequence_fold materialises the label array for the full frame,
    # although it selects labels only at train/test anchors. A deterministic
    # filler keeps purged and unused anchors castable without leaking a
    # fold-specific large-move threshold into model inputs.
    labelled["V7_Target_Label"] = 0
    labelled.loc[train_indices, "V7_Target_Label"] = y_train
    labelled.loc[test_indices, "V7_Target_Label"] = y_test
    fold = prepare_sequence_fold(
        labelled,
        features,
        train_indices,
        test_indices,
        lookback=lookback,
        target_column="V7_Target_Label",
    )
    models, probability, _, _ = fit_lstm_fold(
        fold,
        lookback=lookback,
        units=units,
        dropout=DEFAULT_LSTM_DROPOUT,
        epochs=epochs,
        batch_size=batch_size,
        seeds=(random_state,),
    )
    anchors = fold.test_anchor_indices
    del models, fold
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()
    return probability, anchors


def summarise_oof_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    if oof.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str | int]] = []
    group_columns = ["Asset", "Target", "Feature Set", "Model"]
    for keys, group in oof.groupby(group_columns, sort=False):
        group = group.sort_values("Date")
        metrics = _classification_metrics(
            group["Actual"],
            group["Predicted"],
            group["Probability"],
        )
        baseline_brier = float(
            np.mean((group["Actual"].to_numpy() - group["Baseline Probability"].to_numpy()) ** 2)
        )
        brier = metrics["Brier Score"]
        brier_skill = (
            float(1 - brier / baseline_brier)
            if baseline_brier > 0 and np.isfinite(brier)
            else np.nan
        )
        rows.append(
            {
                "Asset": keys[0],
                "Target": keys[1],
                "Feature Set": keys[2],
                "Model": keys[3],
                "OOF Rows": len(group),
                "OOF Start": group["Date"].min(),
                "OOF End": group["Date"].max(),
                "Positive Prevalence": float(group["Actual"].mean()),
                "Brier Skill": brier_skill,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_dual_target_benchmark(
    study_frames: dict[str, pd.DataFrame] | None = None,
    *,
    feature_sets: Iterable[str] = tuple(SENTIMENT_FEATURE_SETS),
    models: Iterable[str] = tuple(ALL_MODEL_NAMES),
    targets: Iterable[str] = tuple(TARGETS),
    n_splits: int = 4,
    probability_threshold: float = 0.50,
    large_move_quantile: float = 0.80,
    random_state: int = 42,
    include_lstm: bool = True,
    lstm_lookback: int = DEFAULT_LSTM_LOOKBACK,
    lstm_units: int = 8,
    lstm_epochs: int = 12,
    lstm_batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
) -> SentimentStudyResult:
    frames = study_frames or prepare_sentiment_substudy()
    model_names = [MAJORITY_MODEL_NAME, *list(models)]
    oof_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for asset, frame in frames.items():
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)
        splits = list(splitter.split(frame))
        for target in targets:
            fold_labels: dict[int, tuple[np.ndarray, np.ndarray, float | None]] = {}
            for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
                values = _fold_labels(
                    frame,
                    train_indices,
                    test_indices,
                    target,
                    large_move_quantile=large_move_quantile,
                )
                fold_labels[fold_number] = values
                threshold_rows.append(
                    {
                        "Asset": asset,
                        "Target": target,
                        "Fold": fold_number,
                        "Train Start": frame.iloc[train_indices]["Date"].min(),
                        "Train End": frame.iloc[train_indices]["Date"].max(),
                        "Purged Anchor Date": frame.iloc[
                            max(int(test_indices[0]) - 1, 0)
                        ]["Date"],
                        "Test Start": frame.iloc[test_indices]["Date"].min(),
                        "Test End": frame.iloc[test_indices]["Date"].max(),
                        "Large-Move Quantile": (
                            large_move_quantile if target == TARGET_LARGE_MOVE else np.nan
                        ),
                        "Large-Move Threshold": values[2],
                        "Train Positive Rate": float(values[0].mean()),
                        "Test Positive Rate": float(values[1].mean()),
                    }
                )
            for feature_set_name in feature_sets:
                features = list(
                    dict.fromkeys(SENTIMENT_FEATURE_SETS[str(feature_set_name)])
                )
                missing = [feature for feature in features if feature not in frame.columns]
                if missing:
                    raise ValueError(
                        f"{asset} {feature_set_name} is missing features: {missing}"
                    )
                for model_name in model_names:
                    if model_name == LSTM_MODEL_NAME and not include_lstm:
                        continue
                    for fold_number, (train_indices, test_indices) in enumerate(
                        splits, start=1
                    ):
                        y_train, y_test, threshold = fold_labels[fold_number]
                        if len(np.unique(y_train)) < 2:
                            continue
                        majority_class = int(pd.Series(y_train).mode().iloc[0])
                        baseline_probability = float(y_train.mean())
                        effective_test_indices = test_indices
                        if model_name == MAJORITY_MODEL_NAME:
                            probability = np.full(len(test_indices), baseline_probability)
                        elif model_name == LSTM_MODEL_NAME:
                            probability, effective_test_indices = _lstm_probability(
                                frame,
                                features,
                                train_indices,
                                test_indices,
                                y_train,
                                y_test,
                                lookback=lstm_lookback,
                                units=lstm_units,
                                epochs=lstm_epochs,
                                batch_size=lstm_batch_size,
                                random_state=random_state,
                            )
                        else:
                            probability = _tabular_probability(
                                frame,
                                features,
                                train_indices,
                                test_indices,
                                y_train,
                                model_name,
                                random_state=random_state,
                            )
                        effective_rows = frame.iloc[effective_test_indices]
                        effective_y_test = (
                            effective_rows[SUPPLEMENTARY_TARGET_DIRECTION_COLUMN]
                            .astype(int)
                            .to_numpy()
                            if target == TARGET_DIRECTION
                            else (
                                effective_rows[
                                    SUPPLEMENTARY_TARGET_ABS_RETURN_COLUMN
                                ].to_numpy()
                                > float(threshold)
                            ).astype(int)
                        )
                        prediction = (
                            np.full(len(probability), majority_class, dtype=int)
                            if model_name == MAJORITY_MODEL_NAME
                            else (probability >= probability_threshold).astype(int)
                        )
                        output = pd.DataFrame(
                            {
                                "Date": effective_rows["Date"].to_numpy(),
                                "Asset": asset,
                                "Target": target,
                                "Feature Set": str(feature_set_name),
                                "Model": model_name,
                                "Fold": fold_number,
                                "Train End": frame.iloc[train_indices]["Date"].max(),
                                "Actual": effective_y_test,
                                "Probability": probability,
                                "Predicted": prediction,
                                "Baseline Probability": baseline_probability,
                                "Large-Move Threshold": threshold,
                            }
                        )
                        oof_frames.append(output)

    oof = (
        pd.concat(oof_frames, ignore_index=True)
        .sort_values(["Asset", "Target", "Feature Set", "Model", "Date"])
        .reset_index(drop=True)
    )
    duplicate_keys = ["Asset", "Target", "Feature Set", "Model", "Date"]
    if oof.duplicated(duplicate_keys).any():
        raise ValueError("OOF predictions are not unique by experiment and date.")
    thresholds = (
        pd.DataFrame(threshold_rows)
        .drop_duplicates(["Asset", "Target", "Fold"])
        .sort_values(["Asset", "Target", "Fold"])
        .reset_index(drop=True)
    )
    summary = summarise_oof_predictions(oof)
    return SentimentStudyResult(
        summary=summary,
        oof_predictions=oof,
        fold_thresholds=thresholds,
        study_frames=frames,
    )


def paired_feature_set_deltas(
    summary: pd.DataFrame,
    *,
    reference_set: str = "F1",
    full_set: str = "F5",
) -> pd.DataFrame:
    metrics = ["Macro F1", "PR-AUC", "Balanced Accuracy", "MCC", "Brier Score"]
    left = summary.loc[summary["Feature Set"] == reference_set].copy()
    right = summary.loc[summary["Feature Set"] == full_set].copy()
    keys = ["Asset", "Target", "Model"]
    merged = left.merge(right, on=keys, suffixes=(f"_{reference_set}", f"_{full_set}"))
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        output: dict[str, object] = {
            key: row[key] for key in keys
        }
        output["Reference Set"] = reference_set
        output["Full Set"] = full_set
        for metric in metrics:
            if metric == "Brier Score":
                output[f"Delta {metric}"] = (
                    row[f"{metric}_{reference_set}"] - row[f"{metric}_{full_set}"]
                )
            else:
                output[f"Delta {metric}"] = (
                    row[f"{metric}_{full_set}"] - row[f"{metric}_{reference_set}"]
                )
        rows.append(output)
    return pd.DataFrame(rows)


def build_bootstrap_evidence(
    oof_predictions: pd.DataFrame,
    *,
    model_name: str = "Random Forest Classifier",
    reference_set: str = "F1",
    full_set: str = "F5",
    block_lengths: Iterable[int] = (5, 10, 20),
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build paired fold-aware CIs for firm-news value beyond broad proxies."""
    from .statistics import paired_moving_block_bootstrap

    rows: list[dict[str, object]] = []
    filtered = oof_predictions.loc[oof_predictions["Model"] == model_name]
    for (asset, target), group in filtered.groupby(["Asset", "Target"]):
        baseline = group.loc[group["Feature Set"] == reference_set].sort_values("Date")
        comparison = group.loc[group["Feature Set"] == full_set].sort_values("Date")
        paired = baseline[
            ["Date", "Fold", "Actual", "Probability"]
        ].merge(
            comparison[["Date", "Fold", "Actual", "Probability"]],
            on="Date",
            suffixes=("_Reference", "_Full"),
            validate="one_to_one",
        )
        if not (
            (paired["Actual_Reference"] == paired["Actual_Full"]).all()
            and (paired["Fold_Reference"] == paired["Fold_Full"]).all()
        ):
            raise ValueError("F1 and F5 OOF ledgers are not paired on target and fold.")
        metric = "Macro F1" if target == TARGET_DIRECTION else "PR-AUC"
        for block_length in block_lengths:
            evidence = paired_moving_block_bootstrap(
                paired["Actual_Reference"],
                paired["Probability_Reference"],
                paired["Probability_Full"],
                folds=paired["Fold_Reference"],
                metric=metric,
                n_bootstrap=n_bootstrap,
                block_length=int(block_length),
                random_state=random_state,
            )
            record = evidence.to_record()
            record.update(
                {
                    "Asset": asset,
                    "Target": target,
                    "Model": model_name,
                    "Reference Set": reference_set,
                    "Full Set": full_set,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _single_rf_large_move_oof(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    n_splits: int,
    random_state: int,
    large_move_quantile: float = 0.80,
) -> pd.DataFrame:
    features = list(dict.fromkeys(SENTIMENT_FEATURE_SETS[feature_set]))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)
    outputs: list[pd.DataFrame] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(frame), start=1):
        y_train, y_test, threshold = _fold_labels(
            frame,
            train_indices,
            test_indices,
            TARGET_LARGE_MOVE,
            large_move_quantile=large_move_quantile,
        )
        probability = _tabular_probability(
            frame,
            features,
            train_indices,
            test_indices,
            y_train,
            "Random Forest Classifier",
            random_state=random_state,
        )
        outputs.append(
            pd.DataFrame(
                {
                    "Date": frame.iloc[test_indices]["Date"].to_numpy(),
                    "Fold": fold,
                    "Actual": y_test,
                    "Probability": probability,
                    "Large-Move Threshold": threshold,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True).sort_values("Date").reset_index(drop=True)


def run_primary_placebo_test(
    tsla_frame: pd.DataFrame,
    observed_oof: pd.DataFrame,
    *,
    iterations: int = 200,
    block_length: int = 5,
    n_splits: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Falsify the pre-specified primary TSLA/RF/large-move/F5-vs-F1 comparison."""
    from .statistics import (
        joint_circular_block_shift,
        plus_one_empirical_p_value,
        probability_metric,
    )

    selected = observed_oof.loc[
        (observed_oof["Asset"] == "TSLA")
        & (observed_oof["Target"] == TARGET_LARGE_MOVE)
        & (observed_oof["Model"] == "Random Forest Classifier")
    ]
    reference = selected.loc[selected["Feature Set"] == "F1"].sort_values("Date")
    full = selected.loc[selected["Feature Set"] == "F5"].sort_values("Date")
    paired = reference[["Date", "Actual", "Probability"]].merge(
        full[["Date", "Actual", "Probability"]],
        on="Date",
        suffixes=("_Reference", "_Full"),
        validate="one_to_one",
    )
    reference_score = probability_metric(
        paired["Actual_Reference"],
        paired["Probability_Reference"],
        metric="PR-AUC",
    )
    full_score = probability_metric(
        paired["Actual_Full"],
        paired["Probability_Full"],
        metric="PR-AUC",
    )
    observed_delta = full_score - reference_score

    news_columns = list(
        dict.fromkeys([*NEWS_ATTENTION_FEATURES, *NEWS_SENTIMENT_FEATURES])
    )
    null_rows: list[dict[str, object]] = []
    for iteration in range(iterations):
        seed = random_state + iteration + 1
        shifted = joint_circular_block_shift(
            tsla_frame,
            news_columns,
            block_length=block_length,
            min_shift_blocks=4,
            random_state=seed,
        )
        placebo = _single_rf_large_move_oof(
            shifted.data,
            feature_set="F5",
            n_splits=n_splits,
            random_state=random_state,
        )
        aligned = reference[["Date", "Actual", "Probability"]].merge(
            placebo[["Date", "Actual", "Probability"]],
            on="Date",
            suffixes=("_Reference", "_Placebo"),
            validate="one_to_one",
        )
        placebo_score = probability_metric(
            aligned["Actual_Placebo"],
            aligned["Probability_Placebo"],
            metric="PR-AUC",
        )
        null_rows.append(
            {
                "Iteration": iteration + 1,
                "Random State": seed,
                "Shift Rows": int(shifted.audit.iloc[0]["Shift Rows"]),
                "Reference PR-AUC": reference_score,
                "Placebo F5 PR-AUC": placebo_score,
                "Placebo Delta PR-AUC": placebo_score - reference_score,
            }
        )
    distribution = pd.DataFrame(null_rows)
    p_value = plus_one_empirical_p_value(
        observed_delta,
        distribution["Placebo Delta PR-AUC"],
        alternative="greater",
    )
    summary = pd.DataFrame(
        [
            {
                "Asset": "TSLA",
                "Target": TARGET_LARGE_MOVE,
                "Model": "Random Forest Classifier",
                "Contrast": "F5 - F1",
                "Metric": "PR-AUC",
                "Observed Reference Score": reference_score,
                "Observed Full Score": full_score,
                "Observed Delta": observed_delta,
                "Null Mean Delta": distribution["Placebo Delta PR-AUC"].mean(),
                "Null 95th Percentile": distribution[
                    "Placebo Delta PR-AUC"
                ].quantile(0.95),
                "Empirical p-value": p_value,
                "Iterations": iterations,
                "Block Length": block_length,
                "Random State": random_state,
            }
        ]
    )
    return summary, distribution
