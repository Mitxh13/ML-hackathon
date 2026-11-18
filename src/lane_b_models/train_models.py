"""Model training across >=5 distinct algorithm families with trip-grouped cross validation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_c_hmm.metrics import compute_macro_f1, compute_log_loss_safe


def get_default_models() -> dict[str, Any]:
    """Instantiate >=5 distinct model families for candidate classification.

    Families:
    1. Linear / Logistic Regression
    2. Random Forest
    3. Extra Trees
    4. LightGBM
    5. XGBoost
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss"),
    }


def train_and_evaluate_models(
    candidates_df: pd.DataFrame,
    n_splits: int = 5,
    models: Optional[dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Train >=5 models using GroupKFold by trip_id, evaluating out-of-fold metrics.

    Anti-leakage: GroupKFold ensures observations from the same trip never appear
    in both train and validation folds.

    Args:
        candidates_df: Candidate DataFrame with 'feat_*' columns, 'is_target', and 'trip_id'.
        n_splits: Number of cross-validation folds.
        models: Dictionary of model name to estimator instance.

    Returns:
        Tuple of (comparison_results_df, trained_models_dict).
    """
    validate_candidate_df(candidates_df)

    # Filter out unlabeled rows (-1)
    labeled_df = candidates_df[candidates_df["is_target"].isin([0, 1])].copy()
    if labeled_df.empty:
        raise ValueError("No labeled samples (is_target in {0, 1}) found for training.")

    feature_cols = [c for c in labeled_df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns prefixed with 'feat_' found in candidates_df.")

    X = labeled_df[feature_cols].fillna(0.0).values
    y = labeled_df["is_target"].values
    groups = labeled_df["trip_id"].values

    if models is None:
        models = get_default_models()

    # Determine safe number of splits
    unique_trips = len(np.unique(groups))
    actual_splits = min(n_splits, unique_trips)
    if actual_splits < 2:
        actual_splits = 2

    gkf = GroupKFold(n_splits=actual_splits)
    comparison_records = []
    fitted_models = {}

    for model_name, model in models.items():
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model.fit(X_train, y_train)

            # Predict probabilities if supported
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = model.predict(X_val)

            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.5).astype(int)

        # Fit model on all data for export
        model.fit(X, y)
        fitted_models[model_name] = model

        # Compute CV metrics
        macro_f1 = compute_macro_f1(y, oof_preds)
        loss = compute_log_loss_safe(y, oof_probs)

        # Observation-level accuracy (candidate with highest prob predicted as true)
        labeled_df["oof_prob"] = oof_probs
        top1_acc = 0.0
        obs_groups = labeled_df.groupby(["trip_id", "obs_idx"])
        correct_top1 = 0
        total_obs = len(obs_groups)

        for _, group in obs_groups:
            pred_candidate = group.loc[group["oof_prob"].idxmax()]
            if pred_candidate["is_target"] == 1:
                correct_top1 += 1

        if total_obs > 0:
            top1_acc = correct_top1 / total_obs

        comparison_records.append(
            {
                "model_family": model_name,
                "cv_macro_f1": macro_f1,
                "cv_top1_accuracy": top1_acc,
                "cv_log_loss": loss,
                "n_folds": actual_splits,
            }
        )

    results_df = pd.DataFrame(comparison_records)
    return results_df, fitted_models
