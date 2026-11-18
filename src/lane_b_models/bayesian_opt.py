"""Bayesian hyperparameter optimization using Optuna with trip-grouped cross-validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.contracts.schemas import validate_candidate_df
from src.lane_c_hmm.metrics import compute_macro_f1


def optimize_lightgbm(
    candidates_df: pd.DataFrame,
    n_trials: int = 20,
    n_splits: int = 5,
    log_file: Optional[str] = "artifacts/logs/bayesian_trials.csv",
) -> Tuple[dict[str, Any], float, optuna.Study]:
    """Run Bayesian hyperparameter search for LightGBM using GroupKFold.

    Args:
        candidates_df: Candidates DataFrame with 'feat_*' columns and 'is_target'.
        n_trials: Number of Optuna optimization trials.
        n_splits: Number of CV folds for GroupKFold.
        log_file: Path to save trials history CSV.

    Returns:
        Tuple of (best_params, best_score, study).
    """
    validate_candidate_df(candidates_df)

    labeled_df = candidates_df[candidates_df["is_target"].isin([0, 1])].copy()
    feature_cols = [c for c in labeled_df.columns if c.startswith("feat_")]
    X = labeled_df[feature_cols].fillna(0.0).values
    y = labeled_df["is_target"].values
    groups = labeled_df["trip_id"].values

    unique_trips = len(np.unique(groups))
    actual_splits = min(n_splits, max(2, unique_trips))
    gkf = GroupKFold(n_splits=actual_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "verbose": -1,
        }

        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            f1 = compute_macro_f1(y_val, preds)
            fold_scores.append(f1)

        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Save trial history
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(log_path, index=False)

    return study.best_params, study.best_value, study
