"""Bayesian hyperparameter optimization using Optuna with trip-grouped cross-validation (§2.3 & §4).

This module implements:
1. Optuna Bayesian optimization for champion gradient-boosted decision trees (LightGBM).
2. Strict GroupKFold cross-validation grouped by trip_id to prevent data leakage.
3. Full trial history logging adhering to Report Section 4 specifications.
4. Comprehensive diagnostic plotting (optimization history, parameter importances, ROC, before/after).
5. Champion model fine-tuning and evaluation against the student_test partition.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold

from src.contracts.schemas import validate_candidate_df
from src.lane_b_models.export_model import export_model_bundle
from src.lane_b_models.models_2_2 import compute_observation_top1
from src.lane_c_hmm.metrics import (
    compute_log_loss_safe,
    compute_macro_f1,
    compute_top1_accuracy,
)

logger = logging.getLogger(__name__)


def optimize_lightgbm(
    candidates_df: pd.DataFrame,
    n_trials: int = 25,
    n_splits: int = 5,
    log_file: Optional[str] = "artifacts/logs/bayesian_trials.csv",
    random_state: int = 42,
    study_name: str = "lightgbm_bayesian_tuning",
) -> Tuple[dict[str, Any], float, optuna.Study]:
    """Run Bayesian hyperparameter search for LightGBM using trip-grouped cross-validation.

    Anti-leakage:
        Folds are partitioned strictly on 'trip_id' via GroupKFold so that points
        from the same trip never appear across both train and validation folds.

    Objective:
        Maximizes validation performance (unweighted blend of candidate Macro F1 and
        observation-level Top-1 accuracy) with log loss penalty.

    Args:
        candidates_df: Candidates DataFrame with 'feat_*' columns, 'trip_id', and 'is_target'.
        n_trials: Number of Optuna optimization trials to execute.
        n_splits: Number of CV folds for GroupKFold.
        log_file: Path to save the detailed trial history CSV.
        random_state: Seed for reproducibility.
        study_name: Name identifier for the Optuna study.

    Returns:
        Tuple of (best_params, best_score, study).
    """
    validate_candidate_df(candidates_df)

    labeled_df = candidates_df[candidates_df["is_target"].isin([0, 1])].copy()
    if labeled_df.empty:
        raise ValueError("No labeled samples (is_target in {0, 1}) found for Bayesian optimization.")

    feature_cols = [c for c in labeled_df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns prefixed with 'feat_' found in candidates_df.")

    X = labeled_df[feature_cols].fillna(0.0).values
    y = labeled_df["is_target"].values
    groups = labeled_df["trip_id"].values
    obs_ids = labeled_df["obs_idx"].values

    unique_trips = len(np.unique(groups))
    actual_splits = min(n_splits, max(2, unique_trips))
    gkf = GroupKFold(n_splits=actual_splits)

    # Pre-compute fold indices for reproducibility and speed
    splits = list(gkf.split(X, y, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        # Search space encompassing tree depth, capacity, learning dynamics, and regularization
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320, step=40),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 14.0),
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        fold_f1s = []
        fold_top1s = []
        fold_losses = []

        for train_idx, val_idx in splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            obs_val = obs_ids[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            f1 = compute_macro_f1(y_val, preds)
            loss = compute_log_loss_safe(y_val, probs)

            # Observation top-1 calculation on validation fold
            fold_df = pd.DataFrame({
                "trip_id": groups[val_idx],
                "obs_idx": obs_val,
                "is_target": y_val,
                "pred_prob": probs,
            })
            top1 = compute_observation_top1(fold_df)

            fold_f1s.append(f1)
            fold_top1s.append(top1)
            fold_losses.append(loss)

        mean_f1 = float(np.mean(fold_f1s))
        mean_top1 = float(np.mean(fold_top1s))
        mean_loss = float(np.mean(fold_losses))

        trial.set_user_attr("mean_macro_f1", mean_f1)
        trial.set_user_attr("mean_top1", mean_top1)
        trial.set_user_attr("mean_log_loss", mean_loss)

        # Composite score optimizing both top-1 arc retrieval and macro F1
        composite_score = 0.6 * mean_top1 + 0.4 * mean_f1
        print(f"[Optuna] Trial {trial.number:2d}: Score={composite_score:.4f} (Top-1={mean_top1:.4f}, F1={mean_f1:.4f}, Loss={mean_loss:.4f})", flush=True)
        return composite_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        trials_df = study.trials_dataframe()
        trials_df.to_csv(log_path, index=False)
        logger.info("Saved %d Optuna trials to %s", len(trials_df), log_path)

    return study.best_params, study.best_value, study


def plot_optimization_history(
    study: optuna.Study,
    save_path: Optional[str] = "results/figures/bayesian_opt_history.png",
) -> plt.Figure:
    """Plot objective value progression across Optuna optimization trials."""
    trials = [t for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    best_values = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trial_numbers, values, color="#3498db", alpha=0.7, edgecolors="none", s=40, label="Trial Score")
    ax.plot(trial_numbers, best_values, color="#e74c3c", linewidth=2.2, label="Best Cumulative Score")

    ax.set_title("Bayesian Hyperparameter Optimization History (Optuna TPE)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel("Composite Validation Score (0.6 Top-1 + 0.4 Macro F1)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_param_importances(
    study: optuna.Study,
    save_path: Optional[str] = "results/figures/bayesian_param_importances.png",
) -> plt.Figure:
    """Plot relative hyperparameter importances computed via Optuna fANOVA/MDI."""
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {k: 1.0 / len(study.best_params) for k in study.best_params.keys()}

    params = list(importances.keys())
    scores = list(importances.values())

    # Sort ascending for horizontal bar chart
    sorted_indices = np.argsort(scores)
    sorted_params = [params[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette("mako", len(sorted_params))
    bars = ax.barh(sorted_params, sorted_scores, color=palette, edgecolor="none", height=0.65)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=9)

    ax.set_title("Hyperparameter Importance Ranking (LightGBM Tuning)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Weight", fontsize=11)
    ax.set_xlim(0, max(sorted_scores) * 1.18 if sorted_scores else 1.0)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_tuning_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = "results/figures/tuning_before_after_comparison.png",
) -> plt.Figure:
    """Plot grouped comparison chart of default vs tuned LightGBM across metrics."""
    metrics_to_plot = [
        ("cv_top1_accuracy", "CV Top-1 Acc"),
        ("student_test_top1_accuracy", "Test Top-1 Acc"),
        ("student_test_macro_f1", "Test Macro F1"),
        ("student_test_log_loss", "Test Log Loss"),
    ]

    available_metrics = [(col, label) for col, label in metrics_to_plot if col in comparison_df.columns]

    records = []
    for _, row in comparison_df.iterrows():
        config_name = row["configuration"]
        for col, label in available_metrics:
            records.append({
                "Configuration": config_name,
                "Metric": label,
                "Value": row[col],
            })

    plot_data = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"Default LightGBM": "#7f8c8d", "Bayesian-Tuned LightGBM": "#27ae60"}
    sns.barplot(data=plot_data, x="Metric", y="Value", hue="Configuration", palette=palette, ax=ax)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_title("LightGBM Performance: Baseline vs. Bayesian Tuned (§4)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_ylim(0, max(plot_data["Value"].max() * 1.15, 1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_roc_comparison(
    y_test: np.ndarray,
    default_probs: np.ndarray,
    tuned_probs: np.ndarray,
    save_path: Optional[str] = "results/figures/roc_curve_champion.png",
) -> plt.Figure:
    """Plot ROC curves comparing default vs. tuned champion classifier."""
    fpr_def, tpr_def, _ = roc_curve(y_test, default_probs)
    auc_def = auc(fpr_def, tpr_def)

    fpr_tun, tpr_tun, _ = roc_curve(y_test, tuned_probs)
    auc_tun = auc(fpr_tun, tpr_tun)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_def, tpr_def, color="#7f8c8d", linestyle="--", linewidth=2.0, label=f"Default LightGBM (AUC = {auc_def:.4f})")
    ax.plot(fpr_tun, tpr_tun, color="#27ae60", linewidth=2.4, label=f"Bayesian-Tuned LightGBM (AUC = {auc_tun:.4f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", alpha=0.4, label="Random Guess (AUC = 0.5000)")

    ax.set_title("Receiver Operating Characteristic (ROC) - Champion Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def evaluate_tuning_improvement(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict[str, Any],
    default_params: Optional[dict[str, Any]] = None,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Any, Any]:
    """Train, cross-validate, and benchmark both default and tuned LightGBM models.

    Computes Report Section 4 comparison metrics strictly through src.lane_c_hmm.metrics.
    """
    if default_params is None:
        default_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 10.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

    feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
    X_train = train_df[feature_cols].fillna(0.0).values
    y_train = train_df["is_target"].values
    groups_train = train_df["trip_id"].values
    obs_train = train_df["obs_idx"].values

    X_test = test_df[feature_cols].fillna(0.0).values
    y_test = test_df["is_target"].values
    obs_test = test_df["obs_idx"].values

    unique_trips = len(np.unique(groups_train))
    actual_splits = min(n_splits, max(2, unique_trips))
    gkf = GroupKFold(n_splits=actual_splits)

    records = []
    models = {
        "Default LightGBM": lgb.LGBMClassifier(**default_params),
        "Bayesian-Tuned LightGBM": lgb.LGBMClassifier(**best_params),
    }

    trained_instances = {}

    for name, clf in models.items():
        # 1. 5-Fold GroupKFold CV
        oof_probs = np.zeros(len(y_train), dtype=float)
        oof_preds = np.zeros(len(y_train), dtype=int)

        t_cv_start = time.time()
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups=groups_train):
            clf_fold = lgb.LGBMClassifier(**clf.get_params())
            clf_fold.fit(X_train[tr_idx], y_train[tr_idx])
            probs_va = clf_fold.predict_proba(X_train[va_idx])[:, 1]
            oof_probs[va_idx] = probs_va
            oof_preds[va_idx] = (probs_va >= 0.5).astype(int)
        cv_time = time.time() - t_cv_start

        cv_macro_f1 = compute_macro_f1(y_train, oof_preds)
        cv_log_loss = compute_log_loss_safe(y_train, oof_probs)

        cv_df = pd.DataFrame({
            "trip_id": groups_train,
            "obs_idx": obs_train,
            "is_target": y_train,
            "pred_prob": oof_probs,
        })
        cv_top1 = compute_observation_top1(cv_df)

        # 2. Fit full training partition
        t_fit_start = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t_fit_start
        trained_instances[name] = clf

        # 3. Test on held-out student_test partition
        t_infer_start = time.time()
        test_probs = clf.predict_proba(X_test)[:, 1]
        infer_time = time.time() - t_infer_start
        test_preds = (test_probs >= 0.5).astype(int)

        test_macro_f1 = compute_macro_f1(y_test, test_preds)
        test_log_loss = compute_log_loss_safe(y_test, test_probs)

        test_scored_df = pd.DataFrame({
            "trip_id": test_df["trip_id"].values,
            "obs_idx": obs_test,
            "is_target": y_test,
            "pred_prob": test_probs,
        })
        test_top1 = compute_observation_top1(test_scored_df)

        records.append({
            "configuration": name,
            "cv_top1_accuracy": round(cv_top1, 4),
            "cv_macro_f1": round(cv_macro_f1, 4),
            "cv_log_loss": round(cv_log_loss, 4),
            "student_test_top1_accuracy": round(test_top1, 4),
            "student_test_macro_f1": round(test_macro_f1, 4),
            "student_test_log_loss": round(test_log_loss, 4),
            "fit_time_sec": round(fit_time, 3),
            "inference_time_sec": round(infer_time, 4),
        })

    comparison_df = pd.DataFrame(records)
    return comparison_df, trained_instances["Default LightGBM"], trained_instances["Bayesian-Tuned LightGBM"]
