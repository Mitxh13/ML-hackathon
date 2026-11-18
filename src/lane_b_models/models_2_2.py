"""Section 2.2: Model implementations for Random Forest, Extra Trees, LightGBM, XGBoost, and MLP.

Trains, evaluates, and diagnoses the 5 high-capacity model families assigned to Section 2.2:
1. Random Forest (Ensemble / Bagging)
2. Extra Trees (Extremely Randomized Trees)
3. LightGBM (Gradient Boosting)
4. XGBoost (Extreme Gradient Boosting)
5. Multi-Layer Perceptron (Neural Network with epoch-wise loss tracking)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from src.contracts.schemas import CANDIDATE_BASE_COLS, MODEL_OUTPUT_COLS, validate_candidate_df, validate_model_output_df
from src.lane_c_hmm.metrics import compute_macro_f1, compute_log_loss_safe, compute_top1_accuracy


def get_section_2_2_models() -> dict[str, Any]:
    """Instantiate the 5 distinct model families for Section 2.2 candidate classification.

    Models:
    1. RandomForest: 150 trees, balanced subsample weighting to handle 87:13 class imbalance.
    2. ExtraTrees: 150 extremely randomized trees for lower variance and faster splits.
    3. LightGBM: Leaf-wise GBDT with histogram binning and scale_pos_weight.
    4. XGBoost: Exact/approximate second-order tree boosting with regularization.
    5. MLP: Multi-layer Perceptron (64, 32) with StandardScaler and Adam optimizer.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=6.8,  # Reflects ~87% negative to ~13% positive ratio
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=42,
                ),
            ),
        ]),
    }


def compute_observation_top1(df: pd.DataFrame, prob_col: str = "pred_prob") -> float:
    """Compute observation-level Top-1 accuracy by selecting candidate with highest prob."""
    correct = 0
    total = 0
    for _, group in df.groupby(["trip_id", "obs_idx"]):
        total += 1
        best_cand = group.loc[group[prob_col].idxmax()]
        if best_cand["is_target"] == 1:
            correct += 1
    return float(correct / max(total, 1))


class Section22Trainer:
    """Trainer and evaluator for Section 2.2 model families."""

    def __init__(self, models: Optional[dict[str, Any]] = None) -> None:
        self.models = models or get_section_2_2_models()
        self.feature_cols: list[str] = []
        self.fitted_models: dict[str, Any] = {}
        self.cv_results_: Optional[pd.DataFrame] = None
        self.test_results_: Optional[pd.DataFrame] = None
        self.learning_curves_: dict[str, list[float]] = {}

    def run_group_kfold_cv(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Run trip-grouped cross-validation across all 5 models.

        Args:
            train_df: Candidate DataFrame with 'feat_*' columns and 'is_target'.
            n_splits: Number of CV folds.

        Returns:
            Tuple of (summary_results_df, oof_predictions_dict).
        """
        validate_candidate_df(train_df)

        labeled_df = train_df[train_df["is_target"].isin([0, 1])].copy()
        self.feature_cols = [c for c in labeled_df.columns if c.startswith("feat_")]
        if not self.feature_cols:
            raise ValueError("No feature columns prefixed with 'feat_' found in train_df.")

        X = labeled_df[self.feature_cols].fillna(0.0).values
        y = labeled_df["is_target"].values.astype(int)
        groups = labeled_df["trip_id"].values

        unique_trips = len(np.unique(groups))
        actual_splits = min(n_splits, max(2, unique_trips))
        gkf = GroupKFold(n_splits=actual_splits)

        summary_records = []
        oof_dict: dict[str, np.ndarray] = {}

        print(f"\n=================================================================")
        print(f" Section 2.2: Running {actual_splits}-Fold GroupKFold Cross-Validation")
        print(f" Samples: {len(y):,} | Unique Trips: {unique_trips} | Features: {len(self.feature_cols)}")
        print(f"=================================================================")

        for model_name, model in self.models.items():
            print(f"\n---> Training & Cross-Validating: {model_name}...")
            t0 = time.time()
            oof_probs = np.zeros(len(y), dtype=float)

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model.fit(X_train, y_train)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)[:, 1]
                else:
                    probs = model.predict(X_val)

                oof_probs[val_idx] = probs

            cv_time = time.time() - t0
            oof_preds = (oof_probs >= 0.5).astype(int)
            oof_dict[model_name] = oof_probs

            # Compute CV Metrics
            macro_f1 = compute_macro_f1(y, oof_preds)
            logloss_val = compute_log_loss_safe(y, oof_probs)

            # Observation-level Top-1 Accuracy
            labeled_df["oof_prob"] = oof_probs
            top1_acc = compute_observation_top1(labeled_df, prob_col="oof_prob")

            print(
                f"     CV Macro F1: {macro_f1:.4f} | "
                f"CV Top-1 Acc: {top1_acc * 100:.2f}% | "
                f"CV Log Loss: {logloss_val:.4f} | "
                f"Time: {cv_time:.2f}s"
            )

            # Fit on entire training dataset for final deployment
            print(f"     Fitting final {model_name} on all {len(y):,} training rows...")
            model.fit(X, y)
            self.fitted_models[model_name] = model

            # Capture real loss curve if model produces it (e.g. MLP)
            if model_name == "MLP":
                mlp_estimator = model.named_steps["mlp"] if isinstance(model, Pipeline) else model
                if hasattr(mlp_estimator, "loss_curve_"):
                    self.learning_curves_["MLP"] = list(mlp_estimator.loss_curve_)

            summary_records.append(
                {
                    "Model": model_name,
                    "Algorithm Family": {
                        "RandomForest": "Ensemble (Bagging)",
                        "ExtraTrees": "Ensemble (Extremely Randomized)",
                        "LightGBM": "Gradient Boosting (Leaf-wise)",
                        "XGBoost": "Gradient Boosting (Exact)",
                        "MLP": "Neural Network (Feedforward)",
                    }.get(model_name, "Other"),
                    "Key Configuration": {
                        "RandomForest": "n_est=150, max_depth=12, class_weight=balanced_subsample",
                        "ExtraTrees": "n_est=150, max_depth=12, class_weight=balanced",
                        "LightGBM": "n_est=150, max_depth=6, lr=0.08, scale_pos_weight=balanced",
                        "XGBoost": "n_est=150, max_depth=6, lr=0.08, scale_pos_weight=6.8",
                        "MLP": "layers=(64,32), relu, adam, early_stopping=True",
                    }.get(model_name, ""),
                    "CV Macro F1": macro_f1,
                    "CV Top-1 Accuracy": top1_acc,
                    "CV Log Loss": logloss_val,
                    "CV Runtime (s)": cv_time,
                }
            )

        self.cv_results_ = pd.DataFrame(summary_records)
        return self.cv_results_, oof_dict

    def evaluate_student_test(
        self,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Evaluate fitted models on the student_test partition.

        Args:
            test_df: Student test candidate DataFrame with ground truth 'is_target'.

        Returns:
            Tuple of (test_comparison_df, scored_test_dfs_dict).
        """
        validate_candidate_df(test_df)
        labeled_test = test_df[test_df["is_target"].isin([0, 1])].copy()
        X_test = labeled_test[self.feature_cols].fillna(0.0).values
        y_test = labeled_test["is_target"].values.astype(int)

        records = []
        scored_dfs = {}

        print(f"\n=================================================================")
        print(f" Section 2.2: Evaluating on Student-Testing Partition")
        print(f" Observations: {labeled_test.groupby(['trip_id', 'obs_idx']).ngroups:,} | "
              f"Candidate Rows: {len(y_test):,}")
        print(f"=================================================================")

        for model_name, model in self.fitted_models.items():
            t0 = time.time()
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.predict(X_test)
            inf_time = time.time() - t0

            preds = (probs >= 0.5).astype(int)

            scored_df = labeled_test.copy()
            scored_df["pred_prob"] = np.clip(probs, 0.0, 1.0)
            scored_df["pred_label"] = preds
            validate_model_output_df(scored_df)
            scored_dfs[model_name] = scored_df

            # Metrics
            test_macro_f1 = compute_macro_f1(y_test, preds)
            test_logloss = compute_log_loss_safe(y_test, probs)
            test_top1 = compute_observation_top1(scored_df, prob_col="pred_prob")

            print(
                f" {model_name:12s} | "
                f"Top-1: {test_top1 * 100:.2f}% | "
                f"Macro F1: {test_macro_f1:.4f} | "
                f"Log Loss: {test_logloss:.4f} | "
                f"Inference Time: {inf_time:.3f}s"
            )

            records.append(
                {
                    "Model": model_name,
                    "Student-Test Top-1 Accuracy": test_top1,
                    "Student-Test Macro F1": test_macro_f1,
                    "Student-Test Log Loss": test_logloss,
                    "Student-Test Inference Time (s)": inf_time,
                }
            )

        self.test_results_ = pd.DataFrame(records)
        return self.test_results_, scored_dfs
