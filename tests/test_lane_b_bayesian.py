"""Unit tests for Section 2.3 Bayesian hyperparameter optimization and model export."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pytest

from src.contracts.schemas import validate_model_output_df
from src.lane_b_models.bayesian_opt import (
    evaluate_tuning_improvement,
    optimize_lightgbm,
    plot_optimization_history,
    plot_param_importances,
    plot_roc_comparison,
    plot_tuning_comparison,
)
from src.lane_b_models.export_model import export_model_bundle, load_model_bundle


@pytest.fixture
def synthetic_candidate_dataset() -> pd.DataFrame:
    """Create a valid synthetic candidate DataFrame adhering to Contract 1 across 4 trips."""
    records = []
    trips = ["trip_01", "trip_02", "trip_03", "trip_04"]

    for t_idx, trip_id in enumerate(trips):
        for obs_idx in range(6):
            true_k = 0
            for k in range(4):
                is_target = 1 if k == true_k else 0
                dist = 2.0 if is_target else 25.0 + k * 5
                records.append({
                    "trip_id": trip_id,
                    "obs_idx": obs_idx,
                    "timestamp": float(obs_idx * 1.0),
                    "gps_lon": 77.50 + obs_idx * 0.001,
                    "gps_lat": 12.90 + obs_idx * 0.001,
                    "candidate_arc_id": obs_idx * 10 + k,
                    "from_node_id": obs_idx * 10 + k,
                    "to_node_id": obs_idx * 10 + k + 1,
                    "dist_to_arc_m": dist,
                    "is_target": is_target,
                    # Features
                    "feat_dist_to_arc_m": dist,
                    "feat_dist_sq": dist ** 2,
                    "feat_heading_diff_rad": 0.05 if is_target else 0.8,
                    "feat_heading_cos_sim": 0.99 if is_target else 0.2,
                    "feat_speed_mps": 12.5,
                    "feat_rank_by_dist": float(k),
                    "feat_candidate_count": 4.0,
                })

    return pd.DataFrame(records)


def test_optimize_lightgbm_smoke(synthetic_candidate_dataset: pd.DataFrame, tmp_path: Path):
    """Test that optimize_lightgbm runs multi-trial GroupKFold search and logs trials."""
    log_file = tmp_path / "test_trials.csv"

    best_params, best_score, study = optimize_lightgbm(
        candidates_df=synthetic_candidate_dataset,
        n_trials=3,
        n_splits=2,
        log_file=str(log_file),
        random_state=42,
    )

    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "n_estimators" in best_params
    assert "num_leaves" in best_params
    assert isinstance(best_score, float)
    assert len(study.trials) == 3

    assert log_file.exists()
    trials_df = pd.read_csv(log_file)
    assert len(trials_df) == 3
    assert "value" in trials_df.columns
    assert "params_learning_rate" in trials_df.columns


def test_export_and_load_model_bundle(synthetic_candidate_dataset: pd.DataFrame, tmp_path: Path):
    """Test model serialization, metadata capture, and deserialization."""
    from lightgbm import LGBMClassifier

    feature_cols = [c for c in synthetic_candidate_dataset.columns if c.startswith("feat_")]
    X = synthetic_candidate_dataset[feature_cols].values
    y = synthetic_candidate_dataset["is_target"].values

    clf = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    clf.fit(X, y)

    save_path = tmp_path / "best_model.joblib"
    metadata = {"accuracy": 0.95, "algorithm": "LightGBM"}

    exported = export_model_bundle(
        model=clf,
        feature_names=feature_cols,
        save_path=str(save_path),
        metadata=metadata,
    )
    assert exported.exists()

    loaded = load_model_bundle(str(save_path))
    assert "model" in loaded
    assert "feature_names" in loaded
    assert "metadata" in loaded
    assert loaded["feature_names"] == feature_cols
    assert loaded["metadata"]["algorithm"] == "LightGBM"

    # Test that deserialized model predicts correctly
    preds = loaded["model"].predict(X[:5])
    assert len(preds) == 5


def test_contract_2_model_output(synthetic_candidate_dataset: pd.DataFrame):
    """Test that scored predictions adhere to Contract 2 schema."""
    from lightgbm import LGBMClassifier

    feature_cols = [c for c in synthetic_candidate_dataset.columns if c.startswith("feat_")]
    X = synthetic_candidate_dataset[feature_cols].values
    y = synthetic_candidate_dataset["is_target"].values

    clf = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    clf.fit(X, y)

    scored_df = synthetic_candidate_dataset.copy()
    probs = clf.predict_proba(X)[:, 1]
    scored_df["pred_prob"] = np.clip(probs, 0.0, 1.0)
    scored_df["pred_label"] = (probs >= 0.5).astype(int)

    # Validates Contract 2 (throws ValueError on violation)
    validate_model_output_df(scored_df)
    assert "pred_prob" in scored_df.columns
    assert "pred_label" in scored_df.columns


def test_diagnostic_plots(synthetic_candidate_dataset: pd.DataFrame, tmp_path: Path):
    """Test that diagnostic plotting functions execute and save valid figures."""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("lr", 0.01, 0.1) * 2.0, n_trials=4)

    # 1. Opt History
    plot_optimization_history(study, save_path=str(tmp_path / "opt_hist.png"))
    assert (tmp_path / "opt_hist.png").exists()

    # 2. Param Importance
    plot_param_importances(study, save_path=str(tmp_path / "param_imp.png"))
    assert (tmp_path / "param_imp.png").exists()

    # 3. Tuning comparison
    comp_df = pd.DataFrame([
        {
            "configuration": "Default LightGBM",
            "cv_top1_accuracy": 0.70,
            "student_test_top1_accuracy": 0.705,
            "student_test_macro_f1": 0.82,
            "student_test_log_loss": 0.21,
        },
        {
            "configuration": "Bayesian-Tuned LightGBM",
            "cv_top1_accuracy": 0.72,
            "student_test_top1_accuracy": 0.725,
            "student_test_macro_f1": 0.84,
            "student_test_log_loss": 0.19,
        },
    ])
    plot_tuning_comparison(comp_df, save_path=str(tmp_path / "comp.png"))
    assert (tmp_path / "comp.png").exists()

    # 4. ROC curve
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    p1 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.2])
    p2 = np.array([0.05, 0.95, 0.15, 0.85, 0.2, 0.8, 0.3, 0.1])
    plot_roc_comparison(y_true, p1, p2, save_path=str(tmp_path / "roc.png"))
    assert (tmp_path / "roc.png").exists()
