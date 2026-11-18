"""Unit tests verifying Section 2.2 model implementations, GroupKFold CV, and Contract 2."""

import numpy as np
import pandas as pd
import pytest

from src.contracts.schemas import validate_model_output_df
from src.lane_b_models.models_2_2 import (
    get_section_2_2_models,
    Section22Trainer,
    compute_observation_top1,
)
from src.lane_b_models.export_model import export_model_bundle, load_model_bundle


@pytest.fixture
def synthetic_training_candidates():
    """Create synthetic candidate DataFrame across 4 distinct trips."""
    records = []
    trips = ["trip_01", "trip_02", "trip_03", "trip_04"]

    for t_idx, trip_id in enumerate(trips):
        for obs_idx in range(6):
            true_k = 0
            for k in range(4):
                is_target = 1 if k == true_k else 0
                records.append(
                    {
                        "trip_id": trip_id,
                        "obs_idx": obs_idx,
                        "timestamp": float(obs_idx * 1.0),
                        "gps_lon": 77.50 + obs_idx * 0.001,
                        "gps_lat": 12.90 + obs_idx * 0.001,
                        "candidate_arc_id": obs_idx * 10 + k,
                        "from_node_id": obs_idx * 10 + k,
                        "to_node_id": obs_idx * 10 + k + 1,
                        "dist_to_arc_m": 2.0 if is_target else 25.0 + k * 5,
                        "is_target": is_target,
                        # Features
                        "feat_dist_to_arc_m": 2.0 if is_target else 25.0 + k * 5,
                        "feat_dist_sq": (2.0 if is_target else 25.0 + k * 5) ** 2,
                        "feat_heading_diff_rad": 0.05 if is_target else 0.8,
                        "feat_heading_cos_sim": 0.99 if is_target else 0.2,
                        "feat_speed_mps": 12.5,
                        "feat_rank_by_dist": float(k),
                        "feat_candidate_count": 4.0,
                    }
                )

    return pd.DataFrame(records)


def test_section_2_2_model_instantiation():
    """Verify that all 5 model families are instantiated with valid hyperparameters."""
    models = get_section_2_2_models()
    expected_names = {"RandomForest", "ExtraTrees", "LightGBM", "XGBoost", "MLP"}
    assert set(models.keys()) == expected_names

    # Check key configurations
    assert models["RandomForest"].n_estimators == 150
    assert models["ExtraTrees"].n_estimators == 150
    assert models["LightGBM"].n_estimators == 150
    assert models["XGBoost"].n_estimators == 150


def test_section_2_2_cv_and_test(synthetic_training_candidates, tmp_path):
    """Verify GroupKFold CV and test evaluation on synthetic data."""
    # Split synthetic data into train (3 trips) and test (1 trip)
    train_df = synthetic_training_candidates[
        synthetic_training_candidates["trip_id"] != "trip_04"
    ].copy()
    test_df = synthetic_training_candidates[
        synthetic_training_candidates["trip_id"] == "trip_04"
    ].copy()

    # Fast models for test
    from sklearn.ensemble import ExtraTreesClassifier
    fast_models = {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=10, random_state=42),
    }

    trainer = Section22Trainer(models=fast_models)
    cv_df, oof_dict = trainer.run_group_kfold_cv(train_df, n_splits=3)

    assert len(cv_df) == 1
    assert "CV Macro F1" in cv_df.columns
    assert "CV Top-1 Accuracy" in cv_df.columns
    assert cv_df.iloc[0]["CV Top-1 Accuracy"] > 0.5

    # Test evaluation
    test_df_results, scored_dfs = trainer.evaluate_student_test(test_df)
    assert len(test_df_results) == 1
    assert "Student-Test Top-1 Accuracy" in test_df_results.columns

    # Verify Contract 2 compliance on output
    scored_extra_trees = scored_dfs["ExtraTrees"]
    assert validate_model_output_df(scored_extra_trees) is True
    assert ((scored_extra_trees["pred_prob"] >= 0.0) & (scored_extra_trees["pred_prob"] <= 1.0)).all()

    # Verify artifact export and reload
    export_file = tmp_path / "best_model.joblib"
    export_model_bundle(
        model=trainer.fitted_models["ExtraTrees"],
        feature_names=trainer.feature_cols,
        save_path=str(export_file),
        metadata={"model_name": "ExtraTrees"},
    )
    loaded = load_model_bundle(str(export_file))
    assert loaded["metadata"]["model_name"] == "ExtraTrees"
    assert loaded["feature_names"] == trainer.feature_cols


def test_compute_observation_top1():
    """Verify top1 calculation assigns point correctly based on highest probability."""
    sample_data = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1"],
            "obs_idx": [0, 0, 1, 1],
            "pred_prob": [0.8, 0.2, 0.3, 0.7],
            "is_target": [1, 0, 0, 1],
        }
    )
    acc = compute_observation_top1(sample_data, prob_col="pred_prob")
    assert acc == 1.0

    # If predicted incorrectly
    bad_data = pd.DataFrame(
        {
            "trip_id": ["t1", "t1"],
            "obs_idx": [0, 0],
            "pred_prob": [0.1, 0.9],
            "is_target": [1, 0],
        }
    )
    assert compute_observation_top1(bad_data, prob_col="pred_prob") == 0.0
