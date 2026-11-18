"""Focused tests for leakage-safe Section 2.1 setup and baseline CV."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.lane_b_models.training_setup import (
    build_baseline_pipeline,
    make_group_folds,
    observation_top1_accuracy,
    prepare_training_data,
    run_baseline_cross_validation,
    score_student_candidates,
)


def candidate_fixture(n_trips=10):
    rows = []
    for trip in range(n_trips):
        for obs in range(4):
            for candidate in range(2):
                target = int(candidate == 0)
                rows.append({"trip_id": f"trip-{trip}", "obs_idx": obs, "candidate_arc_id": candidate,
                    "from_node_id": candidate, "to_node_id": candidate + 1,
                    "timestamp": float(obs), "gps_lon": 1., "gps_lat": 2., "dist_to_arc_m": float(candidate),
                    "is_target": target, "feat_distance": float(candidate + target + trip * .01),
                    "feat_heading": np.nan if obs == 0 else float(obs - candidate)})
        rows.append({"trip_id": f"trip-{trip}", "obs_idx": 9, "candidate_arc_id": 7,
            "from_node_id": 1, "to_node_id": 2, "timestamp": 9., "gps_lon": 1., "gps_lat": 2.,
            "dist_to_arc_m": 20., "is_target": -1, "feat_distance": np.nan, "feat_heading": np.nan})
    return pd.DataFrame(rows)


def test_preparation_excludes_unlabelled_and_preserves_candidate_ids():
    frame = candidate_fixture()
    data = prepare_training_data(frame)
    assert len(data.X) == (len(frame) - 10)
    assert data.report["unlabelled_rows"] == 10
    assert data.feature_columns == ["feat_distance", "feat_heading"]
    assert data.identifiers.candidate_arc_id.notna().all()
    assert data.report["trip_count"] == 10
    assert data.report["features_with_missing_values"] == 1


def test_group_folds_are_fixed_five_and_trip_disjoint():
    data = prepare_training_data(candidate_fixture())
    folds = make_group_folds(data)
    assert len(folds) == 5
    for train, val in folds:
        assert not set(data.groups.iloc[train]) & set(data.groups.iloc[val])


def test_baseline_pipelines_fit_imputation_and_scaling_inside_pipeline():
    for name in ("LogisticRegression", "KNN"):
        pipe = build_baseline_pipeline(name)
        assert isinstance(pipe.named_steps["imputer"], SimpleImputer)
        assert pipe.named_steps["imputer"].strategy == "median"
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    nb = build_baseline_pipeline("GaussianNB")
    assert nb.named_steps["classifier"].priors is None


def test_all_three_baselines_emit_oof_probabilities_and_fold_metrics():
    metrics, predictions, fitted = run_baseline_cross_validation(candidate_fixture())
    assert set(metrics.model_name) == {"LogisticRegression", "GaussianNB", "KNN"}
    assert len(metrics) == 15
    assert len(predictions) == 3 * 10 * 4 * 2
    assert predictions.pred_prob.between(0, 1).all()
    assert set(predictions.model_name) == set(fitted)
    assert metrics.observation_top1_accuracy.between(0, 1).all()


def test_student_labels_join_after_scoring_only():
    frame = candidate_fixture(5)
    data = prepare_training_data(frame)
    model = build_baseline_pipeline("LogisticRegression").fit(data.X, data.y)
    inputs = frame.loc[frame.is_target.ne(-1), [
        "trip_id", "obs_idx", "timestamp", "gps_lon", "gps_lat", "candidate_arc_id",
        "from_node_id", "to_node_id", "dist_to_arc_m", "feat_distance", "feat_heading",
    ]].copy()
    inputs["is_target"] = -1
    labels = inputs[["trip_id", "obs_idx", "candidate_arc_id"]].copy()
    labels["reference_arc_id"] = labels.candidate_arc_id
    labels["label"] = (labels.candidate_arc_id == 0).astype(int)
    scored, evaluated = score_student_candidates(model, inputs, data.feature_columns, labels)
    assert "label" not in scored
    assert "reference_arc_id" not in scored
    assert evaluated.label.notna().all()
    probabilities = np.where(evaluated.candidate_arc_id.to_numpy() == 1, 0.9, 0.1)
    assert observation_top1_accuracy(evaluated, probabilities, target_column="label") == 0.0
