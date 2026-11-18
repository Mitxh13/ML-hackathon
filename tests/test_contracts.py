"""Unit tests verifying schemas in src/contracts/schemas.py and metrics in src/lane_c_hmm/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.contracts.schemas import (
    CANDIDATE_BASE_COLS,
    MODEL_OUTPUT_COLS,
    POINT_GT_COLS,
    DecoderOutput,
    EvaluationResponse,
    validate_candidate_df,
    validate_model_output_df,
    validate_point_gt_df,
)
from src.lane_c_hmm.metrics import (
    compute_candidate_miss_rate,
    compute_log_loss_safe,
    compute_macro_f1,
    compute_route_continuity,
    compute_top1_accuracy,
    evaluate_predictions,
)
from src.lane_c_hmm.decoder import ViterbiRouteDecoder


@pytest.fixture
def synthetic_5point_trip():
    """Synthetic 5-observation trip fixture."""
    # 5 GPS points along a road corridor
    track_df = pd.DataFrame(
        {
            "obs_idx": [0, 1, 2, 3, 4],
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "gps_lon": [77.500, 77.501, 77.502, 77.503, 77.504],
            "gps_lat": [12.900, 12.901, 12.902, 12.903, 12.904],
        }
    )

    # 4 connected road arcs (0 -> 1 -> 2 -> 3)
    # nodes 0(77.500, 12.900), 1(77.501, 12.901), 2(77.502, 12.902), 3(77.503, 12.903), 4(77.504, 12.904)
    nodes_df = pd.DataFrame(
        {
            "lon": [77.500, 77.501, 77.502, 77.503, 77.504],
            "lat": [12.900, 12.901, 12.902, 12.903, 12.904],
        }
    )

    arcs_df = pd.DataFrame(
        {
            "from_node_id": [0, 1, 2, 3],
            "to_node_id": [1, 2, 3, 4],
        }
    )

    point_gt_df = pd.DataFrame(
        {
            "trip_id": ["test_trip_01"] * 5,
            "obs_idx": [0, 1, 2, 3, 4],
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "gps_lon": [77.500, 77.501, 77.502, 77.503, 77.504],
            "gps_lat": [12.900, 12.901, 12.902, 12.903, 12.904],
            "true_arc_id": [0, 1, 2, 3, 3],
        }
    )

    # Candidate table (2 candidates per observation: 1 true, 1 distractor)
    records = []
    for t in range(5):
        true_arc = point_gt_df.iloc[t]["true_arc_id"]
        distractor_arc = (true_arc + 1) % 4

        # True candidate
        records.append(
            {
                "trip_id": "test_trip_01",
                "obs_idx": t,
                "timestamp": float(t),
                "gps_lon": 77.500 + t * 0.001,
                "gps_lat": 12.900 + t * 0.001,
                "candidate_arc_id": true_arc,
                "from_node_id": int(arcs_df.iloc[true_arc]["from_node_id"]),
                "to_node_id": int(arcs_df.iloc[true_arc]["to_node_id"]),
                "dist_to_arc_m": 2.5,
                "is_target": 1,
                "feat_dist_m": 2.5,
                "feat_heading_diff": 0.05,
            }
        )

        # Distractor candidate
        records.append(
            {
                "trip_id": "test_trip_01",
                "obs_idx": t,
                "timestamp": float(t),
                "gps_lon": 77.500 + t * 0.001,
                "gps_lat": 12.900 + t * 0.001,
                "candidate_arc_id": distractor_arc,
                "from_node_id": int(arcs_df.iloc[distractor_arc]["from_node_id"]),
                "to_node_id": int(arcs_df.iloc[distractor_arc]["to_node_id"]),
                "dist_to_arc_m": 28.0,
                "is_target": 0,
                "feat_dist_m": 28.0,
                "feat_heading_diff": 0.85,
            }
        )

    candidates_df = pd.DataFrame(records)

    return {
        "track_df": track_df,
        "nodes_df": nodes_df,
        "arcs_df": arcs_df,
        "point_gt_df": point_gt_df,
        "candidates_df": candidates_df,
    }


def test_candidate_schema_contract(synthetic_5point_trip):
    """Test Contract 1 (Candidate DataFrame validation)."""
    cand_df = synthetic_5point_trip["candidates_df"]
    assert validate_candidate_df(cand_df) is True

    # Missing column should raise ValueError
    broken_df = cand_df.drop(columns=["dist_to_arc_m"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candidate_df(broken_df)

    # Feature column without 'feat_' prefix should raise ValueError
    invalid_feat_df = cand_df.copy()
    invalid_feat_df["raw_speed"] = 12.0
    with pytest.raises(ValueError, match="prefixed with 'feat_'"):
        validate_candidate_df(invalid_feat_df)


def test_model_output_schema_contract(synthetic_5point_trip):
    """Test Contract 2 (Model Output DataFrame validation)."""
    cand_df = synthetic_5point_trip["candidates_df"].copy()
    cand_df["pred_prob"] = [0.95, 0.05] * 5
    cand_df["pred_label"] = [1, 0] * 5

    assert validate_model_output_df(cand_df) is True

    # Out of bounds probability should raise ValueError
    bad_prob_df = cand_df.copy()
    bad_prob_df.loc[0, "pred_prob"] = 1.5
    with pytest.raises(ValueError, match="bounded within"):
        validate_model_output_df(bad_prob_df)


def test_point_gt_schema_contract(synthetic_5point_trip):
    """Test Contract 3 (Point Ground-Truth DataFrame validation)."""
    gt_df = synthetic_5point_trip["point_gt_df"]
    assert validate_point_gt_df(gt_df) is True

    broken_gt = gt_df.drop(columns=["true_arc_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_point_gt_df(broken_gt)


def test_decoder_output_and_eval_response_models():
    """Test Contract 4 & 5 (Pydantic models)."""
    dec_out = DecoderOutput(
        trip_id="test_01",
        mode="ml_hmm",
        point_predictions=[0, 1, 2, 3, 3],
        reconstructed_route=[0, 1, 2, 3],
        inference_time_sec=0.012,
    )
    assert dec_out.trip_id == "test_01"
    assert dec_out.reconstructed_route == [0, 1, 2, 3]

    eval_resp = EvaluationResponse(
        trip_id="test_01",
        num_gps_points=5,
        point_predictions=[0, 1, 2, 3, 3],
        reconstructed_route=[0, 1, 2, 3],
        metrics={"top1_accuracy": 1.0, "macro_f1": 1.0},
        folium_map_html="<html><body>Map</body></html>",
    )
    assert eval_resp.num_gps_points == 5
    assert eval_resp.metrics["top1_accuracy"] == 1.0


def test_metrics_on_synthetic_trip(synthetic_5point_trip):
    """Test all shared metrics on synthetic 5-point trip."""
    gt_df = synthetic_5point_trip["point_gt_df"]
    arcs_df = synthetic_5point_trip["arcs_df"]
    cand_df = synthetic_5point_trip["candidates_df"]

    y_true = gt_df["true_arc_id"].values
    # Exact match predictions
    y_pred_perfect = [0, 1, 2, 3, 3]

    # 1. Top-1 Accuracy
    assert compute_top1_accuracy(y_true, y_pred_perfect) == 1.0
    assert compute_top1_accuracy(y_true, [0, 1, 2, 3, 0]) == 0.8

    # 2. Macro F1
    assert compute_macro_f1(y_true, y_pred_perfect) == 1.0

    # 3. Route Continuity (0 -> 1 -> 2 -> 3 -> 3 are all continuous)
    continuity = compute_route_continuity(y_pred_perfect, arcs_df)
    assert continuity == 100.0

    # Broken continuity test (e.g., jump from 0 directly to 3)
    broken_preds = [0, 3, 1, 2, 3]
    broken_continuity = compute_route_continuity(broken_preds, arcs_df)
    assert broken_continuity < 100.0

    # 4. Candidate Miss Rate
    assert compute_candidate_miss_rate(cand_df) == 0.0

    # 5. Log Loss Safe
    loss = compute_log_loss_safe([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert loss is not None and loss < 0.5

    # 6. Evaluate Predictions dictionary
    metrics = evaluate_predictions(
        point_gt_df=gt_df,
        pred_arcs=y_pred_perfect,
        arcs_df=arcs_df,
        candidates_df=cand_df,
        inference_time_sec=0.005,
    )
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["route_continuity"] == 100.0
    assert metrics["candidate_miss_rate"] == 0.0
    assert metrics["inference_time_sec"] == 0.005


def test_viterbi_decoder_smoke(synthetic_5point_trip):
    """Test ViterbiRouteDecoder on synthetic candidate data."""
    cand_df = synthetic_5point_trip["candidates_df"].copy()
    nodes_df = synthetic_5point_trip["nodes_df"]
    arcs_df = synthetic_5point_trip["arcs_df"]

    cand_df["pred_prob"] = np.where(cand_df["is_target"] == 1, 0.95, 0.05)
    cand_df["pred_label"] = cand_df["is_target"]

    decoder = ViterbiRouteDecoder()

    # Test baseline mode
    out_dist = decoder.decode_trip(cand_df, nodes_df, arcs_df, mode="distance_baseline")
    assert out_dist.point_predictions == [0, 1, 2, 3, 3]
    assert out_dist.reconstructed_route == [0, 1, 2, 3]

    # Test ML mode
    out_ml = decoder.decode_trip(cand_df, nodes_df, arcs_df, mode="ml_hmm")
    assert out_ml.point_predictions == [0, 1, 2, 3, 3]
    assert out_ml.reconstructed_route == [0, 1, 2, 3]
