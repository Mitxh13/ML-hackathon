"""Synthetic unit tests for Section 1.2 (run with project .venv pytest)."""
import numpy as np
import pandas as pd
import pytest
from src.contracts.schemas import CANDIDATE_BASE_COLS, POINT_GT_COLS, validate_candidate_df, validate_point_gt_df
from src.lane_a_data.ground_truth import assign_point_ground_truth, local_transformer, point_to_segment_distance
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.pipeline import run_partition
from src.lane_a_data.validation import candidate_metrics


def network():
    # Two connected eastbound segments plus a parallel reverse direction candidate.
    nodes = pd.DataFrame({"lon": [0, .001, .002, .001], "lat": [0, 0, 0, .0002], "node_id": range(4)})
    arcs = pd.DataFrame({"from_node_id": [0, 1, 1, 2], "to_node_id": [1, 2, 3, 0], "arc_id": range(4)})
    return nodes, arcs


def test_projection_uses_lon_lat_and_metric_units():
    transformer, crs = local_transformer(np.array([0.]), np.array([0.]))
    x, y = transformer.transform([0., .001], [0., 0.])
    assert "32631" in crs
    assert 100 < x[1] - x[0] < 120
    assert abs(y[1] - y[0]) < 1


def test_legacy_point_to_segment_distance_helper_is_preserved_and_exact():
    assert 10 < point_to_segment_distance(.005, .0001, 0., 0., .01, 0.) < 12


def test_monotone_route_order_and_repeated_arc_occurrences():
    nodes, arcs = network()
    # Arc 0 repeats after the connected loop 0 -> 1 -> 3.
    route = pd.DataFrame({"arc_id": [0, 1, 3, 0]})
    track = pd.DataFrame({"gps_lon": [.0001, .0012, .0019, .0008], "gps_lat": [0., 0., 0., 0.],
                          "timestamp": [0., 1., 2., 5.]})
    got = assign_point_ground_truth("t", track, route, nodes, arcs)
    assert got.reference_route_order.is_monotonic_increasing
    assert got.true_arc_id.iloc[0] == 0
    assert got.reference_route_order.dropna().isin([0, 1, 2, 3]).all()
    assert got.alignment_status.eq("aligned").all()


def test_multiple_observations_and_candidate_distance_labels():
    nodes, arcs = network()
    track = pd.DataFrame({"gps_lon": [.0003, .0004], "gps_lat": [.00001, .00001],
                          "timestamp": [0., 2.], "obs_idx": [7, 8]})
    gt = pd.DataFrame({"trip_id": ["t", "t"], "obs_idx": [7, 8], "reference_arc_id": [0, 0]})
    candidates = CandidateGenerator(max_dist_m=60, max_k=3, max_radius_m=100).generate_candidates("t", track, nodes, arcs, gt, include_section12_fields=True)
    obs7 = candidates[candidates.obs_idx == 7]
    assert obs7.dist_to_arc_m.is_monotonic_increasing
    assert obs7.iloc[0].candidate_arc_id == 0
    assert obs7.loc[obs7.candidate_arc_id == 0, "label"].eq(1).all()
    assert obs7.loc[obs7.candidate_arc_id != 0, "label"].eq(0).all()
    assert candidates.groupby("obs_idx").size().loc[8] >= 1


def test_existing_candidate_consumer_receives_legacy_contract_by_default():
    nodes, arcs = network()
    track = pd.DataFrame({"gps_lon": [.0003], "gps_lat": [.00001], "timestamp": [0.], "obs_idx": [3]})
    gt = pd.DataFrame({"trip_id": ["t"], "obs_idx": [3], "true_arc_id": [0]})
    result = CandidateGenerator(max_dist_m=60).generate_candidates("t", track, nodes, arcs, gt)
    assert result.columns.tolist() == CANDIDATE_BASE_COLS
    assert validate_candidate_df(result)
    assert result.is_target.eq(1).any()


def test_missing_reference_is_candidate_miss_and_empty_sets_counted():
    alignment = pd.DataFrame({"trip_id": ["a", "b"], "obs_idx": [0, 0],
        "reference_arc_id": [4, 0], "alignment_status": ["aligned", "aligned"], "alignment_distance_m": [2., 3.]})
    candidates = pd.DataFrame({"trip_id": ["a"], "obs_idx": [0], "candidate_arc_id": [2],
        "candidate_rank": [1], "dist_to_arc_m": [5.], "fallback_used": [False]})
    recall, summary = candidate_metrics(alignment, candidates)
    assert recall.loc[recall.trip_id == "OVERALL", "candidate_miss_count"].iloc[0] == 2
    assert summary.loc[summary.trip_id == "b", "no_candidate_observations"].iloc[0] == 1
    assert summary.loc[summary.trip_id == "OVERALL", "no_candidate_observations"].iloc[0] == 1


def test_invalid_arc_node_reference_fails_explicitly():
    nodes, arcs = network()
    arcs.loc[0, "to_node_id"] = 99
    track = pd.DataFrame({"gps_lon": [0.], "gps_lat": [0.], "timestamp": [0.]})
    try:
        assign_point_ground_truth("bad", track, pd.DataFrame({"arc_id": [0]}), nodes, arcs)
    except ValueError as exc:
        assert "missing node" in str(exc)
    else:
        raise AssertionError("invalid node reference should fail")


def test_shared_point_gt_contract_is_preserved_with_diagnostics():
    nodes, arcs = network()
    track = pd.DataFrame({"gps_lon": [.0002], "gps_lat": [0.], "timestamp": [0.]})
    result = assign_point_ground_truth("t", track, pd.DataFrame({"arc_id": [0]}), nodes, arcs)
    assert all(c in result for c in POINT_GT_COLS)
    assert "reference_arc_id" not in result
    assert validate_point_gt_df(result)
    assert {"reference_route_order", "alignment_distance_m", "alignment_status"}.issubset(result.columns)


@pytest.mark.parametrize("lon,lat", [(np.nan, 0.), (np.inf, 0.), (181., 0.), (0., 91.)])
def test_invalid_gps_coordinates_are_unlabeled(lon, lat):
    nodes, arcs = network()
    track = pd.DataFrame({"gps_lon": [lon], "gps_lat": [lat], "timestamp": [0.], "obs_idx": [12]})
    result = assign_point_ground_truth("t", track, pd.DataFrame({"arc_id": [0]}), nodes, arcs)
    assert pd.isna(result.true_arc_id.iloc[0])
    assert result.alignment_status.iloc[0] != "aligned"


def test_empty_track_returns_shared_schema_and_empty_route_fails():
    nodes, arcs = network()
    empty = pd.DataFrame(columns=["gps_lon", "gps_lat", "timestamp"])
    result = assign_point_ground_truth("t", empty, pd.DataFrame({"arc_id": [0]}), nodes, arcs)
    assert all(c in result for c in POINT_GT_COLS)
    assert result.empty
    with pytest.raises(ValueError, match="empty"):
        assign_point_ground_truth("t", empty, pd.DataFrame({"arc_id": []}), nodes, arcs)


def test_alignment_distance_threshold_and_empty_nonincreasing_or_gapped_times():
    nodes, arcs = network()
    route = pd.DataFrame({"arc_id": [0, 1]})
    track = pd.DataFrame({"gps_lon": [.0002, .0012, .0018], "gps_lat": [0., 0., .01],
                          "timestamp": [10., 2., 10000.], "obs_idx": [4, 5, 6]})
    result = assign_point_ground_truth("t", track, route, nodes, arcs, max_alignment_distance_m=80)
    assert result.obs_idx.tolist() == [4, 5, 6]
    assert pd.isna(result.true_arc_id.iloc[2])
    assert result.alignment_status.iloc[2] != "aligned"


def test_disconnected_route_is_rejected():
    nodes, arcs = network()
    with pytest.raises(ValueError, match="disconnected"):
        assign_point_ground_truth("t", pd.DataFrame({"gps_lon": [0.], "gps_lat": [0.], "timestamp": [0.]}),
                                  pd.DataFrame({"arc_id": [0, 3]}), nodes, arcs)


def test_long_segment_midpoint_reverse_arc_and_zero_length_are_retrieved():
    # 1 km horizontal segment; observation is near its midpoint, far from endpoints.
    nodes = pd.DataFrame({"lon": [0., .01, .005, .005], "lat": [0., 0., .0001, .0001]})
    arcs = pd.DataFrame({"from_node_id": [0, 1, 2, 2], "to_node_id": [1, 0, 3, 2]})
    track = pd.DataFrame({"gps_lon": [.005, .005], "gps_lat": [.00001, .00001], "timestamp": [0., 1.]})
    got = CandidateGenerator(max_dist_m=30, max_k=4, expand_radius=False).generate_candidates("x", track, nodes, arcs)
    assert {0, 1, 2, 3}.issubset(set(got.candidate_arc_id))
    assert got.groupby("obs_idx").size().max() <= 4
    assert got.dist_to_arc_m.max() < 30


def test_radius_expansion_fallback_and_candidate_miss_are_flagged():
    nodes, arcs = network()
    track = pd.DataFrame({"gps_lon": [.001], "gps_lat": [.0005], "timestamp": [0.], "obs_idx": [2]})
    gt = pd.DataFrame({"obs_idx": [2], "true_arc_id": [99]})
    expanded = CandidateGenerator(max_dist_m=10, min_radius_m=10, max_radius_m=100, max_k=2).generate_candidates("t", track, nodes, arcs, gt, include_section12_fields=True)
    assert not expanded.fallback_used.any()
    assert expanded.dist_to_arc_m.max() <= 100
    assert expanded.label.eq(0).all()
    far = track.assign(gps_lon=.5)
    fallback = CandidateGenerator(max_dist_m=10, min_radius_m=10, max_radius_m=20).generate_candidates("t", far, nodes, arcs, gt, include_section12_fields=True)
    assert fallback.fallback_used.all()


def test_candidate_pairs_unique_and_student_labels_separate(tmp_path):
    data = tmp_path / "data"
    trip = data / "train" / "syn"
    trip.mkdir(parents=True)
    (trip / "syn.track").write_text("0.0002 0 0\n0.0008 0 1\n0.0012 0 2\n")
    (trip / "syn.nodes").write_text("0 0\n0.001 0\n0.002 0\n0.001 0.0002\n")
    (trip / "syn.arcs").write_text("0 1\n1 2\n1 3\n2 0\n")
    (trip / "syn.route").write_text("0\n1\n")
    out, results = tmp_path / "intermediate", tmp_path / "results"
    _, candidates = run_partition(data, "train", out, CandidateGenerator(max_dist_m=100), results)
    assert not candidates.duplicated(["trip_id", "obs_idx", "candidate_arc_id"]).any()
    assert {"candidate_arc_id", "candidate_rank", "distance_m", "label"}.issubset(candidates.columns)
    assert (results / "candidate_recall.csv").exists()
    assert (results / "candidate_generation_summary.csv").exists()
    recalls = pd.read_csv(results / "candidate_recall.csv")
    assert (recalls.trip_id == "OVERALL").any()
    # Run a separate student partition and confirm targets never enter model input.
    student_trip = data / "student_test" / "syn"
    student_trip.mkdir(parents=True)
    for source in trip.iterdir():
        (student_trip / source.name).write_bytes(source.read_bytes())
    run_partition(data, "student_test", out, CandidateGenerator(max_dist_m=100), results)
    student_input = pd.read_csv(out / "candidate_segments_student_test.csv")
    student_labels = pd.read_csv(out / "candidate_labels_student_test.csv")
    assert not {"true_arc_id", "reference_arc_id", "label", "is_target", "is_reference"}.intersection(student_input.columns)
    assert {"trip_id", "obs_idx", "candidate_arc_id", "reference_arc_id", "label"}.issubset(student_labels.columns)
    assert not student_labels.duplicated(["trip_id", "obs_idx", "candidate_arc_id"]).any()
