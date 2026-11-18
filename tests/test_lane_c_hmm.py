"""Tests for Section 3: road graph, shared metrics, Viterbi decoder, comparison and maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lane_c_hmm.compare_hmm import compare_decoders
from src.lane_c_hmm.decoder import HMMParams, ViterbiRouteDecoder, collapse_consecutive_duplicates
from src.lane_c_hmm.folium_maps import create_route_map, render_map_html
from src.lane_c_hmm.metrics import (
    compute_candidate_miss_rate,
    compute_macro_f1,
    compute_route_continuity,
    compute_route_overlap,
    compute_top1_accuracy,
    evaluate_predictions,
)
from src.lane_c_hmm.road_graph import RoadNetworkGraph

LON0, LAT0 = 8.6, 50.1


def lonlat(x_m, y_m):
    """Metres east / north of (LON0, LAT0) to degrees (local approximation, fine for tests)."""
    lat = LAT0 + np.asarray(y_m, float) / 111_320.0
    lon = LON0 + np.asarray(x_m, float) / (111_320.0 * np.cos(np.radians(LAT0)))
    return lon, lat


def ladder():
    """Two parallel one-way streets 30 m apart (north street eastbound, south street
    eastbound), joined only at x = 0 and x = 300 by short connectors.

        nodes: 0 (0,0) 1 (100,0) 2 (200,0) 3 (300,0)      south street, arcs 0,1,2 eastbound
               4 (0,30) 5 (100,30) 6 (200,30) 7 (300,30)  north street, arcs 3,4,5 eastbound
        arc 6: 0 -> 4 and arc 7: 7 -> 3 connect the streets at the ends.
    """
    xs = [0, 100, 200, 300] * 2
    ys = [0] * 4 + [30] * 4
    lon, lat = lonlat(xs, ys)
    nodes = pd.DataFrame({"lon": lon, "lat": lat})
    arcs = pd.DataFrame({"from_node_id": [0, 1, 2, 4, 5, 6, 0, 7], "to_node_id": [1, 2, 3, 5, 6, 7, 4, 3]})
    return nodes, arcs


def candidates_for(points_xy, timestamps, arcs_xy_dist, probs=None, truth=None, trip_id="t1"):
    rows = []
    for i, ((x, y), ts) in enumerate(zip(points_xy, timestamps)):
        lon, lat = lonlat(x, y)
        for k, (arc, d) in enumerate(arcs_xy_dist[i]):
            rows.append({"trip_id": trip_id, "obs_idx": i, "timestamp": float(ts), "gps_lon": float(lon),
                         "gps_lat": float(lat), "candidate_arc_id": arc, "from_node_id": 0, "to_node_id": 0,
                         "dist_to_arc_m": float(d),
                         "is_target": -1 if truth is None else int(arc == truth[i]),
                         **({"pred_prob": probs[i][k]} if probs is not None else {})})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- road graph
def test_graph_lengths_are_metric_and_paths_are_directed():
    nodes, arcs = ladder()
    g = RoadNetworkGraph(nodes, arcs)
    assert g.crs == "EPSG:32632"
    assert np.allclose(g.arc_length_m[:6], 100.0, rtol=0.01)
    assert g.shortest_path_distance(0, 3) == pytest.approx(300.0, rel=0.01)
    assert np.isinf(g.shortest_path_distance(3, 0))            # one-way streets: no way back
    assert g.shortest_path_arcs(0, 3) == [0, 1, 2]
    assert g.shortest_path_arcs(1, 4) is None
    assert np.isinf(g.shortest_path_distance(0, 3, limit_m=150.0))  # bounded search


def test_connect_route_fills_skipped_arcs_and_projection_is_metric():
    nodes, arcs = ladder()
    g = RoadNetworkGraph(nodes, arcs)
    assert g.connect_route([0, 2]) == [0, 1, 2]
    pts = g.project(*lonlat([50.0], [5.0]))
    d, along = g.project_onto_arcs(pts, np.array([0]))
    assert d[0] == pytest.approx(5.0, abs=0.1) and along[0] == pytest.approx(50.0, abs=0.5)


# ---------------------------------------------------------------------------- metrics
def test_metrics_skip_unlabelled_points():
    truth = pd.array([1, 2, pd.NA, 3], dtype="Int64")
    pred = np.array([1, 2, 99, 4])
    assert compute_top1_accuracy(truth, pred) == pytest.approx(2 / 3)
    assert 0.0 < compute_macro_f1(truth, pred) < 1.0


def test_candidate_miss_rate_ignores_unlabelled_observations():
    cands = pd.DataFrame({"trip_id": ["a"] * 5, "obs_idx": [0, 0, 1, 1, 2],
                          "is_target": [1, 0, 0, 0, -1]})
    assert compute_candidate_miss_rate(cands) == pytest.approx(0.5)


def test_continuity_uses_graph_and_elapsed_time():
    nodes, arcs = ladder()
    g = RoadNetworkGraph(nodes, arcs)
    assert compute_route_continuity([0, 0, 1, 2], arcs) == 100.0
    # 0 -> 2 skips arc 1 (100 m): continuous within 2 s only through the graph check
    assert compute_route_continuity([0, 2], arcs) == 0.0
    assert compute_route_continuity([0, 2], arcs, graph=g, timestamps=[0, 2]) == 100.0
    # south street -> north street mid-way is not reachable at all
    assert compute_route_continuity([1, 4], arcs, graph=g, timestamps=[0, 1]) == 0.0


def test_route_overlap_and_evaluate_keys():
    assert compute_route_overlap([1, 2, 3], [2, 3, 4]) == pytest.approx(
        {"route_precision": 2 / 3, "route_recall": 2 / 3, "route_f1": 2 / 3})
    nodes, arcs = ladder()
    gt = pd.DataFrame({"trip_id": "t", "obs_idx": [0, 1, 2], "timestamp": [0., 1., 2.],
                       "gps_lon": 0.0, "gps_lat": 0.0, "true_arc_id": pd.array([0, 1, pd.NA], dtype="Int64")})
    m = evaluate_predictions(gt, [0, 1, 2], arcs, graph=RoadNetworkGraph(nodes, arcs))
    assert m["top1_accuracy"] == 1.0 and m["label_coverage"] == pytest.approx(2 / 3)
    assert m["route_continuity"] == 100.0
    assert {"macro_f1", "log_loss", "candidate_miss_rate", "inference_time_sec", "route_f1"} <= set(m)


# ---------------------------------------------------------------------------- decoder
def ladder_trip():
    """A car drives along the south street (arcs 0 -> 1 -> 2). Fix 3 drifts 20 m north,
    so the north street (arc 4) is closer and the classifier prefers it, but arc 4 cannot
    be reached from arc 1 and back again."""
    pts = [(20, 2), (60, 3), (120, 4), (150, 20), (180, 3), (250, 2)]
    cands = [
        [(0, 2), (3, 28)], [(0, 3), (3, 27)], [(1, 4), (4, 26)],
        [(4, 10), (1, 20)], [(1, 3), (4, 27)], [(2, 2), (5, 28)],
    ]
    probs = [[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
    truth = [0, 0, 1, 1, 1, 2]
    return candidates_for(pts, range(6), cands, probs, truth), truth


def test_hmm_overrides_an_unreachable_classifier_choice():
    nodes, arcs = ladder()
    cands, truth = ladder_trip()
    dec = ViterbiRouteDecoder(HMMParams(beta_m=5.0, disconnected_penalty=10.0))
    clf = dec.decode_trip(cands, nodes, arcs, mode="classifier_only")
    hmm = dec.decode_trip(cands, nodes, arcs, mode="ml_hmm")
    assert clf.point_predictions[3] == 4                 # classifier jumps to the other street
    assert hmm.point_predictions == truth                # HMM keeps the connected route
    assert hmm.reconstructed_route == [0, 1, 2]
    assert hmm.disconnected_steps == 0
    dist = dec.decode_trip(cands, nodes, arcs, mode="distance_baseline")  # legacy alias
    assert dist.mode == "distance_hmm" and dist.point_predictions == truth


def test_large_gap_restarts_chain_and_missing_observation_is_filled():
    nodes, arcs = ladder()
    cands, _ = ladder_trip()
    cands = cands[cands.obs_idx != 2].copy()              # observation 2 has no candidates
    cands.loc[cands.obs_idx >= 4, "timestamp"] += 500.0   # 500 s gap before observation 4
    dec = ViterbiRouteDecoder(HMMParams(break_gap_s=120.0))
    trellis = dec.build_trellis(cands, RoadNetworkGraph(nodes, arcs), obs_index=range(6))
    out = dec.predict_arcs(trellis, "ml_hmm")
    assert out["chain_restarts"] == 1
    assert out["filled"] == 1 and len(out["point_predictions"]) == 6
    assert out["point_predictions"][2] == out["point_predictions"][1]


def test_disconnected_data_still_decodes():
    # Only unreachable candidates in the middle: the decoder must still return a path.
    nodes, arcs = ladder()
    pts = [(20, 2), (150, 30), (250, 2)]
    cands = candidates_for(pts, range(3), [[(0, 2)], [(4, 1)], [(2, 2)]], [[0.9], [0.9], [0.9]])
    out = ViterbiRouteDecoder().decode_trip(cands, nodes, arcs, mode="ml_hmm")
    assert out.point_predictions == [0, 4, 2]
    assert out.disconnected_steps == 2


def test_ml_modes_require_probabilities():
    nodes, arcs = ladder()
    cands, _ = ladder_trip()
    with pytest.raises(ValueError, match="pred_prob"):
        ViterbiRouteDecoder().decode_trip(cands.drop(columns="pred_prob"), nodes, arcs, mode="ml_hmm")


def test_collapse_and_compare_decoders():
    assert collapse_consecutive_duplicates([10, 10, 15, 15, 22, 10]) == [10, 15, 22, 10]
    nodes, arcs = ladder()
    cands, truth = ladder_trip()
    gt = cands.drop_duplicates("obs_idx")[["trip_id", "obs_idx", "timestamp", "gps_lon", "gps_lat"]].copy()
    gt["true_arc_id"] = truth
    table = compare_decoders(cands, nodes, arcs, gt, decoder=ViterbiRouteDecoder(HMMParams(beta_m=5.0)))
    top1 = table.set_index("mode")["top1_accuracy"]
    assert set(top1.index) == {"nearest", "classifier_only", "distance_hmm", "ml_hmm"}
    assert top1["ml_hmm"] == 1.0 and top1["classifier_only"] < 1.0


# ---------------------------------------------------------------------------- maps
def test_route_map_has_layers_and_keeps_reference_optional():
    nodes, arcs = ladder()
    lon, lat = lonlat([20, 60, 120], [2, 3, 4])
    track = pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "timestamp": [0., 1., 2.], "obs_idx": [0, 1, 2]})
    html = render_map_html(create_route_map(track, nodes, arcs, predicted_arcs=[0, 1], reference_route=[0, 1],
                                            classifier_arcs=[0, 4], wrong_mask=[False, True, False]))
    for name in ("Road network (nearby)", "Reference route (.route)", "Classifier-only route",
                 "Predicted route (HMM)", "GPS observations", "Wrong points"):
        assert name in html
    ui_html = render_map_html(create_route_map(track, nodes, arcs, predicted_arcs=[0, 1]))
    assert "Reference route" not in ui_html
