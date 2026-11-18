"""Lane C: Road graph, HMM/Viterbi decoding, shared evaluation metrics, Folium maps."""

from src.lane_c_hmm.metrics import (
    compute_top1_accuracy,
    compute_macro_f1,
    compute_route_continuity,
    compute_candidate_miss_rate,
    compute_log_loss_safe,
    evaluate_predictions,
)
from src.lane_c_hmm.road_graph import RoadNetworkGraph, haversine_distance_m
from src.lane_c_hmm.decoder import ViterbiRouteDecoder, collapse_consecutive_duplicates
from src.lane_c_hmm.compare_hmm import compare_decoders
from src.lane_c_hmm.folium_maps import create_route_map, render_map_html

__all__ = [
    "compute_top1_accuracy",
    "compute_macro_f1",
    "compute_route_continuity",
    "compute_candidate_miss_rate",
    "compute_log_loss_safe",
    "evaluate_predictions",
    "RoadNetworkGraph",
    "haversine_distance_m",
    "ViterbiRouteDecoder",
    "collapse_consecutive_duplicates",
    "compare_decoders",
    "create_route_map",
    "render_map_html",
]
