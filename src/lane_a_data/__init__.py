"""Lane A: Data processing, EDA, candidate generation, and feature extraction."""

from src.lane_a_data.loaders import (
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    load_trip,
)
from src.lane_a_data.preprocessing import project_coordinates, preprocess_track
from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.features import extract_candidate_features
from src.lane_a_data.eda_plots import (
    plot_missing_values,
    plot_speed_distribution,
    plot_candidate_distances,
)

__all__ = [
    "load_track",
    "load_nodes",
    "load_arcs",
    "load_route",
    "load_trip",
    "project_coordinates",
    "preprocess_track",
    "assign_point_ground_truth",
    "CandidateGenerator",
    "extract_candidate_features",
    "plot_missing_values",
    "plot_speed_distribution",
    "plot_candidate_distances",
]
