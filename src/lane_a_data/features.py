"""Feature engineering for map-matching candidate pairs.

All engineered feature columns MUST be prefixed with 'feat_'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_c_hmm.road_graph import haversine_distance_m


def extract_candidate_features(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    track_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract geometric, temporal, and kinematic features for candidate arc pairs.

    Features generated (all prefixed with 'feat_'):
    - feat_dist_to_arc_m: Metric distance from observation to candidate segment.
    - feat_dist_sq: Squared metric distance.
    - feat_arc_length_m: Geodesic length of the candidate road segment.
    - feat_heading_diff: Absolute heading difference between vehicle motion and segment direction.
    - feat_speed_mps: Instantaneous vehicle speed.
    - feat_proj_fraction: Scalar projection ratio of point along segment [0.0, 1.0].
    - feat_rank_by_dist: Rank of candidate sorted by distance (0 = closest).

    Args:
        candidates_df: Candidate DataFrame conforming to Contract 1.
        nodes_df: Nodes DataFrame.
        arcs_df: Arcs DataFrame.
        track_df: Optional preprocessed track DataFrame with kinematics.

    Returns:
        Enriched candidate DataFrame with 'feat_*' columns.
    """
    df = candidates_df.copy()
    if df.empty:
        return df

    # Feature 1: Distance and distance squared
    df["feat_dist_to_arc_m"] = df["dist_to_arc_m"]
    df["feat_dist_sq"] = df["dist_to_arc_m"] ** 2

    # Feature 2: Rank by distance within each observation
    df["feat_rank_by_dist"] = df.groupby("obs_idx")["dist_to_arc_m"].rank(method="dense") - 1.0

    # Feature 3: Arc geometric properties
    from_nodes = df["from_node_id"].values
    to_nodes = df["to_node_id"].values

    u_lons = nodes_df.iloc[from_nodes]["lon"].values
    u_lats = nodes_df.iloc[from_nodes]["lat"].values
    v_lons = nodes_df.iloc[to_nodes]["lon"].values
    v_lats = nodes_df.iloc[to_nodes]["lat"].values

    # Segment length
    arc_lengths = [
        haversine_distance_m(u_lons[i], u_lats[i], v_lons[i], v_lats[i])
        for i in range(len(df))
    ]
    df["feat_arc_length_m"] = arc_lengths

    # Segment heading (-pi to pi)
    d_lon = v_lons - u_lons
    d_lat = v_lats - u_lats
    segment_headings = np.arctan2(d_lat, d_lon)

    # GPS trajectory heading difference
    if track_df is not None and "heading_rad" in track_df.columns:
        obs_heading = track_df.set_index("obs_idx")["heading_rad"].to_dict()
        point_headings = df["obs_idx"].map(obs_heading).fillna(0.0).values
        diff = np.abs(segment_headings - point_headings)
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        df["feat_heading_diff"] = diff
    else:
        df["feat_heading_diff"] = 0.0

    # Speed
    if track_df is not None and "speed_mps" in track_df.columns:
        obs_speed = track_df.set_index("obs_idx")["speed_mps"].to_dict()
        df["feat_speed_mps"] = df["obs_idx"].map(obs_speed).fillna(0.0).values
    else:
        df["feat_speed_mps"] = 0.0

    # Projection fraction [0, 1] along segment
    p_lons = df["gps_lon"].values
    p_lats = df["gps_lat"].values
    proj_fractions = []
    for i in range(len(df)):
        seg_len = max(1.0, arc_lengths[i])
        d_from = haversine_distance_m(p_lons[i], p_lats[i], u_lons[i], u_lats[i])
        proj_fractions.append(float(np.clip(d_from / seg_len, 0.0, 1.0)))
    df["feat_proj_fraction"] = proj_fractions

    validate_candidate_df(df)
    return df
