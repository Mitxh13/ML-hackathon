"""Feature engineering for map-matching candidate pairs (Section 1.3: Features & Candidate Tables).

Extracts geometric, kinematic, directional, topological, and temporal sequential
features for (GPS observation, candidate road arc) pairs.
All engineered feature columns MUST be prefixed with 'feat_'.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_a_data.preprocessing import haversine_distance_m, project_coordinates


def compute_node_degrees(arcs_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Compute out-degree and in-degree for road network nodes."""
    from_col = arcs_df.columns[0]
    to_col = arcs_df.columns[1]
    out_deg = arcs_df[from_col].value_counts().to_dict()
    in_deg = arcs_df[to_col].value_counts().to_dict()
    return out_deg, in_deg


def extract_candidate_features(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    track_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract comprehensive features for candidate road segment pairs.

    Features generated (all prefixed with 'feat_'):
    1. Distance & Geometry:
       - feat_dist_to_arc_m: Orthogonal distance from observation to candidate segment.
       - feat_dist_sq: Squared metric distance.
       - feat_dist_to_from_node_m: Geodesic distance to arc start node.
       - feat_dist_to_to_node_m: Geodesic distance to arc end node.
       - feat_min_endpoint_dist_m: Minimum distance to either endpoint.
       - feat_proj_fraction: Scalar projection ratio along segment [0.0, 1.0].
       - feat_arc_length_m: Road segment length in meters.
    2. Heading & Direction:
       - feat_heading_diff_rad: Absolute angular heading difference [0, pi].
       - feat_heading_cos_sim: Cosine similarity of movement vs segment direction [-1, 1].
       - feat_is_same_direction: 1 if cos_sim > 0.5, else 0.
    3. Kinematics & Temporal:
       - feat_speed_mps: Instantaneous vehicle speed.
       - feat_delta_t_s: Sampling interval in seconds.
       - feat_acceleration_mps2: Vehicle acceleration (dv/dt).
       - feat_is_stationary: 1 if speed < 0.5 m/s, else 0.
    4. Relative / Candidate Set Context:
       - feat_rank_by_dist: Rank of candidate sorted by distance (0 = closest).
       - feat_dist_ratio_to_min: Ratio of distance to minimum candidate distance.
       - feat_dist_diff_from_min: Distance difference from closest candidate.
       - feat_candidate_count: Number of candidates at this timestamp.
    5. Network Topology:
       - feat_from_node_out_degree: Out-degree of start node in network.
       - feat_to_node_in_degree: In-degree of end node in network.
    6. Sequential Context:
       - feat_connected_to_prev_nearest: 1 if arc connects from t-1 closest arc, else 0.
       - feat_same_as_prev_nearest: 1 if arc is identical to t-1 closest arc, else 0.

    Args:
        candidates_df: Candidate DataFrame conforming to Contract 1.
        nodes_df: Nodes DataFrame.
        arcs_df: Arcs DataFrame.
        track_df: Optional preprocessed track DataFrame with kinematics.

    Returns:
        Enriched candidate DataFrame with all 'feat_*' columns.
    """
    df = candidates_df.copy()
    if df.empty:
        return df

    # Project coordinates if needed for metric calculations
    proj_nodes = (
        nodes_df
        if "x_m" in nodes_df.columns
        else project_coordinates(nodes_df, lon_col="lon", lat_col="lat")
    )
    proj_track = (
        track_df
        if (track_df is not None and "x_m" in track_df.columns)
        else (project_coordinates(track_df, lon_col="gps_lon", lat_col="gps_lat") if track_df is not None else None)
    )

    # 1. Basic distance features
    df["feat_dist_to_arc_m"] = df["dist_to_arc_m"]
    df["feat_dist_sq"] = df["dist_to_arc_m"] ** 2

    # 2. Endpoint distances & Arc geometry
    from_nodes = df["from_node_id"].values.astype(int)
    to_nodes = df["to_node_id"].values.astype(int)

    u_lons = nodes_df.iloc[from_nodes]["lon"].values
    u_lats = nodes_df.iloc[from_nodes]["lat"].values
    v_lons = nodes_df.iloc[to_nodes]["lon"].values
    v_lats = nodes_df.iloc[to_nodes]["lat"].values

    p_lons = df["gps_lon"].values
    p_lats = df["gps_lat"].values

    # Endpoint distances via vectorized Haversine
    r = 6371000.0
    def _vec_haversine(lon1, lat1, lon2, lat2):
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
        return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))

    d_from = _vec_haversine(p_lons, p_lats, u_lons, u_lats)
    d_to = _vec_haversine(p_lons, p_lats, v_lons, v_lats)
    arc_lengths = _vec_haversine(u_lons, u_lats, v_lons, v_lats)

    df["feat_dist_to_from_node_m"] = d_from
    df["feat_dist_to_to_node_m"] = d_to
    df["feat_min_endpoint_dist_m"] = np.minimum(d_from, d_to)
    df["feat_arc_length_m"] = np.maximum(arc_lengths, 1.0)

    # Vectorized orthogonal projection fraction
    u_x = proj_nodes["x_m"].values[from_nodes]
    u_y = proj_nodes["y_m"].values[from_nodes]
    v_x = proj_nodes["x_m"].values[to_nodes]
    v_y = proj_nodes["y_m"].values[to_nodes]

    if proj_track is not None and "obs_idx" in proj_track.columns:
        obs_x_map = proj_track.set_index("obs_idx")["x_m"].to_dict()
        obs_y_map = proj_track.set_index("obs_idx")["y_m"].to_dict()
        pt_x = df["obs_idx"].map(obs_x_map).values
        pt_y = df["obs_idx"].map(obs_y_map).values
    else:
        pt_x = u_x
        pt_y = u_y

    dx = v_x - u_x
    dy = v_y - u_y
    l2 = np.maximum(dx * dx + dy * dy, 1e-6)
    alpha = ((pt_x - u_x) * dx + (pt_y - u_y) * dy) / l2
    df["feat_proj_fraction"] = np.clip(alpha, 0.0, 1.0)

    # 3. Heading & Direction alignment
    segment_headings = np.arctan2(dy, dx)
    if track_df is not None and "heading_rad" in track_df.columns:
        obs_heading = track_df.set_index("obs_idx")["heading_rad"].to_dict()
        point_headings = df["obs_idx"].map(obs_heading).fillna(0.0).values
        diff = np.abs(segment_headings - point_headings)
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        df["feat_heading_diff_rad"] = diff
        cos_sim = np.cos(diff)
        df["feat_heading_cos_sim"] = cos_sim
        df["feat_is_same_direction"] = (cos_sim > 0.5).astype(float)
    else:
        df["feat_heading_diff_rad"] = 0.0
        df["feat_heading_cos_sim"] = 1.0
        df["feat_is_same_direction"] = 1.0

    # 4. Kinematics (speed, delta_t, acceleration, stationary)
    if track_df is not None:
        if "speed_mps" in track_df.columns:
            obs_speed = track_df.set_index("obs_idx")["speed_mps"].to_dict()
            speeds = df["obs_idx"].map(obs_speed).fillna(0.0).values
            df["feat_speed_mps"] = speeds
            df["feat_is_stationary"] = (speeds < 0.5).astype(float)
        else:
            df["feat_speed_mps"] = 0.0
            df["feat_is_stationary"] = 0.0

        if "delta_t" in track_df.columns:
            obs_dt = track_df.set_index("obs_idx")["delta_t"].to_dict()
            df["feat_delta_t_s"] = df["obs_idx"].map(obs_dt).fillna(1.0).values
        else:
            df["feat_delta_t_s"] = 1.0

        # Acceleration
        if "speed_mps" in track_df.columns and "delta_t" in track_df.columns:
            acc_series = track_df["speed_mps"].diff().fillna(0.0) / np.maximum(track_df["delta_t"], 0.1)
            obs_acc = dict(zip(track_df["obs_idx"], acc_series))
            df["feat_acceleration_mps2"] = df["obs_idx"].map(obs_acc).fillna(0.0).values
        else:
            df["feat_acceleration_mps2"] = 0.0
    else:
        df["feat_speed_mps"] = 0.0
        df["feat_is_stationary"] = 0.0
        df["feat_delta_t_s"] = 1.0
        df["feat_acceleration_mps2"] = 0.0

    # 5. Relative / Candidate Set Context at observation t
    min_dist_by_obs = df.groupby("obs_idx")["dist_to_arc_m"].transform("min")
    count_by_obs = df.groupby("obs_idx")["dist_to_arc_m"].transform("count")

    df["feat_rank_by_dist"] = df.groupby("obs_idx")["dist_to_arc_m"].rank(method="dense") - 1.0
    df["feat_dist_ratio_to_min"] = df["dist_to_arc_m"] / np.maximum(min_dist_by_obs, 1.0)
    df["feat_dist_diff_from_min"] = df["dist_to_arc_m"] - min_dist_by_obs
    df["feat_candidate_count"] = count_by_obs.astype(float)

    # 6. Topological node degrees
    out_deg, in_deg = compute_node_degrees(arcs_df)
    df["feat_from_node_out_degree"] = df["from_node_id"].map(out_deg).fillna(1.0).astype(float)
    df["feat_to_node_in_degree"] = df["to_node_id"].map(in_deg).fillna(1.0).astype(float)

    # 7. Sequential Continuity Context (t-1 -> t)
    # Determine nearest candidate at each observation
    nearest_cands = df[df["feat_rank_by_dist"] == 0.0].drop_duplicates("obs_idx")
    prev_nearest_arc = dict(zip(nearest_cands["obs_idx"] + 1, nearest_cands["candidate_arc_id"]))
    prev_nearest_to_node = dict(zip(nearest_cands["obs_idx"] + 1, nearest_cands["to_node_id"]))

    prev_arc_series = df["obs_idx"].map(prev_nearest_arc)
    prev_to_node_series = df["obs_idx"].map(prev_nearest_to_node)

    df["feat_same_as_prev_nearest"] = (df["candidate_arc_id"] == prev_arc_series).astype(float).fillna(0.0)
    df["feat_connected_to_prev_nearest"] = (df["from_node_id"] == prev_to_node_series).astype(float).fillna(0.0)

    validate_candidate_df(df)
    return df
