"""Point-to-segment ground-truth alignment from .route to .track observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.schemas import POINT_GT_COLS, validate_point_gt_df
from src.lane_c_hmm.road_graph import haversine_distance_m


def point_to_segment_distance(
    p_lon: float,
    p_lat: float,
    u_lon: float,
    u_lat: float,
    v_lon: float,
    v_lat: float,
) -> float:
    """Approximate perpendicular/geodesic distance from a point to a road segment (u, v)."""
    # Midpoint distance approximation or direct projection
    mid_lon = (u_lon + v_lon) / 2.0
    mid_lat = (u_lat + v_lat) / 2.0
    d_u = haversine_distance_m(p_lon, p_lat, u_lon, u_lat)
    d_v = haversine_distance_m(p_lon, p_lat, v_lon, v_lat)
    d_mid = haversine_distance_m(p_lon, p_lat, mid_lon, mid_lat)
    return float(min(d_u, d_v, d_mid))


def assign_point_ground_truth(
    trip_id: str,
    track_df: pd.DataFrame,
    route_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align GPS observations to the ground-truth .route sequence monotonically.

    For each GPS observation t in .track, assigns the optimal true_arc_id from the
    chronological sequence of traversed segments in .route, minimizing metric distance
    while enforcing forward/monotonic progress along .route.

    Args:
        trip_id: Identifier of the trip.
        track_df: Track observations with ['gps_lon', 'gps_lat', 'timestamp'].
        route_df: Traversed arc IDs sequence.
        nodes_df: Road nodes coordinates.
        arcs_df: Road arcs connectivity.

    Returns:
        DataFrame conforming to Contract 3 (POINT_GT_COLS).
    """
    route_arcs = route_df["arc_id"].values.tolist()
    if not route_arcs:
        raise ValueError(f"Route for trip {trip_id} is empty.")

    t_len = len(track_df)
    r_len = len(route_arcs)

    # Pre-extract coordinates of route segments
    route_coords = []
    for arc_id in route_arcs:
        u = int(arcs_df.iloc[arc_id, 0])
        v = int(arcs_df.iloc[arc_id, 1])
        u_lon, u_lat = float(nodes_df.iloc[u, 0]), float(nodes_df.iloc[u, 1])
        v_lon, v_lat = float(nodes_df.iloc[v, 0]), float(nodes_df.iloc[v, 1])
        route_coords.append((u_lon, u_lat, v_lon, v_lat))

    # Dynamic programming / monotonic alignment
    cost_matrix = np.full((t_len, r_len), np.inf)

    # Compute metric distance from each GPS point to each route segment
    for t in range(t_len):
        p_lon = float(track_df.iloc[t]["gps_lon"])
        p_lat = float(track_df.iloc[t]["gps_lat"])
        for r in range(r_len):
            u_lon, u_lat, v_lon, v_lat = route_coords[r]
            cost_matrix[t, r] = point_to_segment_distance(p_lon, p_lat, u_lon, u_lat, v_lon, v_lat)

    # Monotonic alignment with DP
    dp = np.full((t_len, r_len), np.inf)
    parent = np.zeros((t_len, r_len), dtype=int)

    # Initialize t = 0
    dp[0, :] = cost_matrix[0, :]

    for t in range(1, t_len):
        min_prev = np.inf
        best_r = 0
        for r in range(r_len):
            # Monotonic: can come from any prev_r <= r
            if dp[t - 1, r] < min_prev:
                min_prev = dp[t - 1, r]
                best_r = r
            dp[t, r] = min_prev + cost_matrix[t, r]
            parent[t, r] = best_r

    # Backtrack
    best_last_r = int(np.argmin(dp[-1, :]))
    assigned_r = [best_last_r]

    for t in range(t_len - 1, 0, -1):
        prev_r = parent[t, assigned_r[-1]]
        assigned_r.append(prev_r)

    assigned_r.reverse()
    true_arcs = [route_arcs[r] for r in assigned_r]

    gt_df = pd.DataFrame(
        {
            "trip_id": [str(trip_id)] * t_len,
            "obs_idx": track_df["obs_idx"].values if "obs_idx" in track_df.columns else np.arange(t_len),
            "timestamp": track_df["timestamp"].values,
            "gps_lon": track_df["gps_lon"].values,
            "gps_lat": track_df["gps_lat"].values,
            "true_arc_id": true_arcs,
        }
    )

    validate_point_gt_df(gt_df)
    return gt_df[POINT_GT_COLS]
