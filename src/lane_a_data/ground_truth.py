"""Point-to-segment ground-truth alignment from .route to .track observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts.schemas import POINT_GT_COLS, validate_point_gt_df
from src.lane_a_data.preprocessing import haversine_distance_m, project_coordinates


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

    # Project coordinates to metric CRS for fast vectorized orthogonal projection
    proj_track = (
        track_df
        if "x_m" in track_df.columns
        else project_coordinates(track_df, lon_col="gps_lon", lat_col="gps_lat")
    )
    proj_nodes = (
        nodes_df
        if "x_m" in nodes_df.columns
        else project_coordinates(nodes_df, lon_col="lon", lat_col="lat")
    )

    u_indices = arcs_df.iloc[route_arcs, 0].values.astype(int)
    v_indices = arcs_df.iloc[route_arcs, 1].values.astype(int)

    ax = proj_nodes["x_m"].values[u_indices][None, :]  # Shape: (1, M)
    ay = proj_nodes["y_m"].values[u_indices][None, :]
    bx = proj_nodes["x_m"].values[v_indices][None, :]
    by = proj_nodes["y_m"].values[v_indices][None, :]

    px = proj_track["x_m"].values[:, None]  # Shape: (T, 1)
    py = proj_track["y_m"].values[:, None]

    dx = bx - ax
    dy = by - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-6)

    # Vectorized orthogonal projection fraction alpha in [0, 1]
    alpha = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    qx = ax + alpha * dx
    qy = ay + alpha * dy

    # Exact orthogonal distance in meters
    cost_matrix = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)

    # Monotonic alignment via dynamic programming
    dp = np.full((t_len, r_len), np.inf)
    parent = np.zeros((t_len, r_len), dtype=int)

    # Initialize t = 0
    dp[0, :] = cost_matrix[0, :]

    for t in range(1, t_len):
        min_prev = np.inf
        best_r = 0
        for r in range(r_len):
            if dp[t - 1, r] < min_prev:
                min_prev = dp[t - 1, r]
                best_r = r
            dp[t, r] = min_prev + cost_matrix[t, r]
            parent[t, r] = best_r

    # Backtrack optimal sequence
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
