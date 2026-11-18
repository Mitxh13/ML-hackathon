"""Spatial candidate generation for GPS observations using metric spatial indexing."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_a_data.preprocessing import project_coordinates
from src.lane_c_hmm.road_graph import haversine_distance_m


class CandidateGenerator:
    """Generates candidate road segments for each GPS observation.

    Attributes:
        max_dist_m: Maximum search radius in meters (default: 100.0).
        max_k: Maximum candidate segments to retain per observation (default: 10).
    """

    def __init__(self, max_dist_m: float = 100.0, max_k: int = 10) -> None:
        self.max_dist_m = max_dist_m
        self.max_k = max_k

    def generate_candidates(
        self,
        trip_id: str,
        track_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        point_gt_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Find candidate road segments for each GPS observation in a trip.

        Args:
            trip_id: Identifier of the trip.
            track_df: Track DataFrame with ['gps_lon', 'gps_lat', 'timestamp'].
            nodes_df: Road nodes DataFrame.
            arcs_df: Road arcs DataFrame.
            point_gt_df: Optional ground-truth point table with 'true_arc_id'.

        Returns:
            DataFrame adhering to Contract 1 (CANDIDATE_BASE_COLS).
        """
        # Ensure projected coordinates
        proj_nodes = project_coordinates(nodes_df, lon_col="lon", lat_col="lat")
        proj_track = project_coordinates(track_df, lon_col="gps_lon", lat_col="gps_lat")

        # Compute midpoints of all arcs in projected meters
        u_nodes = arcs_df.iloc[:, 0].values
        v_nodes = arcs_df.iloc[:, 1].values

        u_x = proj_nodes.iloc[u_nodes]["x_m"].values
        u_y = proj_nodes.iloc[u_nodes]["y_m"].values
        v_x = proj_nodes.iloc[v_nodes]["x_m"].values
        v_y = proj_nodes.iloc[v_nodes]["y_m"].values

        arc_mid_x = (u_x + v_x) / 2.0
        arc_mid_y = (u_y + v_y) / 2.0
        arc_tree = cKDTree(np.column_stack([arc_mid_x, arc_mid_y]))

        # Map true_arc_id by obs_idx if ground truth provided
        gt_map = {}
        if point_gt_df is not None and "true_arc_id" in point_gt_df.columns:
            gt_map = dict(zip(point_gt_df["obs_idx"], point_gt_df["true_arc_id"]))

        records = []
        track_x = proj_track["x_m"].values
        track_y = proj_track["y_m"].values
        lons = track_df["gps_lon"].values
        lats = track_df["gps_lat"].values
        times = track_df["timestamp"].values
        obs_indices = track_df["obs_idx"].values if "obs_idx" in track_df.columns else np.arange(len(track_df))

        # Query KDTree
        # Use slightly larger search window in tree to ensure coverage after perpendicular refinement
        search_radius = self.max_dist_m * 1.5

        for t in range(len(track_df)):
            pt_x, pt_y = track_x[t], track_y[t]
            p_lon, p_lat = lons[t], lats[t]
            obs_idx = int(obs_indices[t])
            true_arc = gt_map.get(obs_idx, None)

            # Query nearby arc midpoints
            nearby_indices = arc_tree.query_ball_point([pt_x, pt_y], r=search_radius)

            if not nearby_indices:
                # Fallback to nearest neighbor
                _, nearest = arc_tree.query([pt_x, pt_y], k=1)
                nearby_indices = [int(nearest)]

            # Compute precise geodesic distances to candidate segments
            scored_candidates = []
            for arc_id in nearby_indices:
                u_id = int(u_nodes[arc_id])
                v_id = int(v_nodes[arc_id])
                u_lon = float(nodes_df.iloc[u_id]["lon"])
                u_lat = float(nodes_df.iloc[u_id]["lat"])
                v_lon = float(nodes_df.iloc[v_id]["lon"])
                v_lat = float(nodes_df.iloc[v_id]["lat"])

                # Distance to segment endpoints and midpoint
                d_u = haversine_distance_m(p_lon, p_lat, u_lon, u_lat)
                d_v = haversine_distance_m(p_lon, p_lat, v_lon, v_lat)
                d_mid = haversine_distance_m(p_lon, p_lat, (u_lon + v_lon) / 2.0, (u_lat + v_lat) / 2.0)
                dist_m = min(d_u, d_v, d_mid)

                if dist_m <= self.max_dist_m:
                    scored_candidates.append((dist_m, arc_id, u_id, v_id))

            if not scored_candidates:
                # If all were beyond max_dist_m, take the closest 1
                arc_id = nearby_indices[0]
                scored_candidates.append(
                    (search_radius, arc_id, int(u_nodes[arc_id]), int(v_nodes[arc_id]))
                )

            # Sort by distance and retain top k
            scored_candidates.sort(key=lambda x: x[0])
            top_k = scored_candidates[: self.max_k]

            for dist_m, arc_id, u_id, v_id in top_k:
                if true_arc is not None:
                    is_target = 1 if arc_id == true_arc else 0
                else:
                    is_target = -1

                records.append(
                    {
                        "trip_id": str(trip_id),
                        "obs_idx": obs_idx,
                        "timestamp": float(times[t]),
                        "gps_lon": float(p_lon),
                        "gps_lat": float(p_lat),
                        "candidate_arc_id": int(arc_id),
                        "from_node_id": int(u_id),
                        "to_node_id": int(v_id),
                        "dist_to_arc_m": float(dist_m),
                        "is_target": int(is_target),
                    }
                )

        candidate_df = pd.DataFrame(records)
        validate_candidate_df(candidate_df)
        return candidate_df
