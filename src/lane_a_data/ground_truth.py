"""Sequence-aware alignment of GPS observations to an ordered reference route."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from src.contracts.schemas import POINT_GT_COLS, validate_point_gt_df


def point_to_segment_distance(
    p_lon: float, p_lat: float, u_lon: float, u_lat: float, v_lon: float, v_lat: float,
) -> float:
    """Return exact local-metric distance from a WGS84 point to a segment in metres."""
    coords_lon = np.asarray([p_lon, u_lon, v_lon], dtype=float)
    coords_lat = np.asarray([p_lat, u_lat, v_lat], dtype=float)
    if not (np.isfinite(coords_lon).all() and np.isfinite(coords_lat).all()):
        raise ValueError("point and segment coordinates must be finite")
    if (np.abs(coords_lon) > 180).any() or (np.abs(coords_lat) > 90).any():
        raise ValueError("coordinates must be valid WGS84 longitude/latitude")
    transformer, _ = local_transformer(coords_lon, coords_lat)
    x, y = transformer.transform(coords_lon, coords_lat)
    return float(_point_segment_distances(np.asarray([[x[0], y[0]]]), np.asarray([[[x[1], y[1]], [x[2], y[2]]]]))[0, 0])


def local_transformer(lons: np.ndarray, lats: np.ndarray) -> tuple[Transformer, str]:
    """Choose a trip-local WGS84 / UTM CRS and return its forward transformer."""
    finite = np.isfinite(lons) & np.isfinite(lats) & (np.abs(lons) <= 180) & (np.abs(lats) <= 90)
    if not finite.any():
        raise ValueError("No valid WGS84 coordinates available to select a local CRS")
    lon, lat = float(np.median(lons[finite])), float(np.median(lats[finite]))
    zone = int(np.floor((lon + 180) / 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + min(max(zone, 1), 60)
    crs = CRS.from_epsg(epsg)
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True), crs.to_string()


def _segments(nodes: pd.DataFrame, arcs: pd.DataFrame, route_ids: list[int], transformer: Transformer):
    """Projected (from, to) endpoints of each route occurrence and the number of route breaks.

    A break is a consecutive pair of route arcs that does not share a node. Some raw routes
    skip short pieces of road; those routes are still aligned (the order is still known), and
    the break count is reported instead of rejecting the whole trip.
    """
    nlon, nlat = nodes["lon"].to_numpy(float), nodes["lat"].to_numpy(float)
    if not (np.isfinite(nlon).all() and np.isfinite(nlat).all() and (np.abs(nlon) <= 180).all() and (np.abs(nlat) <= 90).all()):
        raise ValueError("nodes contain invalid or non-finite WGS84 coordinates")
    xy = np.column_stack(transformer.transform(nlon, nlat))
    endpoints = []
    arc_nodes = []
    for arc_id in route_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            raise ValueError(f"route arc_id {arc_id} is outside arcs table")
        u, v = int(arcs.iloc[arc_id]["from_node_id"]), int(arcs.iloc[arc_id]["to_node_id"])
        if min(u, v) < 0 or max(u, v) >= len(nodes):
            raise ValueError(f"arc_id {arc_id} references missing node ({u}, {v})")
        endpoints.append((xy[u], xy[v]))
        arc_nodes.append((u, v))
    breaks = sum(arc_nodes[i][1] != arc_nodes[i + 1][0] for i in range(len(arc_nodes) - 1))
    return np.asarray(endpoints, dtype=float), int(breaks)


def _track_heading(track_df: pd.DataFrame, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Travel heading (radians, east = 0) and whether it can be trusted.

    Uses the smoothed heading from ``preprocess_track`` when present; otherwise a central
    difference of the projected points, trusted only when the points are >= 2 m apart.
    """
    if {"heading_rad", "heading_valid"}.issubset(track_df.columns):
        heading = track_df["heading_rad"].to_numpy(float)
        valid = track_df["heading_valid"].to_numpy(bool) & np.isfinite(heading)
        return np.where(valid, heading, 0.0), valid
    n = len(points)
    if n < 2:
        return np.zeros(n), np.zeros(n, dtype=bool)
    nxt = np.r_[points[1:], points[-1:]]
    prv = np.r_[points[:1], points[:-1]]
    step = nxt - prv
    valid = np.isfinite(step).all(axis=1) & (np.hypot(step[:, 0], step[:, 1]) >= 2.0)
    heading = np.arctan2(np.nan_to_num(step[:, 1]), np.nan_to_num(step[:, 0]))
    return heading, valid


def _point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    delta = points[:, None, :] - a[None, :, :]
    fraction = np.einsum("tij,ij->ti", delta, ab) / np.where(denom == 0, 1, denom)[None, :]
    fraction = np.clip(fraction, 0, 1)
    projected = a[None, :, :] + fraction[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - projected, axis=2)


def assign_point_ground_truth(trip_id: str, track_df: pd.DataFrame, route_df: pd.DataFrame,
                              nodes_df: pd.DataFrame, arcs_df: pd.DataFrame,
                              max_alignment_distance_m: float = 100.0,
                              sigma_m: float = 25.0,
                              heading_weight: float = 1.0) -> pd.DataFrame:
    """Monotone alignment of every GPS point to one occurrence of the ordered ``.route``.

    States are route *occurrences* (positions in ``.route``), not arc IDs, so loops and
    repeated arcs keep their chronology. Between two GPS points the alignment may stay on
    the same occurrence or move forward any number of occurrences, never backwards.

    Cost of putting point t on occurrence r (lower is better):
        min(d, 500)^2 / (2 sigma^2)                    metric point-to-segment distance
      + heading_weight * (1 - cos(heading - arc_dir)) / 2   only when the heading is trusted

    The heading term only matters when two occurrences are equally close, e.g. the two
    directions of a two-way street, and makes the label follow the direction of travel.
    The optimum is found exactly with dynamic programming: dp[t, r] = cost[t, r] +
    min over r' <= r of dp[t-1, r'], computed with a running minimum (O(T x R)).

    Points further than ``max_alignment_distance_m`` from their occurrence, or with invalid
    coordinates, keep their row but get no label (``true_arc_id`` is NA). The default (100 m)
    equals the candidate search radius: a reference arc further away than that can never be
    a candidate, and on the development trips such distances only occur where the reference
    route does not describe where the vehicle drove (trip 54's map mismatch, trip 33's
    incomplete route), so those points are left unlabeled instead of becoming noise.
    """
    required = {"gps_lon", "gps_lat", "timestamp"}
    if not required.issubset(track_df.columns):
        raise ValueError(f"track missing columns: {sorted(required - set(track_df.columns))}")
    route_ids = route_df["arc_id"].astype(int).tolist()
    if not route_ids:
        raise ValueError(f"Route for trip {trip_id} is empty")
    if track_df.empty:
        return pd.DataFrame(columns=POINT_GT_COLS + ["reference_route_order", "alignment_distance_m", "alignment_status"])
    lons, lats = track_df.gps_lon.to_numpy(float), track_df.gps_lat.to_numpy(float)
    valid_coords = np.isfinite(lons) & np.isfinite(lats) & (np.abs(lons) <= 180) & (np.abs(lats) <= 90)
    transformer, _ = local_transformer(np.r_[lons[valid_coords], nodes_df.lon.to_numpy(float)],
                                       np.r_[lats[valid_coords], nodes_df.lat.to_numpy(float)])
    safe_lons, safe_lats = np.where(valid_coords, lons, 0.0), np.where(valid_coords, lats, 0.0)
    points = np.column_stack(transformer.transform(safe_lons, safe_lats))
    segments, route_breaks = _segments(nodes_df, arcs_df, route_ids, transformer)

    dist = _point_segment_distances(points, segments)
    cost = np.minimum(dist, 500.0) ** 2 / (2.0 * sigma_m ** 2)
    heading, heading_ok = _track_heading(track_df, np.where(valid_coords[:, None], points, np.nan))
    arc_dir = np.arctan2(segments[:, 1, 1] - segments[:, 0, 1], segments[:, 1, 0] - segments[:, 0, 0])
    turn = 0.5 * (1.0 - np.cos(heading[:, None] - arc_dir[None, :]))
    cost += heading_weight * np.where(heading_ok[:, None], turn, 0.0)
    cost[~valid_coords, :] = 0.0  # invalid points do not constrain the path; they stay unlabeled

    t_count, r_count = cost.shape
    back = np.zeros((t_count, r_count), dtype=np.int32)
    idx = np.arange(r_count)
    dp = cost[0].copy()
    for t in range(1, t_count):
        run_min = np.minimum.accumulate(dp)
        # earliest occurrence that reaches the running minimum (ties keep the earlier one)
        new_min = np.r_[True, dp[1:] < run_min[:-1]]
        back[t] = np.maximum.accumulate(np.where(new_min, idx, 0))
        dp = run_min + cost[t]
    states = np.empty(t_count, dtype=np.int64)
    states[-1] = int(np.argmin(dp))
    for t in range(t_count - 1, 0, -1):
        states[t - 1] = back[t, states[t]]

    chosen_dist = dist[np.arange(t_count), states]
    obs_idx = track_df["obs_idx"].to_numpy() if "obs_idx" in track_df else np.arange(t_count)
    valid = valid_coords & np.isfinite(points).all(axis=1) & (chosen_dist <= max_alignment_distance_m)
    route_arr = np.asarray(route_ids, dtype=np.int64)
    labels = pd.array(np.where(valid, route_arr[states], 0), dtype="Int64")
    labels[~valid] = pd.NA
    order = pd.array(states, dtype="Int64")
    order[~valid] = pd.NA
    result = pd.DataFrame({
        "trip_id": str(trip_id), "obs_idx": obs_idx,
        "timestamp": track_df.timestamp.to_numpy(float), "gps_lon": lons, "gps_lat": lats,
        "true_arc_id": labels,
        "reference_route_order": order,
        "alignment_distance_m": np.where(valid_coords, chosen_dist, np.nan),
        "alignment_status": np.where(valid, "aligned", "failed_distance_or_coordinate"),
    })
    result.attrs["route_breaks"] = route_breaks
    validate_point_gt_df(result)
    return result
