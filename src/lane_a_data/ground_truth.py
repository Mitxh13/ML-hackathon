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
    breaks = [i for i in range(len(arc_nodes)-1) if arc_nodes[i][1] != arc_nodes[i+1][0]]
    if breaks:
        raise ValueError(f"reference route has {len(breaks)} disconnected directed transition(s), first at route order {breaks[0]}")
    return np.asarray(endpoints, dtype=float)


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
                              max_alignment_distance_m: float = 250.0) -> pd.DataFrame:
    """Viterbi-style monotone route-order alignment with metric point/segment emissions.

    States are route *occurrences*, not unique arc IDs, so loops/repeated arcs retain
    chronology. Staying in state or advancing forward is allowed. Transitions are
    softly penalized for route progress inconsistent with observed displacement/time.
    Distances over ``max_alignment_distance_m`` receive a failed status, never a label.
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
    transformer, _ = local_transformer(np.r_[lons[valid_coords], nodes_df.lon.to_numpy(float)], np.r_[lats[valid_coords], nodes_df.lat.to_numpy(float)])
    safe_lons, safe_lats = np.where(valid_coords, lons, 0.0), np.where(valid_coords, lats, 0.0)
    points = np.column_stack(transformer.transform(safe_lons, safe_lats))
    segments = _segments(nodes_df, arcs_df, route_ids, transformer)
    dist = _point_segment_distances(points, segments)
    dist[~valid_coords, :] = np.inf
    # Robust emissions reduce the influence of occasional GPS outliers.
    emission = np.minimum(dist, 500.0) ** 2 / (2 * 25.0 ** 2)
    emission[~valid_coords, :] = 1e6
    route_lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    cumulative = np.r_[0.0, np.cumsum(route_lengths)]
    t_count, r_count = dist.shape
    dp = np.full((t_count, r_count), np.inf)
    back = np.zeros((t_count, r_count), dtype=np.int32)
    # The first GPS observation begins at the first ordered route occurrence.
    # This disambiguates repeated geometry at loops from a later, identical Arc ID.
    dp[0, 0] = emission[0, 0]
    timestamps = track_df.timestamp.to_numpy(float)
    timestamps = np.where(np.isfinite(timestamps), timestamps, np.nan)
    for t in range(1, t_count):
        dt = float(timestamps[t] - timestamps[t-1]) if np.isfinite(timestamps[t:t+1]).all() and np.isfinite(timestamps[t-1:t]).all() else 0.0
        dt = max(dt, 0.0)
        gps_step = float(np.linalg.norm(points[t] - points[t-1]))
        # Expected route progress is bounded by observed movement plus GPS tolerance.
        expected = min(gps_step, max(dt, 1.0) * 45.0)
        for r in range(r_count):
            # Route-distance window keeps the transition search close to O(T*R)
            # for long routes while retaining a wide allowance for noisy fixes/gaps.
            allowance = max(120.0, expected * 2.0 + 40.0)
            lo = int(np.searchsorted(cumulative[:r+1], cumulative[r] - expected - allowance, side="left"))
            prev = np.arange(lo, r + 1)
            progress = cumulative[r] - cumulative[prev]
            trans = ((progress - expected) / max(20.0, 8.0 + 0.35 * expected)) ** 2
            # Small per-occurrence advance penalty resolves intersections in route order.
            trans += np.maximum(r - prev - 1, 0) * 0.15
            costs = dp[t-1, prev] + trans
            p = int(np.argmin(costs))
            dp[t, r], back[t, r] = costs[p] + emission[t, r], int(prev[p])
    state = int(np.argmin(dp[-1]))
    states = [state]
    for t in range(t_count - 1, 0, -1):
        state = int(back[t, state]); states.append(state)
    states.reverse()
    chosen_dist = dist[np.arange(t_count), states]
    obs_idx = track_df["obs_idx"].to_numpy() if "obs_idx" in track_df else np.arange(t_count)
    valid = valid_coords & np.isfinite(points).all(axis=1) & (chosen_dist <= max_alignment_distance_m)
    labels = pd.array([route_ids[r] if ok else pd.NA for r, ok in zip(states, valid)], dtype="Int64")
    result = pd.DataFrame({
        "trip_id": str(trip_id), "obs_idx": obs_idx,
        "timestamp": track_df.timestamp.to_numpy(float), "gps_lon": lons, "gps_lat": lats,
        "true_arc_id": labels,
        "reference_route_order": [r if ok else pd.NA for r, ok in zip(states, valid)],
        "alignment_distance_m": chosen_dist,
        "alignment_status": np.where(valid, "aligned", "failed_distance_or_coordinate"),
    })
    validate_point_gt_df(result)
    return result
