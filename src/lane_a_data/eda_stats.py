"""Per-trip statistics for the PathSnap exploratory data analysis (part 1.1).

Everything here is read-only: it loads raw trips, measures them and returns
plain numbers. Plots live in ``eda_plots.py``; ``run_eda.py`` ties both
together and writes the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.lane_a_data.loaders import TRIP_FILE_TYPES, load_trip, validate_trip
from src.lane_a_data.preprocessing import (
    HEADING_HALF_WINDOW,
    HEADING_MIN_DISP_M,
    LARGE_GAP_S,
    SMOOTH_WINDOW,
    point_segment_distance,
    preprocess_track,
    project_coordinates,
    smooth_xy,
    windowed_motion,
)

# Timestamps carry floating-point noise (0.9999999999999 instead of 1), so
# "exactly 1 s" is tested with a tolerance.
DT_TOLERANCE_S = 1e-3
GAP_LEVELS_S = (LARGE_GAP_S, 30.0, 300.0)
ZERO_LENGTH_ARC_M = 0.01
REVERSAL_TURN_RAD = np.radians(150.0)

# Heading methods compared against the direction of the nearest reference-route arc.
# (label, smoothing window in points, half-width of the difference window)
HEADING_METHODS = [
    ("Raw, consecutive points", 1, 0),
    ("Raw, ±2 points", 1, 2),
    ("Raw, ±3 points", 1, 3),
    ("Smoothed 5, ±1 points", 5, 1),
    ("Smoothed 5, ±2 points", 5, 2),
    (f"Smoothed {SMOOTH_WINDOW}, ±{HEADING_HALF_WINDOW} points (chosen)", SMOOTH_WINDOW, HEADING_HALF_WINDOW),
    ("Smoothed 9, ±4 points", 9, 4),
]
HEADING_STUDY_MAX_DIST_M = 15.0


@dataclass
class ProjectedTrip:
    """A loaded trip with its track preprocessed and nodes projected into the same CRS."""

    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None
    crs: str
    raw: dict

    @property
    def arc_xy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Start and end coordinates (metres) of every arc: ax, ay, bx, by."""
        x = self.nodes["x_m"].to_numpy()
        y = self.nodes["y_m"].to_numpy()
        u = self.arcs["from_node_id"].to_numpy()
        v = self.arcs["to_node_id"].to_numpy()
        return x[u], y[u], x[v], y[v]

    @property
    def arc_length_m(self) -> np.ndarray:
        ax, ay, bx, by = self.arc_xy
        return np.hypot(bx - ax, by - ay)


def load_projected_trip(trip_dir: str | Path, partition: str) -> ProjectedTrip:
    """Load a trip, preprocess its track and project its nodes into the track's UTM zone."""
    trip_dir = Path(trip_dir)
    raw = load_trip(trip_dir)
    track = preprocess_track(raw["track"])
    crs = track.attrs["crs"]
    nodes = project_coordinates(raw["nodes"], lon_col="lon", lat_col="lat", crs_to=crs)
    return ProjectedTrip(trip_dir.name, partition, track, nodes, raw["arcs"], raw.get("route"), crs, raw)


def file_sizes(trip_dir: str | Path) -> dict[str, int]:
    trip_dir = Path(trip_dir)
    sizes = {}
    for ext in TRIP_FILE_TYPES:
        path = trip_dir / f"{trip_dir.name}.{ext}"
        sizes[f"{ext}_bytes"] = path.stat().st_size if path.exists() else 0
    return sizes


def sampling_stats(track: pd.DataFrame) -> dict:
    """Sampling interval statistics for one track."""
    ts = track["timestamp"].to_numpy()
    dt = np.diff(ts)
    if dt.size == 0:
        return {"n_steps": 0}
    out = {
        "n_steps": int(dt.size),
        "duration_s": float(ts[-1] - ts[0]),
        "first_timestamp": float(ts[0]),
        "share_dt_1s": float(np.mean(np.abs(dt - 1.0) < DT_TOLERANCE_S)),
        "dt_median_s": float(np.median(dt)),
        "dt_max_s": float(dt.max()),
        "non_integer_timestamps": int(np.sum(np.abs(ts - np.round(ts)) > DT_TOLERANCE_S)),
        "gap_time_s": float(np.sum(dt[dt > LARGE_GAP_S] - 1.0)),
    }
    for level in GAP_LEVELS_S:
        out[f"gaps_over_{int(level)}s"] = int(np.sum(dt > level))
    return out


def jitter_reversals(track: pd.DataFrame) -> tuple[int, float]:
    """Count and share of consecutive raw steps (both at least 1 m) that turn by more than 150 degrees.

    Real vehicles almost never do this from one second to the next, so a high
    share means the raw positions zigzag around the true path.
    """
    dx = np.diff(track["x_m"].to_numpy())
    dy = np.diff(track["y_m"].to_numpy())
    if dx.size < 2:
        return 0, 0.0
    h = np.arctan2(dy, dx)
    turn = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
    real = (np.hypot(dx, dy)[1:] >= 1.0) & (np.hypot(dx, dy)[:-1] >= 1.0)
    if not real.any():
        return 0, 0.0
    reversed_ = turn[real] > REVERSAL_TURN_RAD
    return int(reversed_.sum()), float(reversed_.mean())


def kinematic_stats(track: pd.DataFrame) -> dict:
    """Distance, speed and stationarity statistics for one preprocessed track."""
    speed = track["speed_mps"].to_numpy()
    plausible = ~track["is_speed_outlier"].to_numpy()
    moving = ~track["is_stationary"].to_numpy()
    lon, lat = track["gps_lon"].to_numpy(), track["gps_lat"].to_numpy()
    repeated = np.sum((np.diff(lon) == 0) & (np.diff(lat) == 0)) if len(track) > 1 else 0
    moving_speed = speed[moving & plausible]
    smooth_steps = np.hypot(np.diff(track["x_smooth_m"].to_numpy()), np.diff(track["y_smooth_m"].to_numpy()))
    n_reversals, reversal_share = jitter_reversals(track)
    return {
        "path_length_km": float(smooth_steps[plausible[1:]].sum() / 1000.0),
        "path_length_raw_km": float(track["step_dist_m"].to_numpy()[plausible].sum() / 1000.0),
        "jitter_reversals": n_reversals,
        "jitter_reversal_share": reversal_share,
        "speed_median_kmh": float(np.median(moving_speed) * 3.6) if moving_speed.size else 0.0,
        "speed_p95_kmh": float(np.percentile(moving_speed, 95) * 3.6) if moving_speed.size else 0.0,
        "speed_raw_max_kmh": float(track["speed_raw_mps"].max() * 3.6) if len(track) else 0.0,
        "speed_outliers": int((~plausible).sum()),
        "share_stationary": float(np.mean(track["is_stationary"])),
        "repeated_positions": int(repeated),
    }


def extent_stats(trip: ProjectedTrip) -> dict:
    t = trip.track
    return {
        "crs": trip.crs,
        "centroid_lon": float(t["gps_lon"].median()),
        "centroid_lat": float(t["gps_lat"].median()),
        "bbox_width_km": float((t["x_m"].max() - t["x_m"].min()) / 1000.0),
        "bbox_height_km": float((t["y_m"].max() - t["y_m"].min()) / 1000.0),
    }


def network_stats(trip: ProjectedTrip) -> dict:
    """Size and connectivity of the trip's road network.

    Components are counted over nodes that appear in at least one arc, so
    isolated nodes are reported separately instead of inflating the counts.
    """
    n_nodes = len(trip.nodes)
    u = trip.arcs["from_node_id"].to_numpy()
    v = trip.arcs["to_node_id"].to_numpy()
    used = np.zeros(n_nodes, dtype=bool)
    used[u] = True
    used[v] = True
    n_used = int(used.sum())

    graph = coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=(n_nodes, n_nodes)).tocsr()
    _, weak = connected_components(graph, directed=True, connection="weak")
    _, strong = connected_components(graph, directed=True, connection="strong")
    weak_used, strong_used = weak[used], strong[used]
    weak_sizes = np.bincount(weak_used) if n_used else np.array([0])
    strong_sizes = np.bincount(strong_used) if n_used else np.array([0])

    keys = u.astype(np.int64) * n_nodes + v
    reverse_keys = v.astype(np.int64) * n_nodes + u
    out_deg = np.bincount(u, minlength=n_nodes)
    in_deg = np.bincount(v, minlength=n_nodes)
    lengths = trip.arc_length_m

    return {
        "isolated_nodes": int(n_nodes - n_used),
        "weak_components": int(np.count_nonzero(weak_sizes)),
        "largest_weak_share": float(weak_sizes.max() / n_used) if n_used else 0.0,
        "strong_components": int(np.count_nonzero(strong_sizes)),
        "largest_strong_share": float(strong_sizes.max() / n_used) if n_used else 0.0,
        "two_way_arc_share": float(np.isin(reverse_keys, keys).mean()) if len(u) else 0.0,
        "dead_end_nodes": int(np.sum((out_deg == 0) & (in_deg > 0))),
        "arc_length_median_m": float(np.median(lengths)) if lengths.size else 0.0,
        "arc_length_p95_m": float(np.percentile(lengths, 95)) if lengths.size else 0.0,
        "arc_length_max_m": float(lengths.max()) if lengths.size else 0.0,
        "zero_length_arcs": int(np.sum(lengths < ZERO_LENGTH_ARC_M)),
    }


def route_stats(trip: ProjectedTrip) -> dict:
    """Structure of the reference route: repeats, continuity and length."""
    if trip.route is None or trip.route.empty:
        return {"route_available": False}
    r = trip.route["arc_id"].to_numpy()
    u = trip.arcs["from_node_id"].to_numpy()
    v = trip.arcs["to_node_id"].to_numpy()
    same = r[1:] == r[:-1]
    joined = v[r[:-1]] == u[r[1:]]
    transitions = ~same
    return {
        "route_available": True,
        "route_unique_arcs": int(np.unique(r).size),
        "route_consecutive_repeats": int(same.sum()),
        "route_revisited_arcs": int(r.size - np.unique(r).size - same.sum()),
        "route_breaks": int(np.sum(transitions & ~joined)),
        "route_connected_share": float(joined[transitions].mean()) if transitions.any() else 1.0,
        "route_length_km": float(trip.arc_length_m[r].sum() / 1000.0),
    }


def nearest_route_arc(trip: ProjectedTrip, chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from every GPS point to the closest arc of its reference route, and that arc's ID.

    This measures GPS noise relative to the reference route. It is not the
    label assignment (part 1.2), which also enforces the route's order.
    """
    n = len(trip.track)
    if trip.route is None or trip.route.empty:
        return np.full(n, np.nan), np.full(n, -1)
    r = np.unique(trip.route["arc_id"].to_numpy())
    ax, ay, bx, by = (a[r][None, :] for a in trip.arc_xy)
    px = trip.track["x_m"].to_numpy()
    py = trip.track["y_m"].to_numpy()
    dist = np.empty(n)
    arc = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        d, _ = point_segment_distance(px[sl, None], py[sl, None], ax, ay, bx, by)
        best = d.argmin(axis=1)
        dist[sl] = d[np.arange(len(best)), best]
        arc[sl] = r[best]
    return dist, arc


def nearest_network_distance(trip: ProjectedTrip, spacing_m: float = 5.0, margin_m: float = 250.0) -> np.ndarray:
    """Approximate distance (m) from every GPS point to the closest arc anywhere in the trip's network.

    Arcs near the track are sampled every ``spacing_m`` metres and indexed in a
    KD-tree, so the result is within ``spacing_m / 2`` of the exact distance.
    Distances beyond ``margin_m`` are reported as ``margin_m``.
    """
    ax, ay, bx, by = trip.arc_xy
    t = trip.track
    x0, x1 = t["x_m"].min() - margin_m, t["x_m"].max() + margin_m
    y0, y1 = t["y_m"].min() - margin_m, t["y_m"].max() + margin_m
    keep = (((ax >= x0) & (ax <= x1) & (ay >= y0) & (ay <= y1))
            | ((bx >= x0) & (bx <= x1) & (by >= y0) & (by <= y1)))
    if not keep.any():
        return np.full(len(t), margin_m)
    ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
    n_samples = np.maximum(np.ceil(np.hypot(bx - ax, by - ay) / spacing_m).astype(int), 1) + 1
    arc_of_sample = np.repeat(np.arange(ax.size), n_samples)
    frac = np.concatenate([np.linspace(0.0, 1.0, k) for k in n_samples])
    sx = ax[arc_of_sample] + frac * (bx - ax)[arc_of_sample]
    sy = ay[arc_of_sample] + frac * (by - ay)[arc_of_sample]
    dist, _ = cKDTree(np.column_stack([sx, sy])).query(
        np.column_stack([t["x_m"].to_numpy(), t["y_m"].to_numpy()]), distance_upper_bound=margin_m
    )
    return np.minimum(dist, margin_m)


def heading_errors(trip: ProjectedTrip, dist: np.ndarray, arc: np.ndarray) -> dict[str, np.ndarray]:
    """Angle (rad) between each heading method and the nearest route arc's direction.

    Only points within 15 m of the route that moved at least 2 m over the
    window count. Junctions can still match the wrong route arc, so these
    errors are an upper bound.
    """
    if trip.route is None or trip.route.empty:
        return {label: np.array([]) for label, _, _ in HEADING_METHODS}
    ax, ay, bx, by = trip.arc_xy
    arc_heading = np.arctan2((by - ay)[arc], (bx - ax)[arc])
    t = trip.track
    x, y, ts = t["x_m"].to_numpy(), t["y_m"].to_numpy(), t["timestamp"].to_numpy()
    seg = t["segment_id"].to_numpy()
    out = {}
    for label, window, half in HEADING_METHODS:
        xs, ys = smooth_xy(x, y, seg, window)
        heading, disp, _ = windowed_motion(xs, ys, ts, seg, half)
        keep = (dist < HEADING_STUDY_MAX_DIST_M) & (disp >= HEADING_MIN_DISP_M)
        out[label] = np.abs((heading - arc_heading + np.pi) % (2 * np.pi) - np.pi)[keep]
    return out


def trip_statistics(trip_dir: str | Path, partition: str) -> tuple[dict, dict]:
    """All EDA numbers for one trip.

    Returns:
        (row, arrays): ``row`` is one flat record for the inventory table;
        ``arrays`` holds per-point values (time steps, speeds, distances to
        the route) for the distribution plots.
    """
    trip = load_projected_trip(trip_dir, partition)
    row = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "n_gps_points": len(trip.track),
        "n_nodes": len(trip.nodes),
        "n_arcs": len(trip.arcs),
        "n_route_arcs": 0 if trip.route is None else len(trip.route),
    }
    row.update(file_sizes(trip_dir))
    row.update({f"issue_{k}": v for k, v in validate_trip(trip.raw).items()})
    row.update(sampling_stats(trip.track))
    row.update(kinematic_stats(trip.track))
    row.update(extent_stats(trip))
    row.update(network_stats(trip))
    row.update(route_stats(trip))

    dist, arc = nearest_route_arc(trip)
    net_dist = nearest_network_distance(trip)
    row.update({
        "gps_network_dist_median_m": float(np.median(net_dist)),
        "gps_network_dist_p95_m": float(np.percentile(net_dist, 95)),
        "points_off_network_50m": int(np.sum(net_dist > 50.0)),
    })
    if np.isfinite(dist).any():
        far = dist > 50.0
        row.update({
            "gps_route_dist_median_m": float(np.nanmedian(dist)),
            "gps_route_dist_p95_m": float(np.nanpercentile(dist, 95)),
            "gps_route_dist_max_m": float(np.nanmax(dist)),
            "share_points_over_50m": float(np.nanmean(far)),
            # far from the route but on some road: the reference route skips part of the drive
            "points_off_route_on_road": int(np.sum(far & (net_dist <= 50.0))),
        })
    arrays = {
        "dt": np.diff(trip.track["timestamp"].to_numpy()),
        "speed_kmh": trip.track.loc[
            ~trip.track["is_stationary"] & ~trip.track["is_speed_outlier"], "speed_mps"
        ].to_numpy() * 3.6,
        "route_dist_m": dist,
        "heading_err": heading_errors(trip, dist, arc),
    }
    return row, arrays
