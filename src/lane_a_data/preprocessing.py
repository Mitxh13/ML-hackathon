"""GPS trajectory cleaning, coordinate projection, speed calculation, and gap detection.

Projection choice
-----------------
The PathSnap trips are spread across the world (longitude -100 to 137,
latitude -35 to 62), so no single metric CRS fits all of them. Each trip is
projected into the UTM zone that contains it (scale error at most about 0.1%
inside a zone). Web Mercator (EPSG:3857) is not used: it stretches distances
by 1 / cos(latitude), which is about 1.55x at 50 degrees and 2.1x at 62 degrees.

Heading convention
------------------
``heading_rad`` is the direction of travel in projected metres, measured
counter-clockwise from east (the ``np.arctan2(dy, dx)`` convention), in
(-pi, pi]. Any arc bearing compared against it must use the same convention
and the same projected coordinates.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

try:  # pyproj is the reference implementation when its compiled library loads.
    from pyproj import Transformer
except ImportError:  # e.g. Windows Smart App Control blocking the pyproj DLL
    Transformer = None

EARTH_RADIUS_M = 6_371_008.8
LARGE_GAP_S = 5.0
MAX_PLAUSIBLE_SPEED_MPS = 70.0  # 252 km/h; anything faster is a GPS jump

# Motion is measured on a smoothed track. In slow urban trips 20-30% of raw
# 1 Hz steps reverse direction because of position jitter, so raw
# point-to-point heading is wrong about 17% of the time on the training trips.
# A 9-point rolling mean plus a +/-2 point central difference brings that to
# about 6% (see results/tables/eda_heading_smoothing.csv, training trips only).
SMOOTH_WINDOW = 9
HEADING_HALF_WINDOW = 2
HEADING_MIN_DISP_M = 2.0  # less movement than this over the window: heading unknown
STATIONARY_SPEED_MPS = 0.5  # 1.8 km/h

_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_UTM_K0 = 0.9996


def utm_crs_for(lon: float, lat: float) -> str:
    """Return the EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def metric_crs_for(df: pd.DataFrame, lon_col: str = "gps_lon", lat_col: str = "gps_lat") -> str:
    """Pick the UTM zone for a table of coordinates from its median position."""
    return utm_crs_for(float(np.median(df[lon_col])), float(np.median(df[lat_col])))


@lru_cache(maxsize=64)
def _pyproj_transformer(crs_to: str):
    return Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)


def _utm_forward_numpy(lon: np.ndarray, lat: np.ndarray, crs_to: str) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 -> UTM using the Krueger series (millimetre accuracy within a zone)."""
    code = int(crs_to.split(":")[1])
    south = code >= 32700
    zone = code - (32700 if south else 32600)
    if not 1 <= zone <= 60:
        raise ValueError(f"{crs_to} is not a WGS84 UTM zone")

    n = _WGS84_F / (2.0 - _WGS84_F)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0)
    alpha = (
        n / 2.0 - 2.0 * n**2 / 3.0 + 5.0 * n**3 / 16.0,
        13.0 * n**2 / 48.0 - 3.0 * n**3 / 5.0,
        61.0 * n**3 / 240.0,
    )
    ecc = 2.0 * np.sqrt(n) / (1.0 + n)

    phi = np.radians(lat)
    dlam = np.radians(lon) - np.radians((zone - 1) * 6 - 180 + 3)
    t = np.sinh(np.arctanh(np.sin(phi)) - ecc * np.arctanh(ecc * np.sin(phi)))
    xi = np.arctan2(t, np.cos(dlam))
    eta = np.arctanh(np.sin(dlam) / np.sqrt(1.0 + t**2))

    easting = eta.copy()
    northing = xi.copy()
    for j, a_j in enumerate(alpha, start=1):
        easting += a_j * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        northing += a_j * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    x = 500_000.0 + _UTM_K0 * big_a * easting
    y = (10_000_000.0 if south else 0.0) + _UTM_K0 * big_a * northing
    return x, y


def lonlat_to_xy(lon, lat, crs_to: str) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude arrays (degrees) to metres in ``crs_to``."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if Transformer is not None:
        x, y = _pyproj_transformer(crs_to).transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return _utm_forward_numpy(lon, lat, crs_to)


def project_coordinates(
    df: pd.DataFrame,
    lon_col: str = "gps_lon",
    lat_col: str = "gps_lat",
    crs_to: Optional[str] = None,
) -> pd.DataFrame:
    """Add projected metre coordinates ``x_m`` and ``y_m`` to a copy of ``df``.

    Args:
        df: Input DataFrame containing angular coordinates.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.
        crs_to: Target metric CRS. ``None`` picks the UTM zone of the data. Pass
            the same CRS for a trip's track and nodes so their metres line up.

    Returns:
        DataFrame with additional columns ['x_m', 'y_m']; the CRS used is
        stored in ``out_df.attrs['crs']``.
    """
    out_df = df.copy()
    if out_df.empty:
        out_df["x_m"] = pd.Series(dtype=float)
        out_df["y_m"] = pd.Series(dtype=float)
        out_df.attrs["crs"] = crs_to
        return out_df

    if crs_to is None:
        crs_to = metric_crs_for(out_df, lon_col, lat_col)
    x_coords, y_coords = lonlat_to_xy(out_df[lon_col].to_numpy(), out_df[lat_col].to_numpy(), crs_to)
    out_df["x_m"] = x_coords
    out_df["y_m"] = y_coords
    out_df.attrs["crs"] = crs_to
    return out_df


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres; accepts scalars or numpy arrays."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_segment_distance(px, py, ax, ay, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Distance from points P to segments AB in projected metres (numpy broadcasting).

    Returns:
        (distance_m, t) where t in [0, 1] is the position of the closest point
        along AB (0 = at A / from-node, 1 = at B / to-node).
    """
    px, py, ax, ay, bx, by = (np.asarray(v, dtype=float) for v in (px, py, ax, ay, bx, by))
    abx, aby = bx - ax, by - ay
    len_sq = abx**2 + aby**2
    safe_len_sq = np.where(len_sq > 0, len_sq, 1.0)
    t = ((px - ax) * abx + (py - ay) * aby) / safe_len_sq
    t = np.where(len_sq > 0, np.clip(t, 0.0, 1.0), 0.0)
    dist = np.hypot(px - (ax + t * abx), py - (ay + t * aby))
    return dist, t


def _segment_bounds(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every point, the first and last index of its gap-free segment."""
    segment = np.asarray(segment)
    n = segment.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    ends = np.r_[starts[1:] - 1, n - 1]
    which = np.repeat(np.arange(starts.size), ends - starts + 1)
    return starts[which], ends[which]


def smooth_xy(x: np.ndarray, y: np.ndarray, segment: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving average of positions that never averages across a time gap.

    ``segment`` labels gap-free stretches of the track (it increases by one
    after every large gap). Near a segment edge the window shrinks
    symmetrically (e.g. 9 -> 7 -> 5 -> 3 -> 1 points) so it stays centred and
    does not pull positions towards the middle of the segment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if window <= 1 or x.size == 0:
        return x, y
    idx = np.arange(x.size)
    seg_start, seg_end = _segment_bounds(segment)
    half = np.minimum.reduce([np.full(x.size, window // 2), idx - seg_start, seg_end - idx])
    lo, hi = idx - half, idx + half
    cx = np.r_[0.0, np.cumsum(x)]
    cy = np.r_[0.0, np.cumsum(y)]
    count = hi - lo + 1
    return (cx[hi + 1] - cx[lo]) / count, (cy[hi + 1] - cy[lo]) / count


def windowed_motion(
    xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, segment: np.ndarray, half: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heading, displacement and speed from point t-half to t+half, inside one segment.

    Near a segment edge the window shrinks to the points that exist on that
    side. ``half=0`` means plain point-to-previous-point differences.

    Returns:
        (heading_rad, displacement_m, speed_mps) arrays.
    """
    idx = np.arange(len(xs))
    seg_start, seg_end = _segment_bounds(segment)
    if half == 0:
        lo, hi = np.maximum(idx - 1, seg_start), idx
    else:
        lo, hi = np.maximum(idx - half, seg_start), np.minimum(idx + half, seg_end)
    dx, dy = xs[hi] - xs[lo], ys[hi] - ys[lo]
    dt = ts[hi] - ts[lo]
    disp = np.hypot(dx, dy)
    speed = np.where(dt > 0, disp / np.where(dt > 0, dt, 1.0), 0.0)
    return np.arctan2(dy, dx), disp, speed


def preprocess_track(
    track_df: pd.DataFrame,
    large_gap_threshold_s: float = LARGE_GAP_S,
    crs_to: Optional[str] = None,
) -> pd.DataFrame:
    """Project a GPS track and add kinematics and data-quality flags.

    Row order and ``obs_idx`` are never changed: they are the join key for
    every later stage. Problem points are flagged, not dropped. Raw positions
    (``x_m``, ``y_m``) stay untouched; only speed and heading use the
    smoothed track.

    Args:
        track_df: Raw track DataFrame with ['gps_lon', 'gps_lat', 'timestamp'].
        large_gap_threshold_s: Gaps longer than this many seconds are flagged
            and split the track into segments for smoothing.
        crs_to: Metric CRS; ``None`` picks the trip's UTM zone.

    Returns:
        Copy of the track with these added columns:
        x_m, y_m (raw projected position), delta_t, is_large_gap,
        is_time_reversed, segment_id, step_dist_m, speed_raw_mps,
        is_speed_outlier, x_smooth_m, y_smooth_m, speed_mps (smoothed),
        is_stationary, heading_rad (smoothed), heading_valid.
    """
    df = project_coordinates(track_df, crs_to=crs_to)
    n = len(df)

    # Time step to the previous point. The first point has none, so it counts as one 1 Hz step.
    df["delta_t"] = df["timestamp"].diff().fillna(1.0)
    df["is_large_gap"] = df["delta_t"] > large_gap_threshold_s
    df["is_time_reversed"] = df["delta_t"] <= 0
    segment = np.cumsum(df["is_large_gap"].to_numpy())
    df["segment_id"] = segment

    # Raw point-to-point movement: used to spot GPS jumps.
    dx = df["x_m"].diff().fillna(0.0).to_numpy()
    dy = df["y_m"].diff().fillna(0.0).to_numpy()
    step_dist_m = np.hypot(dx, dy)
    df["step_dist_m"] = step_dist_m
    safe_dt = np.where(df["delta_t"].to_numpy() > 0, df["delta_t"].to_numpy(), 1.0)
    speed_raw = step_dist_m / safe_dt
    if n > 1:
        speed_raw[0] = speed_raw[1]  # no predecessor: borrow the next step's speed
    df["speed_raw_mps"] = speed_raw
    df["is_speed_outlier"] = speed_raw > MAX_PLAUSIBLE_SPEED_MPS

    # Smoothed movement: used for speed and heading.
    xs, ys = smooth_xy(df["x_m"].to_numpy(), df["y_m"].to_numpy(), segment, SMOOTH_WINDOW)
    df["x_smooth_m"] = xs
    df["y_smooth_m"] = ys
    heading, disp, speed = windowed_motion(xs, ys, df["timestamp"].to_numpy(), segment, HEADING_HALF_WINDOW)
    df["speed_mps"] = speed
    df["is_stationary"] = speed < STATIONARY_SPEED_MPS

    # Heading only where the vehicle really moved; otherwise keep the last real heading.
    valid = disp >= HEADING_MIN_DISP_M
    df["heading_valid"] = valid
    df["heading_rad"] = pd.Series(np.where(valid, heading, np.nan), index=df.index).ffill().bfill().fillna(0.0).to_numpy()
    return df
