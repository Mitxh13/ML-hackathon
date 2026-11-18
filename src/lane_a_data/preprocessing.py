"""GPS trajectory cleaning, coordinate projection, speed calculation, and gap detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate Haversine geodesic distance in meters between two lon/lat coordinates."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def haversine_distance_np(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Vectorized Haversine geodesic distance in meters."""
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def project_coordinates(
    df: pd.DataFrame,
    lon_col: str = "gps_lon",
    lat_col: str = "gps_lat",
    crs_to: str = "EPSG:3857",
) -> pd.DataFrame:
    """Project longitude and latitude coordinates to a metric CRS (default Web Mercator / UTM).

    Args:
        df: Input DataFrame containing angular coordinates.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.
        crs_to: Target metric projection EPSG code.

    Returns:
        DataFrame with additional columns ['x_m', 'y_m'].
    """
    out_df = df.copy()
    if out_df.empty:
        out_df["x_m"] = []
        out_df["y_m"] = []
        return out_df

    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    x_coords, y_coords = transformer.transform(out_df[lon_col].values, out_df[lat_col].values)
    out_df["x_m"] = x_coords
    out_df["y_m"] = y_coords
    return out_df


def preprocess_track(
    track_df: pd.DataFrame,
    large_gap_threshold_s: float = 5.0,
) -> pd.DataFrame:
    """Clean GPS track observations, computing delta_t, speeds, headings, and gap flags.

    Args:
        track_df: Raw track DataFrame with ['gps_lon', 'gps_lat', 'timestamp'].
        large_gap_threshold_s: Threshold in seconds to flag large time gaps.

    Returns:
        Cleaned track DataFrame with projected coordinates and kinematics.
    """
    df = project_coordinates(track_df)

    # Time delta
    df["delta_t"] = df["timestamp"].diff().fillna(1.0)
    df["is_large_gap"] = df["delta_t"] > large_gap_threshold_s

    # Spatial delta
    dx = df["x_m"].diff().fillna(0.0)
    dy = df["y_m"].diff().fillna(0.0)
    step_dist_m = np.sqrt(dx**2 + dy**2)
    df["step_dist_m"] = step_dist_m

    # Speed in m/s (avoid zero division)
    safe_dt = np.where(df["delta_t"] > 0, df["delta_t"], 1.0)
    df["speed_mps"] = step_dist_m / safe_dt

    # Heading in radians (-pi to pi)
    df["heading_rad"] = np.arctan2(dy, dx).fillna(0.0)

    return df
