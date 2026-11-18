"""Tests for part 1.1: loaders, projection, track preprocessing and EDA statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lane_a_data import preprocessing as P
from src.lane_a_data.eda_stats import load_projected_trip, network_stats, route_stats
from src.lane_a_data.loaders import list_trip_dirs, load_trip, validate_trip
from src.lane_a_data.run_eda import DEV_PARTITIONS


def write_trip(root: Path, trip_id: str, track, nodes, arcs, route) -> Path:
    """Write a synthetic trip in the raw whitespace-separated format."""
    d = root / trip_id
    d.mkdir(parents=True)
    np.savetxt(d / f"{trip_id}.track", np.asarray(track, dtype=float), fmt="%.7f")
    np.savetxt(d / f"{trip_id}.nodes", np.asarray(nodes, dtype=float), fmt="%.7f")
    np.savetxt(d / f"{trip_id}.arcs", np.asarray(arcs, dtype=int), fmt="%d")
    np.savetxt(d / f"{trip_id}.route", np.asarray(route, dtype=int), fmt="%d")
    return d


def metres_to_lonlat(x_m, y_m, lon0=8.6, lat0=50.1):
    """Small offsets in metres around (lon0, lat0) to degrees (local approximation, fine for tests)."""
    lat = lat0 + np.asarray(y_m) / 111_320.0
    lon = lon0 + np.asarray(x_m) / (111_320.0 * np.cos(np.radians(lat0)))
    return lon, lat


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lon, lat, expected", [
    (8.6, 50.1, "EPSG:32632"),      # Frankfurt
    (-58.4, -34.6, "EPSG:32721"),   # Buenos Aires, southern hemisphere
    (137.2, 35.0, "EPSG:32653"),
    (-100.4, 20.6, "EPSG:32614"),
    (180.0, 10.0, "EPSG:32601"),    # the antimeridian wraps to zone 1
])
def test_utm_zone_lookup(lon, lat, expected):
    assert P.utm_crs_for(lon, lat) == expected


def test_numpy_utm_matches_pyproj():
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(0)
    for crs, lon_c, lat_c in [("EPSG:32632", 9.0, 50.0), ("EPSG:32721", -57.0, -34.6), ("EPSG:32633", 15.0, 61.7)]:
        lon = lon_c + rng.uniform(-2.9, 2.9, 200)
        lat = lat_c + rng.uniform(-1.0, 1.0, 200)
        x_np, y_np = P._utm_forward_numpy(lon, lat, crs)
        x_pp, y_pp = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        assert np.max(np.hypot(x_np - x_pp, y_np - y_pp)) < 1e-3  # under 1 mm


@pytest.mark.parametrize("lat", [-34.6, 0.5, 50.1, 61.7])
def test_projected_distance_matches_haversine(lat):
    """A 100 m step must measure about 100 m at any latitude (Web Mercator would not)."""
    lon = np.array([10.0, 10.0 + 100.0 / (111_320.0 * np.cos(np.radians(lat)))])
    lats = np.array([lat, lat])
    crs = P.utm_crs_for(lon[0], lat)
    x, y = P.lonlat_to_xy(lon, lats, crs)
    projected = np.hypot(np.diff(x), np.diff(y))[0]
    true = P.haversine_m(lon[0], lat, lon[1], lat)
    assert abs(projected - true) / true < 0.006


def test_project_coordinates_records_crs_and_keeps_rows():
    lon, lat = metres_to_lonlat([0, 100, 200], [0, 0, 0])
    df = pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "obs_idx": [0, 1, 2]})
    out = P.project_coordinates(df)
    assert out.attrs["crs"] == "EPSG:32632"
    assert list(out["obs_idx"]) == [0, 1, 2]
    assert np.allclose(np.diff(out["x_m"]), 100.0, rtol=0.01)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_point_segment_distance_cases():
    # perpendicular foot inside the segment
    d, t = P.point_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(3.0) and t == pytest.approx(0.5)
    # beyond the end: distance to the end point
    d, t = P.point_segment_distance(13.0, 4.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(5.0) and t == pytest.approx(1.0)
    # zero-length segment behaves like a point
    d, t = P.point_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(5.0) and t == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Track preprocessing
# ---------------------------------------------------------------------------

def jittery_eastward_track(n=60, speed=10.0, jitter=4.0):
    """Vehicle driving east at ``speed`` m/s while GPS zigzags ``jitter`` m left and right."""
    t = np.arange(n, dtype=float)
    x = speed * t
    y = np.where(np.arange(n) % 2 == 0, jitter, -jitter)
    lon, lat = metres_to_lonlat(x, y)
    return pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "timestamp": t + 1, "obs_idx": np.arange(n)})


def test_smoothed_heading_and_speed_ignore_jitter():
    out = P.preprocess_track(jittery_eastward_track())
    middle = out.iloc[5:-5]
    raw_heading = np.arctan2(np.diff(out["y_m"]), np.diff(out["x_m"]))
    assert np.max(np.abs(raw_heading)) > np.radians(30)  # raw steps zigzag
    assert np.max(np.abs(middle["heading_rad"])) < np.radians(5)  # smoothed heading points east
    assert middle["speed_mps"].between(9.0, 11.0).all()
    assert middle["heading_valid"].all()


def test_rows_never_dropped_or_reordered():
    raw = jittery_eastward_track()
    out = P.preprocess_track(raw)
    assert len(out) == len(raw)
    assert list(out["obs_idx"]) == list(raw["obs_idx"])
    assert np.allclose(out["gps_lon"], raw["gps_lon"])


def test_gap_splits_segments_and_smoothing_does_not_cross_it():
    # 20 s eastward, a 60 s gap, then 20 s northward
    t1 = np.arange(20.0)
    t2 = np.arange(20.0) + 80.0
    x = np.r_[10.0 * t1, np.full(20, 190.0)]
    y = np.r_[np.zeros(20), 10.0 * (t2 - 80.0) + 500.0]
    lon, lat = metres_to_lonlat(x, y)
    track = pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "timestamp": np.r_[t1, t2], "obs_idx": np.arange(40)})
    out = P.preprocess_track(track)
    assert out["is_large_gap"].sum() == 1 and bool(out.loc[20, "is_large_gap"])
    assert list(out["segment_id"].unique()) == [0, 1]
    # the first point after the gap must head north, not towards the pre-gap points
    assert abs(out.loc[20, "heading_rad"] - np.pi / 2) < np.radians(5)
    assert abs(out.loc[19, "heading_rad"]) < np.radians(5)


def test_stationary_points_keep_previous_heading():
    x = np.r_[10.0 * np.arange(15), np.full(15, 140.0)]  # drive north-east, then park
    y = x.copy()
    lon, lat = metres_to_lonlat(x, y)
    track = pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "timestamp": np.arange(30.0), "obs_idx": np.arange(30)})
    out = P.preprocess_track(track)
    parked = out.iloc[22:]
    assert parked["is_stationary"].all()
    assert not parked["heading_valid"].any()
    assert np.allclose(parked["heading_rad"], np.pi / 4, atol=np.radians(5))


def test_speed_jump_is_flagged_not_removed():
    x = 10.0 * np.arange(10)
    x[5] += 500.0  # one fix teleports 500 m
    lon, lat = metres_to_lonlat(x, np.zeros(10))
    track = pd.DataFrame({"gps_lon": lon, "gps_lat": lat, "timestamp": np.arange(10.0), "obs_idx": np.arange(10)})
    out = P.preprocess_track(track)
    assert len(out) == 10
    assert out["is_speed_outlier"].sum() == 2  # jumping out and back in


# ---------------------------------------------------------------------------
# Loaders and validation
# ---------------------------------------------------------------------------

def square_trip(tmp_path: Path, **overrides) -> Path:
    """A 100 m square road network with two-way arcs, driven once around."""
    nodes_xy = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    nlon, nlat = metres_to_lonlat(nodes_xy[:, 0], nodes_xy[:, 1])
    arcs = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 0], [2, 1], [3, 2], [0, 3]]
    steps = np.linspace(0, 1, 11)[:-1]
    px = np.r_[100 * steps, np.full(10, 100.0), 100 - 100 * steps, np.zeros(10)]
    py = np.r_[np.zeros(10), 100 * steps, np.full(10, 100.0), 100 - 100 * steps]
    tlon, tlat = metres_to_lonlat(px, py + 3.0)
    track = np.column_stack([tlon, tlat, np.arange(1, 41)])
    parts = {"track": track, "nodes": np.column_stack([nlon, nlat]), "arcs": arcs, "route": [0, 1, 2, 3]}
    parts.update(overrides)
    return write_trip(tmp_path, "00000001", **parts)


def test_list_trip_dirs_only_returns_trip_folders(tmp_path):
    square_trip(tmp_path)
    (tmp_path / "notes").mkdir()
    (tmp_path / "readme.txt").write_text("x")
    assert [d.name for d in list_trip_dirs(tmp_path)] == ["00000001"]
    assert list_trip_dirs(tmp_path / "missing") == []


def test_validate_trip_clean(tmp_path):
    issues = validate_trip(load_trip(square_trip(tmp_path)))
    assert all(v == 0 for v in issues.values()), issues


def test_validate_trip_counts_problems(tmp_path):
    d = square_trip(tmp_path, arcs=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 1], [2, 9], [3, 3]], route=[0, 1, 2, 3, 42])
    trip = load_trip(d)
    trip["track"].loc[5, "timestamp"] = trip["track"].loc[4, "timestamp"]  # repeated timestamp
    issues = validate_trip(trip)
    assert issues["duplicate_arcs"] == 1
    assert issues["arcs_with_unknown_node"] == 1
    assert issues["self_loop_arcs"] == 1
    assert issues["route_arcs_unknown"] == 1
    assert issues["repeated_timestamps"] == 1


# ---------------------------------------------------------------------------
# EDA statistics
# ---------------------------------------------------------------------------

def test_network_and_route_stats_on_square(tmp_path):
    trip = load_projected_trip(square_trip(tmp_path), "train")
    net = network_stats(trip)
    assert net["two_way_arc_share"] == 1.0
    assert net["weak_components"] == 1 and net["strong_components"] == 1
    assert net["arc_length_median_m"] == pytest.approx(100.0, rel=0.01)
    route = route_stats(trip)
    assert route["route_connected_share"] == 1.0 and route["route_breaks"] == 0
    assert route["route_length_km"] == pytest.approx(0.4, rel=0.01)


def test_route_break_is_counted(tmp_path):
    trip = load_projected_trip(square_trip(tmp_path, route=[0, 2, 3]), "train")  # skips arc 1
    assert route_stats(trip)["route_breaks"] == 1


def test_eda_never_reads_evaluator_partition():
    assert "evaluator_test" not in DEV_PARTITIONS
