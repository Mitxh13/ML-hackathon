"""Data parsers and loaders for PathSnap raw trip files (.track, .nodes, .arcs, .route)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union
import pandas as pd


def load_track(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .track file containing GPS observations.

    Format: 3 whitespace-separated columns (lon, lat, timestamp).
    """
    path = Path(file_path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["gps_lon", "gps_lat", "timestamp"],
        dtype={"gps_lon": float, "gps_lat": float, "timestamp": float},
    )
    df["obs_idx"] = df.index
    return df


def load_nodes(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .nodes file containing road network vertices.

    Format: 2 whitespace-separated columns (lon, lat).
    0-based row index is node_id.
    """
    path = Path(file_path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": float, "lat": float},
    )
    df["node_id"] = df.index
    return df


def load_arcs(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load an .arcs file containing directed road network edges.

    Format: 2 whitespace-separated integer columns (from_node_id, to_node_id).
    0-based row index is arc_id.
    """
    path = Path(file_path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype={"from_node_id": int, "to_node_id": int},
    )
    df["arc_id"] = df.index
    return df


def load_route(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a .route file containing ground-truth traversed road segments.

    Format: 1 whitespace-separated integer column (arc_id).
    """
    path = Path(file_path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype={"arc_id": int},
    )
    return df


def load_trip(trip_dir: Union[str, Path]) -> dict[str, pd.DataFrame]:
    """Load all 4 files (.track, .nodes, .arcs, .route) for a single trip directory.

    Args:
        trip_dir: Directory containing <trip_id>.track, .nodes, .arcs, .route.

    Returns:
        Dictionary with keys 'track', 'nodes', 'arcs', 'route' (if route exists).
    """
    dir_path = Path(trip_dir)
    trip_id = dir_path.name

    data = {}
    track_file = dir_path / f"{trip_id}.track"
    nodes_file = dir_path / f"{trip_id}.nodes"
    arcs_file = dir_path / f"{trip_id}.arcs"
    route_file = dir_path / f"{trip_id}.route"

    if track_file.exists():
        data["track"] = load_track(track_file)
    if nodes_file.exists():
        data["nodes"] = load_nodes(nodes_file)
    if arcs_file.exists():
        data["arcs"] = load_arcs(arcs_file)
    if route_file.exists():
        data["route"] = load_route(route_file)

    return data


TRIP_FILE_TYPES = ("track", "nodes", "arcs", "route")


def list_trip_dirs(partition_dir: Union[str, Path]) -> list[Path]:
    """Return the trip folders of a partition, sorted by trip ID.

    A trip folder is any sub-directory holding a ``<trip_id>.track`` file.
    """
    root = Path(partition_dir)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


def validate_trip(trip: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Count data-quality problems in a loaded trip. Every count is 0 for a clean trip.

    Checks: missing files, NaN values, coordinates outside valid ranges or at
    (0, 0), arcs pointing at nodes that do not exist, route arcs that do not
    exist, self-loop and duplicate arcs, and repeated or backwards timestamps.
    """
    issues: Dict[str, int] = {f"missing_{name}_file": int(name not in trip) for name in TRIP_FILE_TYPES}
    track = trip.get("track")
    nodes = trip.get("nodes")
    arcs = trip.get("arcs")
    route = trip.get("route")

    issues["nan_values"] = int(sum(int(df.isna().sum().sum()) for df in trip.values()))

    def _bad_coords(lon: pd.Series, lat: pd.Series) -> int:
        out_of_range = (lon.abs() > 180) | (lat.abs() > 90)
        null_island = (lon == 0) & (lat == 0)
        return int((out_of_range | null_island).sum())

    issues["invalid_gps_coords"] = _bad_coords(track["gps_lon"], track["gps_lat"]) if track is not None else 0
    issues["invalid_node_coords"] = _bad_coords(nodes["lon"], nodes["lat"]) if nodes is not None else 0

    if track is not None:
        dt = track["timestamp"].diff().iloc[1:]
        issues["repeated_timestamps"] = int((dt == 0).sum())
        issues["backwards_timestamps"] = int((dt < 0).sum())
    else:
        issues["repeated_timestamps"] = issues["backwards_timestamps"] = 0

    if arcs is not None and nodes is not None:
        n_nodes = len(nodes)
        ends = arcs[["from_node_id", "to_node_id"]]
        issues["arcs_with_unknown_node"] = int(((ends < 0) | (ends >= n_nodes)).any(axis=1).sum())
        issues["self_loop_arcs"] = int((arcs["from_node_id"] == arcs["to_node_id"]).sum())
        issues["duplicate_arcs"] = int(ends.duplicated().sum())
    else:
        issues["arcs_with_unknown_node"] = issues["self_loop_arcs"] = issues["duplicate_arcs"] = 0

    if route is not None and arcs is not None:
        ids = route["arc_id"]
        issues["route_arcs_unknown"] = int(((ids < 0) | (ids >= len(arcs))).sum())
    else:
        issues["route_arcs_unknown"] = 0
    return issues
