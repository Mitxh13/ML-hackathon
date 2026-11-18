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
