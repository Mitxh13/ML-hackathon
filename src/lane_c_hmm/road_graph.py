"""Road network graph representation and shortest-path calculation for PathSnap."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate Haversine geodesic distance in meters between two lon/lat coordinates."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


class RoadNetworkGraph:
    """Directed road network graph constructed from .nodes and .arcs.

    Attributes:
        nodes_df: DataFrame of nodes (lon, lat) indexed by node_id.
        arcs_df: DataFrame of arcs (from_node_id, to_node_id) indexed by arc_id.
        nx_graph: NetworkX DiGraph with edge weights representing arc length in meters.
    """

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> None:
        self.nodes_df = nodes_df.copy()
        self.arcs_df = arcs_df.copy()

        # Normalize column names if missing
        if "lon" not in self.nodes_df.columns:
            self.nodes_df.columns = ["lon", "lat"][: len(self.nodes_df.columns)]
        if "from_node_id" not in self.arcs_df.columns:
            self.arcs_df.columns = ["from_node_id", "to_node_id"][: len(self.arcs_df.columns)]

        self.nx_graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Add nodes and weighted edges to directed graph."""
        # Add nodes with coordinates
        for node_id, row in self.nodes_df.iterrows():
            self.nx_graph.add_node(int(node_id), lon=float(row["lon"]), lat=float(row["lat"]))

        # Add edges with arc length in meters
        for arc_id, row in self.arcs_df.iterrows():
            u = int(row["from_node_id"])
            v = int(row["to_node_id"])

            if u in self.nx_graph and v in self.nx_graph:
                u_lon, u_lat = self.nx_graph.nodes[u]["lon"], self.nx_graph.nodes[u]["lat"]
                v_lon, v_lat = self.nx_graph.nodes[v]["lon"], self.nx_graph.nodes[v]["lat"]
                length_m = max(1.0, haversine_distance_m(u_lon, u_lat, v_lon, v_lat))
            else:
                length_m = 10.0

            self.nx_graph.add_edge(u, v, arc_id=int(arc_id), weight=length_m)

    def get_arc_nodes(self, arc_id: int) -> Tuple[int, int]:
        """Return (from_node_id, to_node_id) for given arc_id."""
        row = self.arcs_df.iloc[arc_id]
        return int(row["from_node_id"]), int(row["to_node_id"])

    def shortest_path_distance(self, from_node: int, to_node: int) -> float:
        """Return shortest network distance in meters from from_node to to_node."""
        if from_node == to_node:
            return 0.0
        try:
            return float(
                nx.shortest_path_length(
                    self.nx_graph, source=from_node, target=to_node, weight="weight"
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def has_path(self, from_node: int, to_node: int, max_hops: Optional[int] = None) -> bool:
        """Check if a directed path exists between from_node and to_node."""
        if from_node == to_node:
            return True
        try:
            if not nx.has_path(self.nx_graph, from_node, to_node):
                return False
            if max_hops is not None:
                hops = nx.shortest_path_length(self.nx_graph, from_node, to_node)
                return hops <= max_hops
            return True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def arc_transition_distance(self, prev_arc_id: int, curr_arc_id: int) -> float:
        """Calculate transition distance from prev_arc to curr_arc in the road network."""
        if prev_arc_id == curr_arc_id:
            return 0.0

        _, prev_to = self.get_arc_nodes(prev_arc_id)
        curr_from, _ = self.get_arc_nodes(curr_arc_id)

        if prev_to == curr_from:
            return 0.0

        return self.shortest_path_distance(prev_to, curr_from)
