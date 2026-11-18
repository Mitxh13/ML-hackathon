"""Directed road network graph with metric arc geometry and bounded shortest paths.

Section 3.1. One ``RoadNetworkGraph`` is built per trip from its ``.nodes`` / ``.arcs`` tables:

* node coordinates are projected once into the trip's UTM zone (metres), so every length
  and every point-to-arc projection is in metres (never degrees);
* arcs are directed: arc ``a`` goes from ``from_node_id`` to ``to_node_id`` and the reverse
  direction of a two-way street is a separate arc;
* adjacency is stored in CSR form (for each node, the arcs leaving it), which makes a
  Dijkstra search a few array look-ups per step;
* shortest-path searches are *bounded*: they stop once every node within ``limit_m`` metres
  has been settled. Map matching only ever needs short paths (the distance a car can cover
  between two GPS fixes), so a bounded search touches a few hundred nodes instead of the
  whole network. Results are cached per source node because consecutive GPS points share
  most of their candidate arcs.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.lane_a_data.preprocessing import lonlat_to_xy, utm_crs_for

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _normalise_tables(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = nodes_df
    arcs = arcs_df
    if "lon" not in nodes.columns or "lat" not in nodes.columns:
        nodes = nodes.iloc[:, :2].set_axis(["lon", "lat"], axis=1)
    if "from_node_id" not in arcs.columns or "to_node_id" not in arcs.columns:
        arcs = arcs.iloc[:, :2].set_axis(["from_node_id", "to_node_id"], axis=1)
    return nodes, arcs


class RoadNetworkGraph:
    """Directed road graph for one trip. Row index of ``.nodes`` / ``.arcs`` = node_id / arc_id.

    Attributes:
        crs: EPSG code of the metric projection used for all geometry.
        node_xy: (N, 2) node coordinates in metres.
        arc_from, arc_to: (A,) node ids of each arc.
        arc_length_m: (A,) arc length in metres.
        arc_heading_rad: (A,) direction of travel along each arc (east = 0, north = pi/2).
    """

    def __init__(self, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame, crs: Optional[str] = None) -> None:
        nodes, arcs = _normalise_tables(nodes_df, arcs_df)
        self.nodes_df = nodes
        self.arcs_df = arcs
        lon = nodes["lon"].to_numpy(dtype=float)
        lat = nodes["lat"].to_numpy(dtype=float)
        if len(lon) == 0:
            raise ValueError("Road network has no nodes")
        self.crs = crs or utm_crs_for(float(np.median(lon)), float(np.median(lat)))
        x, y = lonlat_to_xy(lon, lat, self.crs)
        self.node_xy = np.column_stack([x, y])
        self.node_lonlat = np.column_stack([lon, lat])

        self.arc_from = arcs["from_node_id"].to_numpy(dtype=np.int64)
        self.arc_to = arcs["to_node_id"].to_numpy(dtype=np.int64)
        n_nodes = len(lon)
        if len(self.arc_from) and (min(self.arc_from.min(), self.arc_to.min()) < 0
                                   or max(self.arc_from.max(), self.arc_to.max()) >= n_nodes):
            raise ValueError("arcs reference node ids outside the nodes table")
        vec = self.node_xy[self.arc_to] - self.node_xy[self.arc_from]
        self.arc_length_m = np.hypot(vec[:, 0], vec[:, 1])
        self.arc_heading_rad = np.arctan2(vec[:, 1], vec[:, 0])
        self.n_nodes = n_nodes
        self.n_arcs = len(self.arc_from)

        # CSR adjacency: out_arcs[out_ptr[u]:out_ptr[u+1]] are the arcs leaving node u.
        order = np.argsort(self.arc_from, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(self._out_ptr, self.arc_from + 1, 1)
        self._out_ptr = np.cumsum(self._out_ptr)
        self._out_to = self.arc_to[order].tolist()
        self._out_len = self.arc_length_m[order].tolist()
        self._out_arc_list = order.tolist()
        self._ptr_list = self._out_ptr.tolist()

        # source node -> (limit searched, {node: distance}, {node: arc used to reach it})
        self._cache: Dict[int, Tuple[float, Dict[int, float], Dict[int, int]]] = {}

    # ------------------------------------------------------------------ basic look-ups
    def get_arc_nodes(self, arc_id: int) -> Tuple[int, int]:
        """(from_node_id, to_node_id) of an arc."""
        return int(self.arc_from[arc_id]), int(self.arc_to[arc_id])

    def project(self, lon, lat) -> np.ndarray:
        """Project lon/lat (degrees) into this graph's metric CRS; returns (n, 2) metres."""
        x, y = lonlat_to_xy(np.atleast_1d(lon), np.atleast_1d(lat), self.crs)
        return np.column_stack([x, y])

    def project_onto_arcs(self, points_xy: np.ndarray, arc_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Row-wise projection of points onto arcs.

        Returns (distance to the arc in metres, position along the arc in metres from its
        start, clipped to [0, arc length]).
        """
        a = self.node_xy[self.arc_from[arc_ids]]
        b = self.node_xy[self.arc_to[arc_ids]]
        ab = b - a
        l2 = np.einsum("ij,ij->i", ab, ab)
        t = np.einsum("ij,ij->i", points_xy - a, ab) / np.where(l2 > 0, l2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t[:, None] * ab
        dist = np.hypot(*(points_xy - closest).T)
        return dist, t * self.arc_length_m[arc_ids]

    # ------------------------------------------------------------------ shortest paths
    def _search(self, source: int, limit_m: float) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Bounded Dijkstra from ``source``: all nodes within ``limit_m`` metres (cached)."""
        cached = self._cache.get(source)
        if cached is not None and cached[0] >= limit_m:
            return cached[1], cached[2]
        dist: Dict[int, float] = {source: 0.0}
        via: Dict[int, int] = {}
        done = set()
        heap = [(0.0, source)]
        ptr, out_to, out_len, out_arc = self._ptr_list, self._out_to, self._out_len, self._out_arc_list
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                nd = d + out_len[k]
                if nd > limit_m:
                    continue
                v = out_to[k]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    via[v] = out_arc[k]
                    heapq.heappush(heap, (nd, v))
        self._cache[source] = (limit_m, dist, via)
        return dist, via

    def node_distances(self, source: int, limit_m: float) -> Dict[int, float]:
        """Exact network distances (metres) from ``source``; contains every node within
        ``limit_m``. A cached result from a wider search may also contain farther nodes, so
        callers compare the returned distance with their own limit."""
        return self._search(int(source), float(limit_m))[0]

    def shortest_path_distance(self, from_node: int, to_node: int, limit_m: float = 5_000.0) -> float:
        """Shortest network distance in metres, or ``inf`` if not reachable within ``limit_m``."""
        if from_node == to_node:
            return 0.0
        d = float(self.node_distances(from_node, limit_m).get(int(to_node), np.inf))
        return d if d <= limit_m else float("inf")

    def shortest_path_arcs(self, from_node: int, to_node: int, limit_m: float = 5_000.0) -> Optional[List[int]]:
        """Arc ids of the shortest path between two nodes, ``[]`` if equal, ``None`` if unreachable."""
        if from_node == to_node:
            return []
        dist, via = self._search(int(from_node), float(limit_m))
        if dist.get(int(to_node), np.inf) > limit_m:
            return None
        path, node = [], int(to_node)
        while node != from_node:
            arc = via[node]
            path.append(arc)
            node = int(self.arc_from[arc])
        return path[::-1]

    def has_path(self, from_node: int, to_node: int, max_hops: Optional[int] = None,
                 limit_m: float = 5_000.0) -> bool:
        """True if ``to_node`` can be reached from ``from_node`` (within ``limit_m`` metres and,
        if given, ``max_hops`` arcs)."""
        path = self.shortest_path_arcs(from_node, to_node, limit_m)
        if path is None:
            return False
        return max_hops is None or len(path) <= max_hops

    def arc_transition_distance(self, prev_arc_id: int, curr_arc_id: int, limit_m: float = 5_000.0) -> float:
        """Network distance from the end of ``prev_arc`` to the start of ``curr_arc`` (0 if the
        same arc or directly connected, ``inf`` if unreachable within ``limit_m``)."""
        if prev_arc_id == curr_arc_id:
            return 0.0
        return self.shortest_path_distance(int(self.arc_to[prev_arc_id]), int(self.arc_from[curr_arc_id]), limit_m)

    def connect_route(self, arcs: Sequence[int], limit_m: float = 2_000.0) -> List[int]:
        """Insert the shortest connecting arcs between consecutive route arcs where needed.

        Used for display only: consecutive decoded arcs that are not adjacent are joined by
        the shortest path between them when one exists within ``limit_m``.
        """
        out: List[int] = []
        for arc in arcs:
            arc = int(arc)
            if out and out[-1] != arc and self.arc_to[out[-1]] != self.arc_from[arc]:
                bridge = self.shortest_path_arcs(int(self.arc_to[out[-1]]), int(self.arc_from[arc]), limit_m)
                if bridge:
                    out.extend(bridge)
            if not out or out[-1] != arc:
                out.append(arc)
        return out

    def arcs_latlon(self, arc_ids: Iterable[int]) -> List[List[List[float]]]:
        """Arcs as [[lat, lon], [lat, lon]] pairs (Folium / Leaflet order)."""
        ids = np.asarray(list(arc_ids), dtype=np.int64)
        if len(ids) == 0:
            return []
        ll = self.node_lonlat[:, ::-1]
        return np.stack([ll[self.arc_from[ids]], ll[self.arc_to[ids]]], axis=1).tolist()
