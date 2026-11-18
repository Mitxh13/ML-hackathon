"""Interactive Folium maps of GPS observations, road network and routes (Section 3.3).

Folium / Leaflet expect [latitude, longitude]; ``.track`` and ``.nodes`` store
[longitude, latitude]. Every coordinate below is flipped explicitly with ``[lat, lon]``.

Layers (each can be switched on/off in the layer control, and the legend names them):
    Road network (nearby)   grey   arcs within ``network_radius_m`` of the GPS track
    Reference route         green  ``.route`` arcs (report maps only, never sent to the UI)
    Classifier-only route   orange arg-max of the classifier per point
    Predicted route (HMM)   red    decoded route
    GPS observations        blue   raw fixes, with start / end markers
    Wrong points            black  fixes whose decoded arc differs from the reference
"""

from __future__ import annotations

from typing import Optional, Sequence

import folium
import numpy as np
import pandas as pd
from branca.element import MacroElement, Template
from scipy.spatial import cKDTree

from src.lane_a_data.preprocessing import lonlat_to_xy, utm_crs_for

COLOURS = {
    "network": "#9CA3AF",
    "reference": "#16A34A",
    "classifier": "#F59E0B",
    "predicted": "#DC2626",
    "gps": "#2563EB",
    "wrong": "#111827",
}
MAX_GPS_MARKERS = 3000
MAX_NETWORK_ARCS = 6000


def _tables(track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame):
    t = track_df
    if "gps_lon" not in t.columns:
        t = t.iloc[:, :3].set_axis(["gps_lon", "gps_lat", "timestamp"], axis=1)
    n = nodes_df if "lon" in nodes_df.columns else nodes_df.iloc[:, :2].set_axis(["lon", "lat"], axis=1)
    a = arcs_df if "from_node_id" in arcs_df.columns else arcs_df.iloc[:, :2].set_axis(["from_node_id", "to_node_id"], axis=1)
    return t, n, a


def _segments_latlon(arc_ids: Sequence[int], nodes: pd.DataFrame, arcs: pd.DataFrame) -> list:
    ids = np.asarray([a for a in arc_ids if 0 <= int(a) < len(arcs)], dtype=np.int64)
    if len(ids) == 0:
        return []
    frm = arcs["from_node_id"].to_numpy(np.int64)[ids]
    to = arcs["to_node_id"].to_numpy(np.int64)[ids]
    latlon = nodes[["lat", "lon"]].to_numpy(float)
    return np.round(np.stack([latlon[frm], latlon[to]], axis=1), 6).tolist()


def _route_lines(arc_ids: Sequence[int], nodes: pd.DataFrame, arcs: pd.DataFrame) -> list:
    """Join consecutive connected arcs into polylines (fewer, smoother map objects)."""
    segs = _segments_latlon(arc_ids, nodes, arcs)
    lines: list = []
    for a, b in segs:
        if lines and lines[-1][-1] == a:
            lines[-1].append(b)
        else:
            lines.append([a, b])
    return lines


def nearby_arcs(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                radius_m: float = 150.0) -> np.ndarray:
    """Arc ids passing within roughly ``radius_m`` of any GPS fix (metric test)."""
    lon, lat = track["gps_lon"].to_numpy(float), track["gps_lat"].to_numpy(float)
    crs = utm_crs_for(float(np.median(lon)), float(np.median(lat)))
    gx, gy = lonlat_to_xy(lon, lat, crs)
    nx_, ny_ = lonlat_to_xy(nodes["lon"].to_numpy(float), nodes["lat"].to_numpy(float), crs)
    frm = arcs["from_node_id"].to_numpy(np.int64)
    to = arcs["to_node_id"].to_numpy(np.int64)
    mid = np.column_stack([(nx_[frm] + nx_[to]) / 2, (ny_[frm] + ny_[to]) / 2])
    half = np.hypot(nx_[to] - nx_[frm], ny_[to] - ny_[frm]) / 2
    d, _ = cKDTree(np.column_stack([gx, gy])).query(mid, k=1)
    ids = np.flatnonzero(d <= radius_m + half)
    if len(ids) > MAX_NETWORK_ARCS:
        ids = ids[np.argsort(d[ids])[:MAX_NETWORK_ARCS]]
    return ids


def _legend(entries: list[tuple[str, str]], title: str) -> MacroElement:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:18px;height:4px;'
        f'background:{c};margin-right:6px;vertical-align:middle"></span>{name}</div>'
        for name, c in entries
    )
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
                padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;
                font:12px/1.4 Arial, sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.2)">
      <div style="font-weight:bold;margin-bottom:4px">{title}</div>{rows}
    </div>
    {{% endmacro %}}"""
    legend = MacroElement()
    legend._template = Template(html)
    return legend


def create_route_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_arcs: Optional[Sequence[int]] = None,
    reference_route: Optional[Sequence[int]] = None,
    classifier_arcs: Optional[Sequence[int]] = None,
    wrong_mask: Optional[Sequence[bool]] = None,
    title: str = "PathSnap route reconstruction",
    network_radius_m: float = 150.0,
    show_network: bool = True,
) -> folium.Map:
    """Interactive map comparing the GPS trajectory with predicted (and reference) routes.

    ``reference_route`` must be left ``None`` for maps sent to the evaluator UI so that
    ground truth stays on Laptop 1.
    """
    track, nodes, arcs = _tables(track_df, nodes_df, arcs_df)
    latlon = track[["gps_lat", "gps_lon"]].to_numpy(float)
    center = latlon.mean(axis=0).tolist() if len(latlon) else [0.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    legend = []

    if show_network and len(track):
        ids = nearby_arcs(track, nodes, arcs, network_radius_m)
        group = folium.FeatureGroup(name="Road network (nearby)", show=True)
        for seg in _segments_latlon(ids, nodes, arcs):
            folium.PolyLine(seg, color=COLOURS["network"], weight=1.5, opacity=0.6).add_to(group)
        group.add_to(fmap)
        legend.append(("Road network (nearby)", COLOURS["network"]))

    def add_route(arc_ids, name, colour, weight, dash=None):
        if arc_ids is None or len(arc_ids) == 0:
            return
        group = folium.FeatureGroup(name=name)
        for line in _route_lines(arc_ids, nodes, arcs):
            folium.PolyLine(line, color=colour, weight=weight, opacity=0.85, dash_array=dash,
                            tooltip=name).add_to(group)
        group.add_to(fmap)
        legend.append((name, colour))

    add_route(reference_route, "Reference route (.route)", COLOURS["reference"], 8)
    add_route(classifier_arcs, "Classifier-only route", COLOURS["classifier"], 4, "6, 6")
    add_route(predicted_arcs, "Predicted route (HMM)", COLOURS["predicted"], 4)

    if len(latlon):
        gps = folium.FeatureGroup(name="GPS observations")
        folium.PolyLine(latlon.tolist(), color=COLOURS["gps"], weight=1, opacity=0.5).add_to(gps)
        step = max(1, int(np.ceil(len(latlon) / MAX_GPS_MARKERS)))
        obs = track["obs_idx"].to_numpy() if "obs_idx" in track.columns else np.arange(len(track))
        ts = track["timestamp"].to_numpy(float)
        for i in range(0, len(latlon), step):
            folium.CircleMarker(latlon[i].tolist(), radius=2.5, color=COLOURS["gps"], fill=True,
                                fill_opacity=0.8, weight=1,
                                tooltip=f"obs {int(obs[i])}, t = {ts[i]:.0f} s").add_to(gps)
        folium.Marker(latlon[0].tolist(), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(gps)
        folium.Marker(latlon[-1].tolist(), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(gps)
        gps.add_to(fmap)
        legend.append(("GPS observations" + (f" (every {step}th marker)" if step > 1 else ""), COLOURS["gps"]))

        if wrong_mask is not None:
            wrong = np.flatnonzero(np.asarray(wrong_mask, dtype=bool))
            if len(wrong):
                group = folium.FeatureGroup(name=f"Wrong points ({len(wrong)})", show=False)
                for i in wrong[:: max(1, int(np.ceil(len(wrong) / MAX_GPS_MARKERS)))]:
                    folium.CircleMarker(latlon[i].tolist(), radius=4, color=COLOURS["wrong"], fill=False,
                                        weight=2, tooltip=f"obs {int(obs[i])}: decoded arc differs from reference").add_to(group)
                group.add_to(fmap)
                legend.append(("Wrong points (toggle in layers)", COLOURS["wrong"]))
        fmap.fit_bounds([latlon.min(axis=0).tolist(), latlon.max(axis=0).tolist()], padding=(20, 20))

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().add_child(_legend(legend, title))
    return fmap


def render_map_html(route_map: folium.Map) -> str:
    """Folium map as a standalone HTML string."""
    return route_map.get_root().render()
