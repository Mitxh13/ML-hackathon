"""Interactive Folium map generation for PathSnap trajectories and routes."""

from __future__ import annotations

from typing import List, Optional, Sequence
import folium
import pandas as pd


def create_route_map(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    predicted_arcs: Optional[Sequence[int]] = None,
    reference_route: Optional[Sequence[int]] = None,
) -> folium.Map:
    """Generate interactive Folium map visualizing GPS trajectory, road network, and routes.

    Note: Folium expects [lat, lon], while .track and .nodes store lon lat.

    Args:
        track_df: DataFrame with ['gps_lon', 'gps_lat', 'timestamp'] or 3 unnamed columns.
        nodes_df: DataFrame with ['lon', 'lat'] indexed by node_id.
        arcs_df: DataFrame with ['from_node_id', 'to_node_id'] indexed by arc_id.
        predicted_arcs: Optional list/array of predicted arc IDs.
        reference_route: Optional list/array of ground truth arc IDs.

    Returns:
        folium.Map object.
    """
    # Normalize track columns
    t_df = track_df.copy()
    if "gps_lon" not in t_df.columns:
        t_df.columns = ["gps_lon", "gps_lat", "timestamp"][: len(t_df.columns)]

    # Compute map center
    mean_lat = float(t_df["gps_lat"].mean()) if not t_df.empty else 0.0
    mean_lon = float(t_df["gps_lon"].mean()) if not t_df.empty else 0.0

    route_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, control_scale=True)

    # 1. Raw GPS Trajectory Layer
    gps_group = folium.FeatureGroup(name="Raw GPS Trajectory")
    gps_coords = []
    for _, row in t_df.iterrows():
        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        gps_coords.append([lat, lon])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2563EB",  # Blue
            fill=True,
            fill_opacity=0.7,
            popup=f"Obs: {row.get('obs_idx', '')} Time: {row.get('timestamp', '')}s",
        ).add_to(gps_group)

    if gps_coords:
        folium.PolyLine(
            gps_coords, color="#3B82F6", weight=2, opacity=0.6, dash_array="4, 4"
        ).add_to(gps_group)
    gps_group.add_to(route_map)

    # Helper to get polyline for a sequence of arcs
    def _arcs_to_polyline(arc_list: Sequence[int]) -> list[list[float]]:
        coords = []
        for arc_id in arc_list:
            if 0 <= arc_id < len(arcs_df):
                u = int(arcs_df.iloc[arc_id, 0])
                v = int(arcs_df.iloc[arc_id, 1])
                if 0 <= u < len(nodes_df) and 0 <= v < len(nodes_df):
                    u_lon, u_lat = float(nodes_df.iloc[u, 0]), float(nodes_df.iloc[u, 1])
                    v_lon, v_lat = float(nodes_df.iloc[v, 0]), float(nodes_df.iloc[v, 1])
                    coords.append([u_lat, u_lon])
                    coords.append([v_lat, v_lon])
        return coords

    # 2. Ground-Truth Reference Route Layer
    if reference_route is not None and len(reference_route) > 0:
        ref_group = folium.FeatureGroup(name="Ground-Truth Route (.route)")
        ref_coords = _arcs_to_polyline(reference_route)
        if ref_coords:
            folium.PolyLine(
                ref_coords, color="#16A34A", weight=5, opacity=0.8, tooltip="Reference Route"
            ).add_to(ref_group)
        ref_group.add_to(route_map)

    # 3. Predicted Decoded Route Layer
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        pred_group = folium.FeatureGroup(name="Predicted Route (PathSnap)")
        pred_coords = _arcs_to_polyline(predicted_arcs)
        if pred_coords:
            folium.PolyLine(
                pred_coords,
                color="#DC2626",
                weight=4,
                opacity=0.9,
                dash_array="6, 6",
                tooltip="Predicted Route",
            ).add_to(pred_group)
        pred_group.add_to(route_map)

    folium.LayerControl().add_to(route_map)
    return route_map


def render_map_html(route_map: folium.Map) -> str:
    """Render Folium map into HTML string."""
    return route_map.get_root().render()
