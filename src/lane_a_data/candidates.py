"""Metric spatial-index candidate generation for directed road arcs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_a_data.ground_truth import local_transformer


class CandidateGenerator:
    """Find exact point-to-segment nearest arcs using a midpoint KD-tree broad phase."""
    def __init__(self, max_dist_m: float = 100.0, max_k: int = 10,
                 min_radius_m: float = 50.0, expand_radius: bool = True,
                 max_radius_m: float = 300.0) -> None:
        if max_dist_m <= 0 or max_k <= 0 or min_radius_m <= 0 or max_radius_m < min_radius_m:
            raise ValueError("radii and max_k must be positive and max_radius_m >= min_radius_m")
        self.max_dist_m, self.max_k = float(max_dist_m), int(max_k)
        self.min_radius_m, self.max_radius_m = float(min_radius_m), float(max_radius_m)
        self.expand_radius = bool(expand_radius)

    @staticmethod
    def _distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b-a
        denom = np.einsum("ij,ij->i", ab, ab)
        frac = np.einsum("tij,ij->ti", points[:, None, :] - a[None, :, :], ab) / np.where(denom == 0, 1, denom)[None, :]
        closest = a[None, :, :] + np.clip(frac, 0, 1)[:, :, None] * ab[None, :, :]
        return np.linalg.norm(points[:, None, :] - closest, axis=2)

    def generate_candidates(self, trip_id: str, track_df: pd.DataFrame, nodes_df: pd.DataFrame,
                            arcs_df: pd.DataFrame, point_gt_df: pd.DataFrame | None = None,
                            include_section12_fields: bool = False) -> pd.DataFrame:
        """Generate candidates using the legacy base schema by default.

        ``include_section12_fields=True`` adds rank, reference aliases and fallback
        diagnostics for the Section 1.2 batch outputs. The default remains compatible
        with the existing feature-extraction and model-scoring callers.
        """
        extra_cols = ["candidate_rank", "reference_arc_id", "is_reference", "distance_m", "label", "fallback_used"]
        cols = CANDIDATE_BASE_COLS + (extra_cols if include_section12_fields else [])
        if arcs_df.empty or nodes_df.empty:
            raise ValueError("Cannot generate candidates from an empty road network")
        track_lon, track_lat = track_df.gps_lon.to_numpy(float), track_df.gps_lat.to_numpy(float)
        node_lon, node_lat = nodes_df.lon.to_numpy(float), nodes_df.lat.to_numpy(float)
        if not (np.isfinite(track_lon).all() and np.isfinite(track_lat).all() and (np.abs(track_lon) <= 180).all() and (np.abs(track_lat) <= 90).all()):
            raise ValueError("track contains invalid or non-finite WGS84 coordinates")
        if not (np.isfinite(node_lon).all() and np.isfinite(node_lat).all() and (np.abs(node_lon) <= 180).all() and (np.abs(node_lat) <= 90).all()):
            raise ValueError("nodes contain invalid or non-finite WGS84 coordinates")
        transformer, _ = local_transformer(np.r_[track_lon, node_lon], np.r_[track_lat, node_lat])
        node_xy = np.column_stack(transformer.transform(node_lon, node_lat))
        endpoints = arcs_df[["from_node_id", "to_node_id"]].to_numpy(int)
        if endpoints.min() < 0 or endpoints.max() >= len(nodes_df):
            raise ValueError("arcs contain node IDs outside nodes table")
        a, b = node_xy[endpoints[:, 0]], node_xy[endpoints[:, 1]]
        centers = (a+b)/2
        half_lengths = np.linalg.norm(b-a, axis=1)/2
        tree = cKDTree(centers)
        gps = np.column_stack(transformer.transform(track_lon, track_lat))
        gt = {}
        if point_gt_df is not None and "reference_arc_id" in point_gt_df:
            gt = dict(zip(point_gt_df.obs_idx, point_gt_df.reference_arc_id))
        elif point_gt_df is not None and "true_arc_id" in point_gt_df:
            gt = dict(zip(point_gt_df.obs_idx, point_gt_df.true_arc_id))
        records = []
        for i, (obs, point) in enumerate(zip(track_df.itertuples(index=False), gps)):
            obs_idx = int(getattr(obs, "obs_idx", i))
            truth = gt.get(obs_idx, pd.NA)
            if pd.isna(truth):
                truth = pd.NA
            radius = min(max(self.max_dist_m, self.min_radius_m), self.max_radius_m)
            chosen_ids, distances, fallback = np.array([], dtype=int), np.array([]), False
            while True:
                # Midpoint distance <= search radius + segment half length is a safe broad phase.
                nearby = tree.query_ball_point(point, radius + half_lengths.max())
                ids = np.asarray([j for j in nearby if np.linalg.norm(point-centers[j]) <= radius + half_lengths[j]], dtype=int)
                if len(ids):
                    d = self._distance(point.reshape(1, 2), a[ids], b[ids])[0]
                    mask = d <= radius
                    ids, d = ids[mask], d[mask]
                    order = np.lexsort((ids, d))
                    chosen_ids, distances = ids[order][:self.max_k], d[order][:self.max_k]
                if len(chosen_ids) or not self.expand_radius or radius >= self.max_radius_m:
                    break
                radius = min(self.max_radius_m, radius * 2)
            if len(chosen_ids) == 0:
                # Keep an explicit nearest-road fallback; the summary records its use.
                _, nearest = tree.query(point, k=1)
                chosen_ids = np.array([int(nearest)])
                distances = self._distance(point.reshape(1, 2), a[chosen_ids], b[chosen_ids])[0]
                fallback = True
            for rank, (arc_id, distance) in enumerate(zip(chosen_ids, distances), start=1):
                label = int(arc_id == int(truth)) if not pd.isna(truth) else -1
                record = {"trip_id": str(trip_id), "obs_idx": obs_idx,
                    "timestamp": float(obs.timestamp), "gps_lon": float(obs.gps_lon), "gps_lat": float(obs.gps_lat),
                    "candidate_arc_id": int(arc_id), "from_node_id": int(endpoints[arc_id, 0]),
                    "to_node_id": int(endpoints[arc_id, 1]), "dist_to_arc_m": float(distance),
                    "is_target": label}
                if include_section12_fields:
                    record.update({"candidate_rank": rank, "reference_arc_id": truth,
                    "is_reference": (bool(label == 1) if label >= 0 else pd.NA), "distance_m": float(distance),
                    "label": label, "fallback_used": fallback})
                records.append(record)
        result = pd.DataFrame(records, columns=cols)
        # Shared contract validators allow feature columns only; validate its base projection.
        validate_candidate_df(result[["trip_id", "obs_idx", "timestamp", "gps_lon", "gps_lat",
            "candidate_arc_id", "from_node_id", "to_node_id", "dist_to_arc_m", "is_target"]])
        return result
