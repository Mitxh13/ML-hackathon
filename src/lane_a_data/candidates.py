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
    def _pair_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Exact distance from points[i] to segment (a[i], b[i]), row by row, in metres."""
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        frac = np.einsum("ij,ij->i", points - a, ab) / np.where(denom == 0, 1, denom)
        closest = a + np.clip(frac, 0, 1)[:, None] * ab
        return np.linalg.norm(points - closest, axis=1)

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
        n_obs = len(gps)
        obs_ids = (track_df["obs_idx"].to_numpy(int) if "obs_idx" in track_df.columns
                   else np.arange(n_obs))
        gt = {}
        if point_gt_df is not None and "reference_arc_id" in point_gt_df:
            gt = dict(zip(point_gt_df.obs_idx, point_gt_df.reference_arc_id))
        elif point_gt_df is not None and "true_arc_id" in point_gt_df:
            gt = dict(zip(point_gt_df.obs_idx, point_gt_df.true_arc_id))

        # All points are searched at once. A point keeps the (up to max_k) arcs whose exact
        # point-to-segment distance is within its radius, ordered by (distance, arc_id).
        # Points with nothing in range retry with a doubled radius (up to max_radius_m).
        chosen_pt, chosen_arc, chosen_d = [], [], []
        radius = np.full(n_obs, min(max(self.max_dist_m, self.min_radius_m), self.max_radius_m))
        todo = np.arange(n_obs)
        while len(todo):
            # Midpoint distance <= search radius + segment half length is a safe broad phase.
            nearby = tree.query_ball_point(gps[todo], radius[todo] + half_lengths.max())
            counts = np.fromiter((len(x) for x in nearby), dtype=np.int64, count=len(todo))
            pt = np.repeat(todo, counts)
            arc = (np.fromiter((j for x in nearby for j in x), dtype=np.int64, count=int(counts.sum()))
                   if counts.sum() else np.array([], dtype=np.int64))
            d = self._pair_distance(gps[pt], a[arc], b[arc])
            keep = d <= radius[pt]
            pt, arc, d = pt[keep], arc[keep], d[keep]
            order = np.lexsort((arc, d, pt))
            pt, arc, d = pt[order], arc[order], d[order]
            first = np.searchsorted(pt, pt, side="left")
            top = (np.arange(len(pt)) - first) < self.max_k
            chosen_pt.append(pt[top]); chosen_arc.append(arc[top]); chosen_d.append(d[top])
            found = np.zeros(n_obs, dtype=bool)
            found[pt] = True
            todo = todo[~found[todo]]
            if not self.expand_radius:
                break
            todo = todo[radius[todo] < self.max_radius_m]
            radius[todo] = np.minimum(self.max_radius_m, radius[todo] * 2)
        pt = np.concatenate(chosen_pt) if chosen_pt else np.array([], dtype=np.int64)
        arc = np.concatenate(chosen_arc) if chosen_arc else np.array([], dtype=np.int64)
        d = np.concatenate(chosen_d) if chosen_d else np.array([])
        fallback = np.zeros(len(pt), dtype=bool)
        empty = np.setdiff1d(np.arange(n_obs), pt)
        if len(empty):
            # Keep an explicit nearest-road fallback; the summary records its use.
            _, nearest = tree.query(gps[empty], k=1)
            nearest = np.asarray(nearest, dtype=np.int64)
            pt = np.r_[pt, empty]
            arc = np.r_[arc, nearest]
            d = np.r_[d, self._pair_distance(gps[empty], a[nearest], b[nearest])]
            fallback = np.r_[fallback, np.ones(len(empty), dtype=bool)]
        order = np.lexsort((arc, d, pt))
        pt, arc, d, fallback = pt[order], arc[order], d[order], fallback[order]
        rank = np.arange(len(pt)) - np.searchsorted(pt, pt, side="left") + 1

        obs = obs_ids[pt]
        truth = pd.array([gt.get(int(o), pd.NA) for o in obs_ids], dtype="Int64")[pt]
        has_truth = ~pd.isna(truth)
        label = np.where(has_truth, arc == truth.fillna(-1).to_numpy(np.int64), -1).astype(int)
        records = {
            "trip_id": np.full(len(pt), str(trip_id), dtype=object), "obs_idx": obs,
            "timestamp": track_df["timestamp"].to_numpy(float)[pt],
            "gps_lon": track_lon[pt], "gps_lat": track_lat[pt],
            "candidate_arc_id": arc, "from_node_id": endpoints[arc, 0], "to_node_id": endpoints[arc, 1],
            "dist_to_arc_m": d, "is_target": label,
        }
        if include_section12_fields:
            is_ref = pd.array(np.where(has_truth, label == 1, False), dtype="boolean")
            is_ref[~has_truth] = pd.NA
            records.update({"candidate_rank": rank, "reference_arc_id": truth, "is_reference": is_ref,
                            "distance_m": d, "label": label, "fallback_used": fallback})
        result = pd.DataFrame(records, columns=cols)
        # Shared contract validators allow feature columns only; validate its base projection.
        validate_candidate_df(result[["trip_id", "obs_idx", "timestamp", "gps_lon", "gps_lat",
            "candidate_arc_id", "from_node_id", "to_node_id", "dist_to_arc_m", "is_target"]])
        return result
