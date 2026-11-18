"""Shared evaluation metrics for PathSnap (Section 3.1).

Every table in Report Section 3 (Model Comparison), Section 4 (Bayesian Tuning) and
Section 5 (HMM Reconstruction) must compute its numbers through this module.

Definitions
-----------
1. Observation-level Top-1 accuracy: share of *labelled* GPS observations whose predicted
   arc equals the reference arc. Observations without a reference label (the alignment left
   them unlabeled because the reference route is > 100 m away) are excluded.
2. Macro F1:
   - candidate level (binary): unweighted mean of the F1 of class 0 and class 1 over
     (observation, candidate) rows -> ``compute_macro_f1(is_target, pred_label)``;
   - observation level (multi-class): macro F1 over the arc ids of one trip
     -> ``compute_macro_f1(true_arc_id, predicted_arc_id)``. Arc ids are local to a trip,
     so over a partition this is averaged trip by trip, never pooled.
3. Route continuity (percent): share of consecutive prediction pairs (a[t-1], a[t]) that are
   the same arc, directly connected (to_node(a[t-1]) == from_node(a[t])), or connected by a
   road path no longer than ``max_speed_mps * dt + slack_m`` (only checked when a road
   graph is supplied).
4. Candidate miss rate: share of labelled observations whose reference arc is not in the
   candidate set.
5. Safe log loss: binary cross-entropy of ``pred_prob`` against ``is_target`` with
   probabilities clipped to [eps, 1 - eps]; ``None`` when no probabilities exist.
6. Route-level (sequence) precision / recall / F1: overlap between the set of arcs the
   predicted route uses and the set of reference arcs.
7. Inference time: wall-clock seconds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

DEFAULT_MAX_SPEED_MPS = 50.0
DEFAULT_CONTINUITY_SLACK_M = 50.0


def _labelled(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions whose reference label is missing; return int64 arrays."""
    p = np.asarray(y_pred)
    raw = y_true.to_numpy() if isinstance(y_true, pd.Series) and y_true.dtype.kind in "iub" else y_true
    if isinstance(raw, np.ndarray) and raw.dtype.kind in "iub":  # fast path: plain integer labels
        if len(raw) != len(p):
            raise ValueError(f"Length mismatch: y_true has {len(raw)} items, y_pred has {len(p)}.")
        return raw.astype(np.int64), p.astype(np.int64)
    t = pd.array(y_true, dtype="Int64") if isinstance(y_true, (pd.Series, pd.api.extensions.ExtensionArray)) \
        else pd.array(np.asarray(y_true, dtype=object), dtype="Int64")
    if len(t) != len(p):
        raise ValueError(f"Length mismatch: y_true has {len(t)} items, y_pred has {len(p)}.")
    mask = ~np.asarray(pd.isna(t), dtype=bool)
    return np.asarray(t[mask], dtype=np.int64), p[mask].astype(np.int64)


def compute_top1_accuracy(y_true_arc, y_pred_arc) -> float:
    """Share of labelled observations whose predicted arc equals the reference arc."""
    t, p = _labelled(y_true_arc, y_pred_arc)
    if len(t) == 0:
        return 0.0
    return float(np.mean(t == p))


def compute_macro_f1(y_true, y_pred, zero_division: float = 0.0) -> float:
    """Macro-averaged F1 (binary candidate labels or multi-class arc ids of one trip).

    Positions with a missing reference label are ignored.
    """
    t, p = _labelled(y_true, y_pred)
    if len(t) == 0:
        return 0.0
    return float(f1_score(t, p, average="macro", zero_division=zero_division))


def _arc_endpoints(arcs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "from_node_id" in arcs_df.columns and "to_node_id" in arcs_df.columns:
        return arcs_df["from_node_id"].to_numpy(np.int64), arcs_df["to_node_id"].to_numpy(np.int64)
    if arcs_df.shape[1] >= 2:
        return arcs_df.iloc[:, 0].to_numpy(np.int64), arcs_df.iloc[:, 1].to_numpy(np.int64)
    raise ValueError("arcs_df must have from_node_id and to_node_id columns.")


def continuity_flags(
    pred_arcs: Sequence[int],
    arcs_df: pd.DataFrame,
    graph: Any = None,
    timestamps: Optional[Sequence[float]] = None,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    slack_m: float = DEFAULT_CONTINUITY_SLACK_M,
) -> np.ndarray:
    """Boolean array (length T-1): is the transition a[t-1] -> a[t] physically continuous?"""
    arcs = np.asarray(pred_arcs, dtype=np.int64)
    if len(arcs) <= 1:
        return np.zeros(0, dtype=bool)
    frm, to = _arc_endpoints(arcs_df)
    prev, curr = arcs[:-1], arcs[1:]
    ok = (prev == curr) | (to[prev] == frm[curr])
    if graph is not None and hasattr(graph, "shortest_path_distance"):
        dt = np.ones(len(prev)) if timestamps is None else np.maximum(np.diff(np.asarray(timestamps, float)), 1.0)
        for i in np.flatnonzero(~ok):
            limit = max_speed_mps * dt[i] + slack_m
            d = graph.shortest_path_distance(int(to[prev[i]]), int(frm[curr[i]]), limit_m=limit)
            ok[i] = bool(np.isfinite(d))
    return ok


def compute_route_continuity(
    pred_arcs,
    arcs_df: pd.DataFrame,
    graph: Any = None,
    timestamps: Optional[Sequence[float]] = None,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
    slack_m: float = DEFAULT_CONTINUITY_SLACK_M,
) -> float:
    """Route continuity in percent [0, 100] (see module docstring, definition 3)."""
    flags = continuity_flags(pred_arcs, arcs_df, graph, timestamps, max_speed_mps, slack_m)
    if len(flags) == 0:
        return 100.0
    return float(flags.mean() * 100.0)


def compute_candidate_miss_rate(candidates_df: pd.DataFrame) -> Optional[float]:
    """Share of labelled observations whose reference arc is absent from the candidates.

    Observations whose candidates are all unlabeled (``is_target == -1``) are excluded.
    Returns ``None`` when there is no labelled observation.
    """
    if candidates_df is None or candidates_df.empty or "is_target" not in candidates_df.columns:
        return None
    keys = [k for k in ("trip_id", "obs_idx") if k in candidates_df.columns]
    best = candidates_df.groupby(keys)["is_target"].max()
    best = best[best >= 0]
    if len(best) == 0:
        return None
    return float((best == 0).mean())


def compute_log_loss_safe(y_true_bin, pred_prob, eps: float = 1e-15) -> Optional[float]:
    """Binary cross-entropy with clipped probabilities; ``None`` if it cannot be computed."""
    y = np.asarray(y_true_bin)
    p = np.asarray(pred_prob, dtype=float)
    if len(y) == 0 or len(p) == 0:
        return None
    if len(y) != len(p):
        raise ValueError("Length mismatch between ground truth and predicted probabilities.")
    p = np.clip(p, eps, 1.0 - eps)
    classes = np.unique(y)
    if len(classes) < 2:
        return float(-np.mean(np.log(p))) if classes[0] == 1 else float(-np.mean(np.log(1.0 - p)))
    return float(log_loss(y, p, labels=[0, 1]))


def compute_route_overlap(
    pred_arcs: Sequence[int],
    reference_arcs: Sequence[int],
    arc_length_m: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Sequence-level overlap between the predicted route and the reference route.

    precision = share of predicted arcs that are reference arcs, recall = share of reference
    arcs that were predicted. With ``arc_length_m`` both are weighted by arc length.
    """
    pred = np.unique(np.asarray(pred_arcs, dtype=np.int64))
    ref = np.unique(np.asarray(reference_arcs, dtype=np.int64))
    if len(pred) == 0 or len(ref) == 0:
        return {"route_precision": 0.0, "route_recall": 0.0, "route_f1": 0.0}
    hit_p = np.isin(pred, ref)
    hit_r = np.isin(ref, pred)
    if arc_length_m is not None:
        wp, wr = np.maximum(arc_length_m[pred], 1e-6), np.maximum(arc_length_m[ref], 1e-6)
        precision = float((wp * hit_p).sum() / wp.sum())
        recall = float((wr * hit_r).sum() / wr.sum())
    else:
        precision, recall = float(hit_p.mean()), float(hit_r.mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"route_precision": precision, "route_recall": recall, "route_f1": float(f1)}


def evaluate_predictions(
    point_gt_df: pd.DataFrame,
    pred_arcs,
    arcs_df: pd.DataFrame,
    candidates_df: Optional[pd.DataFrame] = None,
    inference_time_sec: float = 0.0,
    graph: Any = None,
    timestamps: Optional[Sequence[float]] = None,
) -> Dict[str, Optional[float]]:
    """Standard PathSnap metrics for one trip (Contract 5 keys plus route-level extras).

    ``pred_arcs`` must be aligned row by row with ``point_gt_df``.
    Keys: top1_accuracy, macro_f1 (observation level), route_continuity (percent),
    log_loss, candidate_miss_rate, inference_time_sec, candidate_macro_f1 (binary, if
    ``pred_label`` is available), route_precision / route_recall / route_f1 (reference arcs
    = arcs of the labelled observations), label_coverage, n_points.
    """
    y_true = point_gt_df["true_arc_id"]
    pred = np.asarray(pred_arcs, dtype=np.int64)
    if timestamps is None and "timestamp" in point_gt_df.columns:
        timestamps = point_gt_df["timestamp"].to_numpy(float)

    labelled = ~pd.isna(pd.array(y_true, dtype="Int64"))
    metrics: Dict[str, Optional[float]] = {
        "top1_accuracy": compute_top1_accuracy(y_true, pred),
        "macro_f1": compute_macro_f1(y_true, pred),
        "route_continuity": compute_route_continuity(pred, arcs_df, graph=graph, timestamps=timestamps),
        "log_loss": None,
        "candidate_miss_rate": None,
        "inference_time_sec": float(inference_time_sec),
    }
    if candidates_df is not None and not candidates_df.empty:
        metrics["candidate_miss_rate"] = compute_candidate_miss_rate(candidates_df)
        if "is_target" in candidates_df.columns:
            rows = candidates_df[candidates_df["is_target"].isin([0, 1])]
            if "pred_prob" in rows.columns and len(rows):
                metrics["log_loss"] = compute_log_loss_safe(rows["is_target"], rows["pred_prob"])
            if "pred_label" in rows.columns and len(rows):
                metrics["candidate_macro_f1"] = compute_macro_f1(rows["is_target"], rows["pred_label"])
    ref = np.asarray(pd.array(y_true, dtype="Int64")[labelled], dtype=np.int64)
    lengths = getattr(graph, "arc_length_m", None)
    metrics.update(compute_route_overlap(pred[labelled], ref, lengths))
    metrics["label_coverage"] = float(labelled.mean()) if len(labelled) else 0.0
    metrics["n_points"] = float(len(pred))
    return metrics
