"""Shared evaluation metrics for PathSnap.

Every table in Report Section 3 (Model Comparison), Section 4 (Bayesian Tuning),
and Section 5 (HMM Reconstruction) must compute metrics through this module.

Standardized Metrics:
1. Top-1 Accuracy: Fraction of GPS observations where predicted arc_id == true_arc_id.
2. Macro F1-Score:
   - Candidate-level binary Macro F1: mean F1 of class 0 and class 1.
   - Observation-level multi-class Macro F1: macro-averaged F1 over active arc_id classes.
3. Route Continuity: Percentage of consecutive prediction pairs (a_{t-1}, a_t) that are
   either identical, directly connected in .arcs, or topologically connected via short path.
4. Candidate Miss Rate: Fraction of GPS observations where true_arc_id is absent from candidates.
5. Safe Log Loss: Binary cross-entropy on candidate probabilities (clipped to [eps, 1-eps]).
6. Inference Time: Wall-clock execution time in seconds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def compute_top1_accuracy(
    y_true_arc: Sequence[int] | np.ndarray | pd.Series,
    y_pred_arc: Sequence[int] | np.ndarray | pd.Series,
) -> float:
    """Calculate observation-level Top-1 accuracy.

    Args:
        y_true_arc: Sequence of true road segment IDs per observation.
        y_pred_arc: Sequence of predicted road segment IDs per observation.

    Returns:
        Fraction of correctly predicted observations in [0.0, 1.0].
    """
    y_true = np.asarray(y_true_arc)
    y_pred = np.asarray(y_pred_arc)

    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}."
        )

    return float(np.mean(y_true == y_pred))


def compute_macro_f1(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    zero_division: float = 0.0,
) -> float:
    """Calculate macro-averaged F1 score.

    Works for both candidate-level binary targets and observation-level
    multi-class arc predictions.

    Args:
        y_true: Ground truth labels (binary 0/1 or multi-class arc IDs).
        y_pred: Predicted labels.
        zero_division: Value to return when there is a zero division.

    Returns:
        Macro F1 score in [0.0, 1.0].
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0:
        return 0.0

    return float(f1_score(y_t, y_p, average="macro", zero_division=zero_division))


def compute_route_continuity(
    pred_arcs: Sequence[int] | np.ndarray | pd.Series,
    arcs_df: pd.DataFrame,
    graph: Any = None,
) -> float:
    """Calculate route continuity percentage for consecutive observation predictions.

    A transition between a_{t-1} and a_t is considered continuous if:
    1. a_{t-1} == a_t (vehicle stayed on same segment)
    2. to_node(a_{t-1}) == from_node(a_t) (directly connected in .arcs)
    3. If graph is provided, reachable via a short graph path (e.g. <= 2 edges).

    Args:
        pred_arcs: Chronological sequence of predicted arc_id values for t = 0 ... T-1.
        arcs_df: DataFrame of road arcs with columns ['from_node_id', 'to_node_id']
            where the 0-based row index corresponds to arc_id.
        graph: Optional NetworkX directed graph or RoadNetworkGraph instance.

    Returns:
        Route continuity percentage in [0.0, 100.0].
    """
    arcs = list(pred_arcs)
    if len(arcs) <= 1:
        return 100.0

    # Ensure arcs_df has columns or index
    if "from_node_id" not in arcs_df.columns or "to_node_id" not in arcs_df.columns:
        # Check if 2 unnamed columns exist
        if len(arcs_df.columns) >= 2:
            from_nodes = arcs_df.iloc[:, 0].to_dict()
            to_nodes = arcs_df.iloc[:, 1].to_dict()
        else:
            raise ValueError("arcs_df must have from_node_id and to_node_id columns.")
    else:
        from_nodes = arcs_df["from_node_id"].to_dict()
        to_nodes = arcs_df["to_node_id"].to_dict()

    continuous_count = 0
    total_transitions = len(arcs) - 1

    for i in range(total_transitions):
        a_prev = arcs[i]
        a_curr = arcs[i + 1]

        # Case 1: Same arc
        if a_prev == a_curr:
            continuous_count += 1
            continue

        # Case 2: Direct connection
        prev_to = to_nodes.get(a_prev)
        curr_from = from_nodes.get(a_curr)
        if prev_to is not None and curr_from is not None and prev_to == curr_from:
            continuous_count += 1
            continue

        # Case 3: Graph shortest path check if graph is supplied
        if graph is not None:
            is_connected = False
            try:
                if hasattr(graph, "has_path"):
                    is_connected = graph.has_path(prev_to, curr_from, max_hops=2)
                elif hasattr(graph, "nx_graph"):
                    import networkx as nx
                    if nx.has_path(graph.nx_graph, prev_to, curr_from):
                        path_len = nx.shortest_path_length(graph.nx_graph, prev_to, curr_from)
                        if path_len <= 2:
                            is_connected = True
            except Exception:
                is_connected = False

            if is_connected:
                continuous_count += 1

    return float((continuous_count / total_transitions) * 100.0)


def compute_candidate_miss_rate(candidates_df: pd.DataFrame) -> float:
    """Calculate the candidate miss rate across observations.

    Candidate Miss Rate is the fraction of GPS observations where the true road
    segment is not included in the candidate set.

    Args:
        candidates_df: Candidate DataFrame with columns ['obs_idx', 'is_target'].

    Returns:
        Miss rate in [0.0, 1.0].
    """
    if candidates_df is None or candidates_df.empty:
        return 0.0

    if "obs_idx" not in candidates_df.columns or "is_target" not in candidates_df.columns:
        return 0.0

    obs_grouped = candidates_df.groupby("obs_idx")["is_target"].max()
    if len(obs_grouped) == 0:
        return 0.0

    # is_target max == 1 means candidate hit; max <= 0 means candidate miss
    misses = (obs_grouped <= 0).sum()
    return float(misses / len(obs_grouped))


def compute_log_loss_safe(
    y_true_bin: Sequence[int] | np.ndarray | pd.Series,
    pred_prob: Sequence[float] | np.ndarray | pd.Series,
    eps: float = 1e-15,
) -> Optional[float]:
    """Calculate binary cross-entropy log loss safely with probability clipping.

    Returns None for non-probabilistic or single-class edge cases.

    Args:
        y_true_bin: Binary ground truth (0 or 1).
        pred_prob: Predicted probabilities P(is_target = 1) in [0.0, 1.0].
        eps: Epsilon clipping boundary.

    Returns:
        Log loss as float, or None if computation is invalid.
    """
    y_t = np.asarray(y_true_bin)
    p_p = np.asarray(pred_prob)

    if len(y_t) == 0 or len(p_p) == 0:
        return None
    if len(y_t) != len(p_p):
        raise ValueError("Length mismatch between ground truth and predicted probabilities.")

    # Unique classes check
    unique_classes = np.unique(y_t)
    if len(unique_classes) < 2 and not (0 in unique_classes and 1 in unique_classes):
        # Only one class present in slice
        p_clipped = np.clip(p_p, eps, 1.0 - eps)
        if unique_classes[0] == 1:
            return float(-np.mean(np.log(p_clipped)))
        else:
            return float(-np.mean(np.log(1.0 - p_clipped)))

    p_clipped = np.clip(p_p, eps, 1.0 - eps)
    return float(log_loss(y_t, p_clipped, labels=[0, 1]))


def evaluate_predictions(
    point_gt_df: pd.DataFrame,
    pred_arcs: Sequence[int] | np.ndarray | pd.Series,
    arcs_df: pd.DataFrame,
    candidates_df: Optional[pd.DataFrame] = None,
    inference_time_sec: float = 0.0,
    graph: Any = None,
) -> dict[str, Optional[float]]:
    """Compute the standardized PathSnap evaluation metrics dictionary.

    Args:
        point_gt_df: Point-level ground truth DataFrame with 'true_arc_id'.
        pred_arcs: Sequence of predicted arc IDs for each point.
        arcs_df: Road arcs DataFrame with from/to nodes.
        candidates_df: Optional candidate DataFrame for candidate miss rate and log loss.
        inference_time_sec: Wall-clock execution time in seconds.
        graph: Optional road network graph.

    Returns:
        Dictionary of metrics conforming to Contract 5.
    """
    y_true_arc = point_gt_df["true_arc_id"].values
    top1 = compute_top1_accuracy(y_true_arc, pred_arcs)
    macro_f1 = compute_macro_f1(y_true_arc, pred_arcs)
    continuity = compute_route_continuity(pred_arcs, arcs_df, graph=graph)

    miss_rate: Optional[float] = None
    logloss_val: Optional[float] = None

    if candidates_df is not None and not candidates_df.empty:
        miss_rate = compute_candidate_miss_rate(candidates_df)
        if "pred_prob" in candidates_df.columns and "is_target" in candidates_df.columns:
            valid_mask = candidates_df["is_target"].isin([0, 1])
            if valid_mask.any():
                logloss_val = compute_log_loss_safe(
                    candidates_df.loc[valid_mask, "is_target"],
                    candidates_df.loc[valid_mask, "pred_prob"],
                )

    return {
        "macro_f1": macro_f1,
        "top1_accuracy": top1,
        "route_continuity": continuity,
        "log_loss": logloss_val,
        "candidate_miss_rate": miss_rate,
        "inference_time_sec": float(inference_time_sec),
    }
