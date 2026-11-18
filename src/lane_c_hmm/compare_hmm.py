"""Side-by-side comparison of decoding strategies on one trip (Section 3.3).

Modes compared on the same candidates and the same road graph:
    nearest          closest candidate arc per point (no model, no sequence)
    classifier_only  arg-max of the classifier probability per point (no sequence)
    distance_hmm     HMM/Viterbi with a Gaussian distance emission (no model)
    ml_hmm           HMM/Viterbi with the classifier probability as emission
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.lane_c_hmm.decoder import MODES, ViterbiRouteDecoder, collapse_consecutive_duplicates
from src.lane_c_hmm.metrics import evaluate_predictions
from src.lane_c_hmm.road_graph import RoadNetworkGraph


def compare_decoders(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    point_gt_df: Optional[pd.DataFrame] = None,
    graph: Optional[RoadNetworkGraph] = None,
    decoder: Optional[ViterbiRouteDecoder] = None,
    modes: Sequence[str] = MODES,
) -> pd.DataFrame:
    """One row per mode with the shared metrics (when ``point_gt_df`` is given).

    Modes that need probabilities are skipped when ``candidates_df`` has no ``pred_prob``.
    Inference time = shared trellis preparation (road distances) + that mode's decoding.
    """
    graph = graph or RoadNetworkGraph(nodes_df, arcs_df)
    decoder = decoder or ViterbiRouteDecoder()
    obs_index = point_gt_df["obs_idx"].to_numpy() if point_gt_df is not None else None
    trellis = decoder.build_trellis(candidates_df, graph, obs_index)
    has_prob = "pred_prob" in candidates_df.columns
    rows = []
    for mode in modes:
        if mode in ("classifier_only", "ml_hmm") and not has_prob:
            continue
        t0 = time.perf_counter()
        out = decoder.predict_arcs(trellis, mode)
        seconds = time.perf_counter() - t0 + (trellis.prep_time_sec if mode.endswith("hmm") else 0.0)
        preds = out["point_predictions"]
        row: Dict[str, object] = {
            "mode": mode,
            "inference_time_sec": seconds,
            "reconstructed_segments_count": len(collapse_consecutive_duplicates(preds)),
            "disconnected_steps": out["disconnected_steps"],
        }
        if point_gt_df is not None:
            metrics = evaluate_predictions(point_gt_df, preds, arcs_df, candidates_df,
                                           inference_time_sec=seconds, graph=graph)
            if mode in ("nearest", "distance_hmm"):
                metrics["log_loss"] = None  # no probabilistic classifier involved
            row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
