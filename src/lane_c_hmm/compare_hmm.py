"""Comparison framework between Classifier-only and Classifier+HMM decoding."""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from src.contracts.schemas import DecoderOutput
from src.lane_c_hmm.decoder import ViterbiRouteDecoder
from src.lane_c_hmm.metrics import evaluate_predictions
from src.lane_c_hmm.road_graph import RoadNetworkGraph


def compare_decoders(
    candidates_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    point_gt_df: Optional[pd.DataFrame] = None,
    graph: Optional[RoadNetworkGraph] = None,
) -> pd.DataFrame:
    """Run and compare multiple decoding strategies on the same candidate dataset.

    Modes compared:
    1. Distance Baseline Viterbi ("distance_hmm")
    2. Classifier-only argmax ("classifier_only")
    3. ML Probability HMM ("ml_hmm")

    Args:
        candidates_df: Candidate DataFrame with features/probabilities.
        nodes_df: Nodes DataFrame.
        arcs_df: Arcs DataFrame.
        point_gt_df: Optional ground truth for metric evaluation.
        graph: Optional road network graph.

    Returns:
        DataFrame summarizing comparative performance.
    """
    if graph is None:
        graph = RoadNetworkGraph(nodes_df, arcs_df)

    decoder = ViterbiRouteDecoder()
    modes = ["distance_baseline", "classifier_only", "ml_hmm"]
    results = []

    for mode in modes:
        output: DecoderOutput = decoder.decode_trip(
            candidates_df=candidates_df,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            mode=mode,
            graph=graph,
        )

        row_data = {
            "mode": mode,
            "inference_time_sec": output.inference_time_sec,
            "reconstructed_segments_count": len(output.reconstructed_route),
        }

        if point_gt_df is not None:
            metrics = evaluate_predictions(
                point_gt_df=point_gt_df,
                pred_arcs=output.point_predictions,
                arcs_df=arcs_df,
                candidates_df=candidates_df,
                inference_time_sec=output.inference_time_sec,
                graph=graph,
            )
            row_data.update(metrics)

        results.append(row_data)

    return pd.DataFrame(results)
