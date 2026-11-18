"""Viterbi route decoder supporting both Distance Baseline and ML+HMM modes."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.contracts.schemas import DecoderOutput
from src.lane_c_hmm.road_graph import RoadNetworkGraph, haversine_distance_m


def collapse_consecutive_duplicates(seq: Sequence[int]) -> list[int]:
    """Collapse consecutive identical arc IDs into a deduplicated route sequence.

    Example: [10, 10, 10, 15, 15, 22] -> [10, 15, 22]
    """
    if not seq:
        return []
    collapsed = [seq[0]]
    for item in seq[1:]:
        if item != collapsed[-1]:
            collapsed.append(item)
    return collapsed


class ViterbiRouteDecoder:
    """Hidden Markov Model Viterbi Decoder for map matching trajectories.

    Supports:
    - mode="distance_baseline": Gaussian emission based on dist_to_arc_m.
    - mode="ml_hmm": Supervised ML probability emission based on pred_prob.
    - mode="classifier_only": Independent point-wise argmax (no transitions).

    Attributes:
        sigma_emission_m: Standard deviation for Gaussian distance emission (meters).
        beta_transition_m: Exponential scale parameter for transition distance difference.
        unconnected_penalty: Log probability penalty for topologically disconnected arcs.
        large_gap_threshold_s: Timestamp gap threshold above which transition penalty is reset.
    """

    def __init__(
        self,
        sigma_emission_m: float = 15.0,
        beta_transition_m: float = 30.0,
        unconnected_penalty: float = -20.0,
        large_gap_threshold_s: float = 5.0,
    ) -> None:
        self.sigma_emission_m = sigma_emission_m
        self.beta_transition_m = beta_transition_m
        self.unconnected_penalty = unconnected_penalty
        self.large_gap_threshold_s = large_gap_threshold_s

    def decode_trip(
        self,
        candidates_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        mode: str = "ml_hmm",
        graph: Optional[RoadNetworkGraph] = None,
    ) -> DecoderOutput:
        """Decode most probable sequence of road segments for a trip.

        Args:
            candidates_df: Candidate DataFrame conforming to Contract 1 or 2.
            nodes_df: Nodes DataFrame.
            arcs_df: Arcs DataFrame.
            mode: "distance_baseline", "ml_hmm", or "classifier_only".
            graph: Optional pre-constructed RoadNetworkGraph instance.

        Returns:
            DecoderOutput conforming to Contract 4.
        """
        start_time = time.perf_counter()

        if candidates_df.empty:
            return DecoderOutput(
                trip_id="unknown",
                mode=mode,
                point_predictions=[],
                reconstructed_route=[],
                inference_time_sec=0.0,
            )

        trip_id = str(candidates_df["trip_id"].iloc[0])
        obs_indices = sorted(candidates_df["obs_idx"].unique())
        t_len = len(obs_indices)

        if graph is None:
            graph = RoadNetworkGraph(nodes_df, arcs_df)

        # Mode: classifier_only (independent argmax)
        if mode == "classifier_only":
            point_predictions = []
            for obs_idx in obs_indices:
                obs_cands = candidates_df[candidates_df["obs_idx"] == obs_idx]
                if "pred_prob" in obs_cands.columns:
                    best_arc = int(obs_cands.loc[obs_cands["pred_prob"].idxmax()]["candidate_arc_id"])
                else:
                    best_arc = int(obs_cands.loc[obs_cands["dist_to_arc_m"].idxmin()]["candidate_arc_id"])
                point_predictions.append(best_arc)

            reconstructed_route = collapse_consecutive_duplicates(point_predictions)
            elapsed = time.perf_counter() - start_time
            return DecoderOutput(
                trip_id=trip_id,
                mode=mode,
                point_predictions=point_predictions,
                reconstructed_route=reconstructed_route,
                inference_time_sec=elapsed,
            )

        # Group candidates by observation index
        cands_by_obs: list[pd.DataFrame] = [
            candidates_df[candidates_df["obs_idx"] == o].reset_index(drop=True) for o in obs_indices
        ]

        # Trellis initialization
        viterbi_probs: list[np.ndarray] = []
        backpointers: list[list[int]] = []

        # Step t = 0
        cands_0 = cands_by_obs[0]
        k0 = len(cands_0)
        v0 = np.zeros(k0, dtype=float)

        for i in range(k0):
            v0[i] = self._calc_log_emission(cands_0.iloc[i], mode)

        viterbi_probs.append(v0)

        # Forward pass t = 1 ... T-1
        for t in range(1, t_len):
            cands_prev = cands_by_obs[t - 1]
            cands_curr = cands_by_obs[t]
            k_prev = len(cands_prev)
            k_curr = len(cands_curr)

            delta_t = float(cands_curr["timestamp"].iloc[0] - cands_prev["timestamp"].iloc[0])
            is_large_gap = delta_t > self.large_gap_threshold_s

            v_curr = np.full(k_curr, -np.inf)
            bp_curr = [0] * k_curr

            for j in range(k_curr):
                curr_row = cands_curr.iloc[j]
                curr_arc = int(curr_row["candidate_arc_id"])
                log_emit = self._calc_log_emission(curr_row, mode)

                best_prev_score = -np.inf
                best_prev_idx = 0

                for i in range(k_prev):
                    prev_arc = int(cands_prev.iloc[i]["candidate_arc_id"])

                    if is_large_gap:
                        log_trans = 0.0
                    else:
                        log_trans = self._calc_log_transition(prev_arc, curr_arc, graph)

                    candidate_score = viterbi_probs[t - 1][i] + log_trans
                    if candidate_score > best_prev_score:
                        best_prev_score = candidate_score
                        best_prev_idx = i

                v_curr[j] = best_prev_score + log_emit
                bp_curr[j] = best_prev_idx

            viterbi_probs.append(v_curr)
            backpointers.append(bp_curr)

        # Backtrack
        best_last_idx = int(np.argmax(viterbi_probs[-1]))
        best_path_indices = [best_last_idx]

        for t in range(t_len - 2, -1, -1):
            best_prev_idx = backpointers[t][best_path_indices[-1]]
            best_path_indices.append(best_prev_idx)

        best_path_indices.reverse()

        point_predictions = [
            int(cands_by_obs[t].iloc[best_path_indices[t]]["candidate_arc_id"])
            for t in range(t_len)
        ]
        reconstructed_route = collapse_consecutive_duplicates(point_predictions)
        elapsed = time.perf_counter() - start_time

        return DecoderOutput(
            trip_id=trip_id,
            mode=mode,
            point_predictions=point_predictions,
            reconstructed_route=reconstructed_route,
            inference_time_sec=elapsed,
        )

    def _calc_log_emission(self, row: pd.Series, mode: str) -> float:
        """Calculate log emission probability for a candidate."""
        eps = 1e-12
        if mode == "distance_baseline":
            dist = float(row["dist_to_arc_m"])
            # Gaussian emission: - d^2 / (2 * sigma^2)
            return float(-0.5 * (dist / self.sigma_emission_m) ** 2)
        else:
            # ML emission
            prob = float(row.get("pred_prob", 0.5))
            prob = np.clip(prob, eps, 1.0 - eps)
            return float(np.log(prob))

    def _calc_log_transition(
        self, prev_arc: int, curr_arc: int, graph: RoadNetworkGraph
    ) -> float:
        """Calculate log transition probability between two road segments."""
        if prev_arc == curr_arc:
            return 0.0

        _, prev_to = graph.get_arc_nodes(prev_arc)
        curr_from, _ = graph.get_arc_nodes(curr_arc)

        if prev_to == curr_from:
            return -0.1  # Direct seamless road connection

        # Check shortest path distance
        dist = graph.shortest_path_distance(prev_to, curr_from)
        if np.isinf(dist):
            return self.unconnected_penalty

        return float(-dist / self.beta_transition_m)
