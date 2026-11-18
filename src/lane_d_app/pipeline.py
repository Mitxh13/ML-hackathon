"""End-to-end PathSnap inference and evaluation pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.contracts.schemas import DecoderOutput, EvaluationResponse
from src.lane_a_data.loaders import load_trip
from src.lane_a_data.preprocessing import preprocess_track
from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.features import extract_candidate_features
from src.lane_b_models.evaluate_models import predict_candidates
from src.lane_c_hmm.decoder import MODE_ALIASES, ViterbiRouteDecoder, load_tuned_decoders
from src.lane_c_hmm.metrics import evaluate_predictions
from src.lane_c_hmm.road_graph import RoadNetworkGraph
from src.lane_c_hmm.folium_maps import create_route_map, render_map_html


class PathSnapPipeline:
    """End-to-end inference and evaluation pipeline for map-matching and route reconstruction.

    Supports both:
    - Mock/baseline mode for early development before models are finalized.
    - Full production pipeline (A -> B -> C -> D).
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[list[str]] = None,
        decoder: Optional[ViterbiRouteDecoder] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        hmm_params_path: str = "configs/hmm_params.json",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        # One decoder per mode with the settings tuned on training trips (configs/hmm_params.json).
        self.decoders = load_tuned_decoders(hmm_params_path)
        if decoder is not None:
            self.decoders = {mode: decoder for mode in self.decoders}
        self.decoder = self.decoders["ml_hmm"]
        self.candidate_generator = candidate_generator or CandidateGenerator()

    def process_trip(
        self,
        trip_id: str,
        trip_dir: Union[str, Path],
        mode: str = "ml_hmm",
    ) -> EvaluationResponse:
        """Execute end-to-end pipeline on a trip directory and return EvaluationResponse.

        Args:
            trip_id: Trip identifier string (e.g. '00000000').
            trip_dir: Directory containing .track, .nodes, .arcs, and optional .route.
            mode: 'ml_hmm', 'distance_baseline', or 'classifier_only'.

        Returns:
            EvaluationResponse conforming to Contract 5.
        """
        dir_path = Path(trip_dir)
        trip_data = load_trip(dir_path)

        if "track" not in trip_data or "nodes" not in trip_data or "arcs" not in trip_data:
            raise ValueError(f"Trip directory {trip_dir} missing required .track, .nodes, or .arcs files.")

        track_df = trip_data["track"]
        nodes_df = trip_data["nodes"]
        arcs_df = trip_data["arcs"]
        has_route = "route" in trip_data

        # Prediction pipeline (timed). Ground truth is NOT used anywhere in these steps.
        t_start = time.perf_counter()
        clean_track = preprocess_track(track_df)

        # Step 1: Candidate Generation (Lane A) - no labels passed in
        candidates_df = self.candidate_generator.generate_candidates(
            trip_id=trip_id,
            track_df=clean_track,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            point_gt_df=None,
        )

        # Step 2: Feature Extraction (Lane A)
        candidates_df = extract_candidate_features(
            candidates_df=candidates_df,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            track_df=clean_track,
        )

        # Step 3: Model Scoring (Lane B)
        decoder_mode = MODE_ALIASES.get(mode, mode)
        if self.model is not None:
            scored_candidates = predict_candidates(
                model=self.model,
                candidates_df=candidates_df,
                feature_cols=self.feature_names,
            )
        else:
            # No trained model available: decode with the classifier-free distance HMM
            # instead of inventing probabilities, and report the mode actually used.
            scored_candidates = candidates_df
            if decoder_mode in ("ml_hmm", "classifier_only"):
                decoder_mode = "distance_hmm"

        # Step 4: Decoding (Lane C)
        graph = RoadNetworkGraph(nodes_df, arcs_df)
        decoder = self.decoders.get(decoder_mode, self.decoder)
        decoder_output: DecoderOutput = decoder.decode_trip(
            candidates_df=scored_candidates,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            mode=decoder_mode,
            graph=graph,
            obs_index=clean_track["obs_idx"].to_numpy(),
        )
        pipeline_seconds = time.perf_counter() - t_start

        # Step 5: Metrics (Lane C) - reference route used server-side only, after prediction
        metrics = {
            "macro_f1": None,
            "top1_accuracy": None,
            "route_continuity": None,
            "log_loss": None,
            "candidate_miss_rate": None,
            "inference_time_sec": pipeline_seconds,
        }

        if has_route:
            try:
                point_gt_df = assign_point_ground_truth(
                    trip_id=trip_id,
                    track_df=clean_track,
                    route_df=trip_data["route"],
                    nodes_df=nodes_df,
                    arcs_df=arcs_df,
                )
            except Exception:
                point_gt_df = None
            if point_gt_df is not None:
                truth = point_gt_df.set_index("obs_idx")["true_arc_id"]
                labelled = scored_candidates.copy()
                ref = labelled["obs_idx"].map(truth)
                labelled["is_target"] = np.where(
                    ref.isna(), -1, (labelled["candidate_arc_id"] == ref.fillna(-1)).astype(int)
                )
                metrics = evaluate_predictions(
                    point_gt_df=point_gt_df,
                    pred_arcs=decoder_output.point_predictions,
                    arcs_df=arcs_df,
                    candidates_df=labelled,
                    inference_time_sec=pipeline_seconds,
                    graph=graph,
                )
        metrics["decode_time_sec"] = decoder_output.inference_time_sec

        # Step 6: Folium Map (Lane C)
        # This map is sent to Laptop 2, so it must NOT contain the reference route
        # (ground truth stays server-side). Reference-vs-predicted maps for the
        # report are produced offline by lane C's compare_hmm / folium_maps.
        display_route = getattr(decoder_output, "connected_route", None) or decoder_output.reconstructed_route
        folium_map = create_route_map(
            track_df=clean_track,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            predicted_arcs=display_route,
            reference_route=None,
            show_network=False,
        )
        map_html = render_map_html(folium_map)

        return EvaluationResponse(
            trip_id=trip_id,
            num_gps_points=len(clean_track),
            point_predictions=decoder_output.point_predictions,
            reconstructed_route=decoder_output.reconstructed_route,
            metrics=metrics,
            folium_map_html=map_html,
            # Extra display geometry for the web UI ([lat, lon] order, as Leaflet expects).
            gps=_gps_latlon(clean_track),
            predicted_route=_route_polylines(display_route, nodes_df, arcs_df),
            decoder_mode=decoder_mode,
        )


def _gps_latlon(track_df: pd.DataFrame) -> list[list[float]]:
    """GPS observations as [lat, lon] pairs (input data, not ground truth)."""
    return np.round(track_df[["gps_lat", "gps_lon"]].to_numpy(dtype=float), 6).tolist()


def _route_polylines(route: list[int], nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> list[list[list[float]]]:
    """Predicted arcs as [[lat, lon], [lat, lon]] segments. Row index == arc_id / node_id."""
    if not route:
        return []
    arcs = arcs_df[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)[np.asarray(route, dtype=np.int64)]
    latlon = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
    return np.round(np.stack([latlon[arcs[:, 0]], latlon[arcs[:, 1]]], axis=1), 6).tolist()
