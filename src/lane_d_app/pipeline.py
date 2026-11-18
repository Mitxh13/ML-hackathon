"""End-to-end PathSnap inference and evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from src.contracts.schemas import DecoderOutput, EvaluationResponse
from src.lane_a_data.loaders import load_trip
from src.lane_a_data.preprocessing import preprocess_track
from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.features import extract_candidate_features
from src.lane_b_models.evaluate_models import predict_candidates
from src.lane_c_hmm.decoder import ViterbiRouteDecoder
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
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.decoder = decoder or ViterbiRouteDecoder()
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

        # Preprocess track
        clean_track = preprocess_track(track_df)

        # Ground truth if available (strictly server-side)
        point_gt_df = None
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

        # Step 1: Candidate Generation (Lane A)
        candidates_df = self.candidate_generator.generate_candidates(
            trip_id=trip_id,
            track_df=clean_track,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            point_gt_df=point_gt_df,
        )

        # Step 2: Feature Extraction (Lane A)
        candidates_df = extract_candidate_features(
            candidates_df=candidates_df,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            track_df=clean_track,
        )

        # Step 3: Model Scoring (Lane B)
        if self.model is not None:
            scored_candidates = predict_candidates(
                model=self.model,
                candidates_df=candidates_df,
                feature_cols=self.feature_names,
            )
        else:
            # Fallback mock probability based on distance
            scored_candidates = candidates_df.copy()
            # Invert distance into pseudo-probability
            scored_candidates["pred_prob"] = np.exp(-0.5 * (scored_candidates["dist_to_arc_m"] / 15.0) ** 2)
            scored_candidates["pred_label"] = (scored_candidates["pred_prob"] >= 0.5).astype(int)

        # Step 4: Decoding (Lane C)
        graph = RoadNetworkGraph(nodes_df, arcs_df)
        decoder_output: DecoderOutput = self.decoder.decode_trip(
            candidates_df=scored_candidates,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            mode=mode,
            graph=graph,
        )

        # Step 5: Metrics (Lane C)
        metrics = {
            "macro_f1": None,
            "top1_accuracy": None,
            "route_continuity": None,
            "log_loss": None,
            "candidate_miss_rate": None,
            "inference_time_sec": decoder_output.inference_time_sec,
        }

        if point_gt_df is not None:
            metrics = evaluate_predictions(
                point_gt_df=point_gt_df,
                pred_arcs=decoder_output.point_predictions,
                arcs_df=arcs_df,
                candidates_df=scored_candidates,
                inference_time_sec=decoder_output.inference_time_sec,
                graph=graph,
            )

        # Step 6: Folium Map (Lane C)
        ref_route = trip_data["route"]["arc_id"].values.tolist() if has_route else None
        folium_map = create_route_map(
            track_df=clean_track,
            nodes_df=nodes_df,
            arcs_df=arcs_df,
            predicted_arcs=decoder_output.reconstructed_route,
            reference_route=ref_route,
        )
        map_html = render_map_html(folium_map)

        return EvaluationResponse(
            trip_id=trip_id,
            num_gps_points=len(clean_track),
            point_predictions=decoder_output.point_predictions,
            reconstructed_route=decoder_output.reconstructed_route,
            metrics=metrics,
            folium_map_html=map_html,
        )
