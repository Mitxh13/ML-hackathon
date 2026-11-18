"""Command line pipeline for reference alignment and candidate generation."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd
from src.lane_a_data.loaders import load_trip
from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.validation import candidate_metrics

LOG = logging.getLogger("pathsnap.labels_candidates")


def run_partition(data_dir: Path, partition: str, output_dir: Path,
                  generator: CandidateGenerator, results_dir: Path = Path("results")) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = data_dir / partition
    tracks = sorted(root.rglob("*.track")) if root.exists() else []
    alignments, candidate_tables = [], []
    for track_path in tracks:
        trip_dir, trip_id = track_path.parent, track_path.stem
        LOG.info("Processing %s/%s", partition, trip_id)
        trip = load_trip(trip_dir)
        required = {"track", "nodes", "arcs", "route"}
        missing = required - trip.keys()
        if missing:
            raise ValueError(f"{trip_id}: missing trip files {sorted(missing)}")
        alignment = assign_point_ground_truth(trip_id, trip["track"], trip["route"], trip["nodes"], trip["arcs"])
        candidates = generator.generate_candidates(trip_id, trip["track"], trip["nodes"], trip["arcs"], alignment,
                                                 include_section12_fields=True)
        # Persist the Section 1.2 name while retaining true_arc_id in the shared API.
        alignment = alignment.rename(columns={"true_arc_id": "reference_arc_id"})
        alignments.append(alignment)
        candidate_tables.append(candidates)
    alignment_df = pd.concat(alignments, ignore_index=True) if alignments else pd.DataFrame(columns=[
        "trip_id", "obs_idx", "reference_arc_id", "reference_route_order", "alignment_distance_m", "alignment_status"])
    candidate_df = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame(columns=[
        "trip_id", "obs_idx", "timestamp", "gps_lon", "gps_lat", "candidate_arc_id", "from_node_id", "to_node_id",
        "dist_to_arc_m", "distance_m", "is_target", "candidate_rank", "reference_arc_id", "is_reference", "label", "fallback_used"])
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_df.to_csv(output_dir / f"reference_alignment_{partition}.csv", index=False)
    if partition == "student_test":
        # Keep development-evaluation targets out of the candidate model-input file.
        label_cols = ["trip_id", "obs_idx", "candidate_arc_id", "reference_arc_id", "is_reference", "label"]
        candidate_df[label_cols].to_csv(output_dir / "candidate_labels_student_test.csv", index=False)
        model_cols = [c for c in candidate_df.columns if c not in {"reference_arc_id", "is_reference", "label", "is_target"}]
        candidate_df[model_cols].to_csv(output_dir / "candidate_segments_student_test.csv", index=False)
    else:
        candidate_df.to_csv(output_dir / f"candidate_segments_{partition}.csv", index=False)
    if partition == "train" and not alignment_df.empty:
        recalls, summaries = candidate_metrics(alignment_df, candidate_df)
        results_dir.mkdir(parents=True, exist_ok=True)
        recalls.to_csv(results_dir / "candidate_recall.csv", index=False)
        summaries.to_csv(results_dir / "candidate_generation_summary.csv", index=False)
        (results_dir / "candidate_generation_report.md").write_text(
            "# Candidate generation validation\n\n"
            f"Parameters: initial radius {generator.max_dist_m:g} m, maximum radius {generator.max_radius_m:g} m, "
            f"maximum {generator.max_k} candidates/observation; radius expansion={generator.expand_radius}.\n\n"
            "Recall and candidate counts are calculated from observed training trips and are in the adjacent CSV files. "
            "Fallback nearest arcs are flagged in the candidate table. Alignment failures are left unlabeled.\n")
    return alignment_df, candidate_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--radius-m", type=float, default=100)
    parser.add_argument("--max-radius-m", type=float, default=300)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"], choices=["train", "student_test"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generator = CandidateGenerator(args.radius_m, args.max_k, max_radius_m=args.max_radius_m)
    for partition in args.partitions:
        run_partition(args.data_dir, partition, args.output_dir, generator)
    if all(not list((args.data_dir / p).rglob("*.track")) for p in args.partitions):
        LOG.warning("No trips found. No synthetic or fabricated CSV results were written.")


if __name__ == "__main__":
    main()
