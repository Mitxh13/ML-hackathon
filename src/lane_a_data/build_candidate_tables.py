"""Build and persist feature-engineered candidate tables (Section 1.3: Features & Candidate Tables).

Processes all trips in data/train/ (80 trips) and data/student_test/ (10 trips),
extracts comprehensive features conforming to Contract 1, and saves consolidated
candidate tables into data/processed/ (Parquet & CSV).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_a_data.loaders import load_trip
from src.lane_a_data.preprocessing import preprocess_track
from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.candidates import CandidateGenerator
from src.lane_a_data.features import extract_candidate_features


def process_partition(
    data_dir: str | Path,
    partition: str = "train",
    max_dist_m: float = 100.0,
    max_k: int = 10,
    max_trips: Optional[int] = None,
) -> pd.DataFrame:
    """Process all trips in a partition to build a single consolidated candidate table.

    Args:
        data_dir: Base data directory (e.g. 'data').
        partition: 'train' or 'student_test'.
        max_dist_m: Candidate search radius in meters.
        max_k: Top-k candidates per observation.
        max_trips: Optional limit on number of trips to process (for testing).

    Returns:
        Consolidated candidate DataFrame with all base and engineered feature columns.
    """
    part_dir = Path(data_dir) / partition
    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    trip_dirs = sorted([d for d in part_dir.iterdir() if d.is_dir()])
    if max_trips is not None:
        trip_dirs = trip_dirs[:max_trips]

    print(f"\n=================================================================")
    print(f" Processing {len(trip_dirs)} trips from '{partition}' partition")
    print(f" Search radius: {max_dist_m}m | Max k: {max_k} candidates/obs")
    print(f"=================================================================")

    generator = CandidateGenerator(max_dist_m=max_dist_m, max_k=max_k)
    all_candidate_dfs: list[pd.DataFrame] = []
    total_start_time = time.time()

    for i, t_dir in enumerate(trip_dirs, 1):
        trip_id = t_dir.name
        t0 = time.time()

        try:
            trip_data = load_trip(t_dir)
            if "track" not in trip_data or "nodes" not in trip_data or "arcs" not in trip_data:
                print(f"[{i}/{len(trip_dirs)}] Warning: Trip {trip_id} missing required files, skipping.")
                continue

            track_df = trip_data["track"]
            nodes_df = trip_data["nodes"]
            arcs_df = trip_data["arcs"]
            has_route = "route" in trip_data

            # 1. Preprocess track kinematics
            clean_track = preprocess_track(track_df)

            # 2. Align ground truth if route exists
            point_gt_df = None
            if has_route:
                point_gt_df = assign_point_ground_truth(
                    trip_id=trip_id,
                    track_df=clean_track,
                    route_df=trip_data["route"],
                    nodes_df=nodes_df,
                    arcs_df=arcs_df,
                )

            # 3. Generate spatial candidates
            candidates_df = generator.generate_candidates(
                trip_id=trip_id,
                track_df=clean_track,
                nodes_df=nodes_df,
                arcs_df=arcs_df,
                point_gt_df=point_gt_df,
            )

            # 4. Extract comprehensive features (22 features)
            enriched_df = extract_candidate_features(
                candidates_df=candidates_df,
                nodes_df=nodes_df,
                arcs_df=arcs_df,
                track_df=clean_track,
            )

            all_candidate_dfs.append(enriched_df)
            elapsed = time.time() - t0

            hit_rate = 0.0
            if "is_target" in enriched_df.columns and (enriched_df["is_target"] == 1).any():
                n_obs = enriched_df["obs_idx"].nunique()
                n_hits = enriched_df[enriched_df["is_target"] == 1]["obs_idx"].nunique()
                hit_rate = (n_hits / max(n_obs, 1)) * 100.0

            if i % 10 == 0 or i == len(trip_dirs) or i == 1:
                print(
                    f"[{i:2d}/{len(trip_dirs)}] Trip {trip_id}: "
                    f"{len(clean_track):4d} obs -> {len(enriched_df):5d} candidates | "
                    f"Candidate Hit Rate: {hit_rate:.1f}% | Time: {elapsed:.2f}s"
                )

        except Exception as e:
            print(f"[{i}/{len(trip_dirs)}] Error processing Trip {trip_id}: {e}")

    if not all_candidate_dfs:
        raise RuntimeError(f"No candidate DataFrames generated for partition '{partition}'.")

    consolidated_df = pd.concat(all_candidate_dfs, ignore_index=True)
    total_time = time.time() - total_start_time

    print(f"\n Partition '{partition}' Complete in {total_time:.2f}s:")
    print(f"   Total Observations: {consolidated_df.groupby(['trip_id', 'obs_idx']).ngroups:,}")
    print(f"   Total Candidate Rows: {len(consolidated_df):,}")
    print(f"   Total Features: {len([c for c in consolidated_df.columns if c.startswith('feat_')])}")
    if "is_target" in consolidated_df.columns:
        counts = consolidated_df["is_target"].value_counts().to_dict()
        print(f"   Target Distribution: {counts}")

    return consolidated_df


def save_candidate_tables(
    df: pd.DataFrame,
    output_dir: str | Path,
    partition: str = "train",
) -> tuple[Path, Path]:
    """Save consolidated candidate table to Parquet and CSV in output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_path / f"{partition}_candidates.parquet"
    csv_file = out_path / f"{partition}_candidates.csv"

    print(f"Saving {partition} candidates to {parquet_file}...")
    df.to_parquet(parquet_file, index=False, engine="pyarrow")

    print(f"Saving {partition} sample CSV (first 5,000 rows) to {csv_file}...")
    df.head(5000).to_csv(csv_file, index=False)

    return parquet_file, csv_file


def main():
    parser = argparse.ArgumentParser(description="Build PathSnap Candidate Tables (Section 1.3)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed candidate tables")
    parser.add_argument("--partition", choices=["all", "train", "student_test"], default="all", help="Partition to build")
    parser.add_argument("--radius", type=float, default=100.0, help="Search radius in meters")
    parser.add_argument("--max-k", type=int, default=10, help="Max candidates per observation")
    parser.add_argument("--limit-trips", type=int, default=None, help="Optional trip limit for quick runs")

    args = parser.parse_args()

    partitions = ["train", "student_test"] if args.partition == "all" else [args.partition]

    for part in partitions:
        df = process_partition(
            data_dir=args.data_dir,
            partition=part,
            max_dist_m=args.radius,
            max_k=args.max_k,
            max_trips=args.limit_trips,
        )
        save_candidate_tables(df, output_dir=args.output_dir, partition=part)

    print("\n All candidate tables built and validated successfully.")


if __name__ == "__main__":
    main()
