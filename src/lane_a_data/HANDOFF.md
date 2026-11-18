# Section 1.2 handoff to Section 1.3

Run from the repository root with `.venv/bin/python -m src.lane_a_data.pipeline`. The command reads only `data/train` and `data/student_test`; evaluator-test is deliberately excluded. It writes:

- `data/intermediate/reference_alignment_train.csv` and `reference_alignment_student_test.csv`
- `data/intermediate/candidate_segments_train.csv` and `candidate_segments_student_test.csv`
- `data/intermediate/candidate_labels_student_test.csv` (evaluation-only targets, separate from student candidate inputs)
- Training diagnostics: `results/candidate_recall.csv`, `results/candidate_generation_summary.csv`, and `results/candidate_generation_report.md`

The Python `assign_point_ground_truth()` result preserves the shared schema columns `trip_id`, `obs_idx`, `timestamp`, `gps_lon`, `gps_lat`, and nullable `true_arc_id`, validated by `validate_point_gt_df`. Its extra diagnostics are `reference_route_order` (nullable position in ordered `.route`), `alignment_distance_m` and `alignment_status`. The pipeline writes the CSV adapter name `reference_arc_id` instead of `true_arc_id`; the two names refer to the same single label, and the CSV does not include both. Failed observations retain their row with a null label; do not train from them.

CandidateGenerator preserves the legacy `CANDIDATE_BASE_COLS` output by default for existing consumers. The Section 1.2 batch pipeline opts in to the additional output fields. Training batch output includes base columns `trip_id`, `obs_idx`, `timestamp` (seconds), `gps_lon`/`gps_lat` (WGS84 degrees), `candidate_arc_id` (original directed arc ID), `from_node_id`, `to_node_id`, `dist_to_arc_m` (metres), and `is_target` (1 positive, 0 negative, -1 unlabeled). Additional fields are `candidate_rank` (1-based distance order), nullable `reference_arc_id`, `distance_m` (metres; alias of `dist_to_arc_m`), nullable boolean `is_reference`, `label` (same values as `is_target`), and `fallback_used`. Reverse arcs remain separate candidates. Student candidate input omits target columns; `candidate_labels_student_test.csv` contains candidate keys plus reference labels for evaluation only.

Join Section 1.3 features to candidate rows on `(trip_id, obs_idx, candidate_arc_id)`; join observation features on `(trip_id, obs_idx)`. Keep `candidate_rank`, metric distance and labels as provided. Use training labels only for model fitting. Student-test labels are for development evaluation and must not enter training. No evaluator-test files are read. A zero-candidate observation is represented by its alignment row with no candidate rows and is counted in diagnostics; fallback rows are explicitly marked.

Coordinates are projected from `[longitude, latitude]` EPSG:4326 to a trip-specific UTM zone in metres for alignment and candidate distances. CSV GPS columns remain in degrees. Candidate search begins at the configured radius (default 100 m), expands by doubling up to 300 m when enabled, keeps at most 10 candidates, then uses a flagged nearest-arc fallback when still empty. These defaults are configurable on the CLI and should be revisited using the generated recall tables when raw trips are present.
