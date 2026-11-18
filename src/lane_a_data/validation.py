"""Candidate and reference-alignment diagnostics."""
from __future__ import annotations
import pandas as pd


def candidate_metrics(alignment: pd.DataFrame, candidates: pd.DataFrame, radii=(50, 100, 200, 300), top_ks=(1, 3, 5, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-trip/overall recall rows and candidate-count/alignment summaries."""
    valid = alignment[alignment.reference_arc_id.notna()].copy()
    keys = ["trip_id", "obs_idx"]
    merged = candidates.merge(valid[keys + ["reference_arc_id"]], on=keys, how="inner", suffixes=("", "_alignment"))
    if "reference_arc_id_alignment" in merged:
        truth = merged.reference_arc_id_alignment
    else:
        truth = merged.reference_arc_id
    merged["hit"] = merged.candidate_arc_id.eq(truth)
    records = []
    groups = [(str(t), g, False) for t, g in valid.groupby("trip_id")] + [("OVERALL", valid, True)]
    for trip_id, obs, overall in groups:
        sub = merged if overall else merged[merged.trip_id.astype(str).eq(trip_id)]
        # Composite keys prevent observation-index collisions in the overall partition.
        obs_keys = list(map(tuple, obs[keys].to_numpy()))
        hit_keys = set(map(tuple, sub.loc[sub.hit, keys].drop_duplicates().to_numpy()))
        hits = sum(key in hit_keys for key in obs_keys)
        n = len(obs)
        row = {"trip_id": trip_id, "valid_reference_observations": n,
               "candidate_recall": hits/n if n else float("nan"), "candidate_miss_count": n-hits,
               "candidate_miss_rate": (n-hits)/n if n else float("nan")}
        for radius in radii:
            near = set(map(tuple, sub.loc[(sub.dist_to_arc_m <= radius) & sub.hit, keys].drop_duplicates().to_numpy()))
            row[f"recall_radius_{radius}m"] = sum(key in near for key in obs_keys)/n if n else float("nan")
        for k in top_ks:
            near = set(map(tuple, sub.loc[(sub.candidate_rank <= k) & sub.hit, keys].drop_duplicates().to_numpy()))
            row[f"recall_top_{k}"] = sum(key in near for key in obs_keys)/n if n else float("nan")
        records.append(row)
    summary_rows = []
    for trip_id, obs in list(alignment.groupby("trip_id")) + [("OVERALL", alignment)]:
        obs_index = pd.MultiIndex.from_frame(obs[keys])
        c = candidates.groupby(keys).size().reindex(obs_index, fill_value=0)
        failed = obs.alignment_status.ne("aligned")
        distances = pd.to_numeric(obs.alignment_distance_m, errors="coerce")
        trip_candidates = candidates if str(trip_id) == "OVERALL" else candidates[candidates.trip_id.astype(str).eq(str(trip_id))]
        summary_rows.append({"trip_id": str(trip_id), "observations": len(obs),
            "average_candidates": float(c.mean()) if len(c) else 0.0,
            "minimum_candidates": int(c.min()) if len(c) else 0,
            "maximum_candidates": int(c.max()) if len(c) else 0,
            "no_candidate_observations": int((c == 0).sum()),
            "fallback_candidate_records": int(trip_candidates.fallback_used.sum()) if "fallback_used" in trip_candidates else 0,
            "alignment_failures": int(failed.sum()),
            "alignment_distance_mean_m": float(distances.mean()) if distances.notna().any() else float("nan"),
            "alignment_distance_median_m": float(distances.median()) if distances.notna().any() else float("nan"),
            "alignment_distance_p95_m": float(distances.quantile(.95)) if distances.notna().any() else float("nan")})
    return pd.DataFrame(records), pd.DataFrame(summary_rows)
