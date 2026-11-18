"""Run the part 1.1 exploratory data analysis and write every output.

Usage (from the repo root):
    .venv/Scripts/python -m src.lane_a_data.run_eda            # Windows
    .venv/bin/python -m src.lane_a_data.run_eda                # Linux / macOS

Reads only data/train and data/student_test. The evaluator partition is never
opened. Writes:
    results/tables/eda_*.csv and eda_data_quality_notes.md
    results/figures/eda_*.png
    results/maps/eda_*.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.lane_a_data import eda_plots as plots
from src.lane_a_data.eda_stats import HEADING_METHODS, load_projected_trip, trip_statistics
from src.lane_a_data.loaders import list_trip_dirs
from src.lane_a_data.preprocessing import HEADING_HALF_WINDOW, SMOOTH_WINDOW

DEV_PARTITIONS = ("train", "student_test")

# (column in the inventory, label in the quality table)
QUALITY_CHECKS = [
    ("issue_missing_files", "Missing trip files"),
    ("issue_nan_values", "NaN / empty values"),
    ("issue_invalid_gps_coords", "GPS coordinates out of range or at (0, 0)"),
    ("issue_invalid_node_coords", "Node coordinates out of range or at (0, 0)"),
    ("issue_arcs_with_unknown_node", "Arcs pointing at a node that does not exist"),
    ("issue_route_arcs_unknown", "Route arcs that do not exist"),
    ("issue_repeated_timestamps", "Repeated timestamps"),
    ("issue_backwards_timestamps", "Timestamps going backwards"),
    ("non_integer_timestamps", "Non-integer timestamps"),
    ("gaps_over_5s", "Time gaps over 5 s"),
    ("gaps_over_300s", "Time gaps over 5 min"),
    ("speed_outliers", "Speed jumps over 252 km/h"),
    ("jitter_reversals", "Raw steps reversing direction (jitter)"),
    ("repeated_positions", "Consecutive identical GPS positions"),
    ("points_over_50m", "GPS points over 50 m from the reference route"),
    ("points_off_route_on_road", "  of which on another road (route incomplete)"),
    ("points_off_network_50m", "GPS points over 50 m from any road in the network"),
    ("route_breaks", "Route steps between unconnected arcs"),
    ("issue_self_loop_arcs", "Self-loop arcs"),
    ("issue_duplicate_arcs", "Duplicate arcs (same from and to node)"),
    ("zero_length_arcs", "Zero-length arcs"),
    ("isolated_nodes", "Nodes not used by any arc"),
]


def collect(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    arrays = {p: {"dt": [], "speed_kmh": [], "route_dist_m": [], "heading_err": {m[0]: [] for m in HEADING_METHODS}}
              for p in DEV_PARTITIONS}
    for part in DEV_PARTITIONS:
        trip_dirs = list_trip_dirs(data_dir / part)
        if not trip_dirs:
            raise FileNotFoundError(f"No trips found in {data_dir / part}. Unzip the dataset first.")
        for trip_dir in trip_dirs:
            row, arr = trip_statistics(trip_dir, part)
            rows.append(row)
            for key in ("dt", "speed_kmh", "route_dist_m"):
                arrays[part][key].append(arr[key])
            for label, errors in arr["heading_err"].items():
                arrays[part]["heading_err"][label].append(errors)
    stats = pd.DataFrame(rows)
    stats["issue_missing_files"] = stats[[c for c in stats if c.startswith("issue_missing_")
                                          and c != "issue_missing_files"]].sum(axis=1)
    stats["points_over_50m"] = (stats["share_points_over_50m"] * stats["n_gps_points"]).round().astype(int)
    pooled = {
        p: {
            **{k: np.concatenate(d[k]) for k in ("dt", "speed_kmh", "route_dist_m")},
            "heading_err": {label: np.concatenate(v) for label, v in d["heading_err"].items()},
        }
        for p, d in arrays.items()
    }
    return stats, pooled


def heading_table(pooled: dict) -> pd.DataFrame:
    """Heading error of each method on the training trips (the data used to choose the method)."""
    chosen = f"Smoothed {SMOOTH_WINDOW}, ±{HEADING_HALF_WINDOW} points (chosen)"
    rows = []
    for label, _, _ in HEADING_METHODS:
        err = pooled["train"]["heading_err"][label]
        rows.append({
            "method": label,
            "points": int(err.size),
            "median_error_deg": float(np.degrees(np.median(err))),
            "share_over_45deg": float(np.mean(err > np.pi / 4)),
            "share_wrong_direction": float(np.mean(err > np.pi / 2)),
            "chosen": label == chosen,
        })
    return pd.DataFrame(rows)


def partition_summary(stats: pd.DataFrame, pooled: dict) -> pd.DataFrame:
    rows = []
    for part, sub in stats.groupby("partition", sort=False):
        dt = pooled[part]["dt"]
        dist = pooled[part]["route_dist_m"]
        rows.append({
            "partition": part,
            "trips": len(sub),
            "gps_points_total": int(sub["n_gps_points"].sum()),
            "gps_points_min": int(sub["n_gps_points"].min()),
            "gps_points_median": float(sub["n_gps_points"].median()),
            "gps_points_max": int(sub["n_gps_points"].max()),
            "network_nodes_median": float(sub["n_nodes"].median()),
            "network_arcs_median": float(sub["n_arcs"].median()),
            "network_arcs_max": int(sub["n_arcs"].max()),
            "route_arcs_median": float(sub["n_route_arcs"].median()),
            "hours_recorded": float(sub["duration_s"].sum() / 3600.0),
            "gps_km_total": float(sub["path_length_km"].sum()),
            "gps_km_total_raw": float(sub["path_length_raw_km"].sum()),
            "route_km_total": float(sub["route_length_km"].sum()),
            "jitter_reversal_share_median": float(sub["jitter_reversal_share"].median()),
            "trips_with_jitter_over_15pct": int((sub["jitter_reversal_share"] > 0.15).sum()),
            "share_steps_1s": float(np.mean(np.abs(dt - 1.0) < 1e-3)),
            "steps_over_5s": int(np.sum(dt > 5.0)),
            "longest_gap_s": float(dt.max()),
            "speed_median_kmh": float(np.median(pooled[part]["speed_kmh"])),
            "gps_route_dist_median_m": float(np.nanmedian(dist)),
            "gps_route_dist_p95_m": float(np.nanpercentile(dist, 95)),
            "utm_zones": int(sub["crs"].nunique()),
            "two_way_arc_share_median": float(sub["two_way_arc_share"].median()),
            "largest_strong_share_median": float(sub["largest_strong_share"].median()),
        })
    return pd.DataFrame(rows)


def quality_table(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in QUALITY_CHECKS:
        for part, sub in stats.groupby("partition", sort=False):
            rows.append({"check": label, "column": col, "partition": part, "n_trips": len(sub),
                         "total": int(sub[col].sum()), "trips_affected": int((sub[col] > 0).sum())})
    return pd.DataFrame(rows)


def anomaly_table(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in stats.itertuples():
        reasons = []
        if row.dt_max_s > 300:
            reasons.append(f"longest gap {row.dt_max_s:,.0f} s")
        if row.share_dt_1s < 0.8:
            reasons.append(f"only {row.share_dt_1s:.0%} of steps are 1 s")
        if row.speed_outliers > 0:
            reasons.append(f"{row.speed_outliers} speed jump(s) over 252 km/h")
        if row.jitter_reversal_share > 0.15:
            reasons.append(f"{row.jitter_reversal_share:.0%} of raw steps reverse direction (jitter)")
        if row.share_points_over_50m > 0.05:
            reasons.append(f"{row.share_points_over_50m:.0%} of GPS points over 50 m from the reference route "
                           f"(median {row.gps_route_dist_median_m:.0f} m), {row.points_off_route_on_road:,} of them "
                           f"on another road")
        if row.points_off_network_50m > 0.05 * row.n_gps_points:
            reasons.append(f"{row.points_off_network_50m / row.n_gps_points:.0%} of GPS points over 50 m from any "
                           f"road in the supplied network (median {row.gps_network_dist_median_m:.0f} m)")
        ratio = row.path_length_km / row.route_length_km if row.route_length_km else np.nan
        if ratio > 1.3 or ratio < 0.77:
            reasons.append(f"smoothed GPS path {row.path_length_km:.1f} km vs route {row.route_length_km:.1f} km")
        if row.route_breaks > 0:
            reasons.append(f"{row.route_breaks} break(s) in the reference route")
        if row.largest_strong_share < 0.8:
            reasons.append(f"largest strongly connected component holds {row.largest_strong_share:.0%} of nodes")
        if reasons:
            rows.append({"partition": row.partition, "trip_id": row.trip_id, "n_gps_points": row.n_gps_points,
                         "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows, columns=["partition", "trip_id", "n_gps_points", "reasons"])


def pick_example_trips(stats: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(partition, trip_id, why) for a typical trip and the three most unusual ones."""
    train = stats[stats["partition"] == "train"]
    clean = train[(train["gps_route_dist_median_m"] < 15) & (train["dt_max_s"] < 60)]
    pool = clean if len(clean) else train
    typical = pool.iloc[(pool["n_gps_points"] - train["n_gps_points"].median()).abs().argsort()].iloc[0]
    picks = [(typical.partition, typical.trip_id, "typical training trip")]
    for col, ascending, why in [
        ("gps_route_dist_median_m", False, "furthest from its reference route"),
        ("dt_max_s", False, "longest time gap"),
        ("share_dt_1s", True, "most irregular sampling"),
    ]:
        for row in stats.sort_values(col, ascending=ascending).itertuples():
            if row.trip_id not in {p[1] for p in picks}:
                picks.append((row.partition, row.trip_id, why))
                break
    return picks


def write_notes(path: Path, stats: pd.DataFrame, summary: pd.DataFrame, quality: pd.DataFrame,
                anomalies: pd.DataFrame, headings: pd.DataFrame) -> None:
    s = summary.set_index("partition")
    tr, st = s.loc["train"], s.loc["student_test"]
    h = headings.set_index("method")
    raw = h.loc["Raw, consecutive points"]
    best = headings[headings["chosen"]].iloc[0]

    def q(label: str) -> str:
        sub = quality[quality["check"] == label].set_index("partition")
        return ", ".join(f"{p}: {int(sub.loc[p, 'total']):,} in {int(sub.loc[p, 'trips_affected'])} trips"
                         for p in sub.index)

    lines = [
        "# Data-quality notes (generated by `python -m src.lane_a_data.run_eda`)",
        "",
        "Partitions analysed: `train` (80 trips) and `student_test` (10 trips). `evaluator_test` was not opened.",
        "",
        "## Inventory",
        f"- Every trip has all four files (.track, .nodes, .arcs, .route). Missing files: {q('Missing trip files')}.",
        f"- GPS points per trip: train {int(tr.gps_points_min):,}–{int(tr.gps_points_max):,} "
        f"(median {tr.gps_points_median:,.0f}); student_test {int(st.gps_points_min):,}–{int(st.gps_points_max):,} "
        f"(median {st.gps_points_median:,.0f}).",
        f"- Each trip ships its own road network. Median arcs per trip: train {tr.network_arcs_median:,.0f}, "
        f"student_test {st.network_arcs_median:,.0f}; largest {int(stats['n_arcs'].max()):,}.",
        f"- Recorded time: train {tr.hours_recorded:.1f} h, student_test {st.hours_recorded:.1f} h.",
        "",
        "## Missing and invalid values",
        f"- NaN values: {q('NaN / empty values')}. Out-of-range coordinates: {q('GPS coordinates out of range or at (0, 0)')}.",
        f"- Arcs referencing unknown nodes: {q('Arcs pointing at a node that does not exist')}. "
        f"Unknown route arcs: {q('Route arcs that do not exist')}.",
        f"- Repeated timestamps: {q('Repeated timestamps')}. Backwards timestamps: {q('Timestamps going backwards')}.",
        "",
        "## Sampling and gaps",
        f"- Share of 1 s steps: train {tr.share_steps_1s:.1%}, student_test {st.share_steps_1s:.1%} "
        "(timestamps contain float noise such as 0.9999999999999, so 1 s is tested with a 1 ms tolerance).",
        f"- Steps longer than 5 s: train {int(tr.steps_over_5s):,}, student_test {int(st.steps_over_5s):,}. "
        f"Longest gap: train {tr.longest_gap_s:,.0f} s, student_test {st.longest_gap_s:,.0f} s.",
        f"- Non-integer timestamps: {q('Non-integer timestamps')}.",
        "",
        "## Geography and projection",
        f"- Trips are spread over {stats['crs'].nunique()} UTM zones (longitude {stats['centroid_lon'].min():.1f} to "
        f"{stats['centroid_lon'].max():.1f}, latitude {stats['centroid_lat'].min():.1f} to {stats['centroid_lat'].max():.1f}).",
        "- Decision: project each trip into its own UTM zone. Web Mercator was rejected because it stretches distances "
        "by 1/cos(latitude) (about 1.55x at 50 degrees).",
        "- For every trip, the track and its road network fall in the same UTM zone, so both project consistently.",
        "",
        "## GPS jitter, speed and heading",
        f"- Raw positions zigzag: median {tr.jitter_reversal_share_median:.0%} of raw training steps turn by more than "
        f"150 degrees from one second to the next; {int(tr.trips_with_jitter_over_15pct)} training trips exceed 15%.",
        f"- Because of this, raw GPS path length overstates distance: training total {tr.gps_km_total_raw:,.0f} km raw vs "
        f"{tr.gps_km_total:,.0f} km after smoothing, against {tr.route_km_total:,.0f} km of reference route.",
        f"- Heading from consecutive raw points points the wrong way (>90 degrees off the route arc) for "
        f"{raw.share_wrong_direction:.1%} of training points (median error {raw.median_error_deg:.1f} degrees). "
        f"The chosen method ({best.method.replace(' (chosen)', '')}) brings this to {best.share_wrong_direction:.1%} "
        f"(median {best.median_error_deg:.1f} degrees).",
        f"- Median moving speed (smoothed): train {tr.speed_median_kmh:.0f} km/h, student_test {st.speed_median_kmh:.0f} km/h.",
        f"- Speed jumps over 252 km/h between raw points: {q('Speed jumps over 252 km/h')}. They are flagged, not removed.",
        f"- Consecutive identical positions: {q('Consecutive identical GPS positions')}.",
        "",
        "## Road network",
        f"- Median share of arcs with a reverse twin (two-way roads): train {tr.two_way_arc_share_median:.0%}. "
        "Two directions of the same street share geometry, so heading is needed to tell them apart.",
        f"- Median share of nodes in the largest strongly connected component: train {tr.largest_strong_share_median:.0%}.",
        f"- Duplicate arcs: {q('Duplicate arcs (same from and to node)')}. Zero-length arcs: {q('Zero-length arcs')}. "
        f"Self-loops: {q('Self-loop arcs')}. Unused nodes: {q('Nodes not used by any arc')}.",
        "",
        "## Reference routes",
        f"- Route steps between unconnected arcs: {q('Route steps between unconnected arcs')}.",
        f"- Distance from GPS points to the nearest reference-route arc: train median {tr.gps_route_dist_median_m:.1f} m "
        f"(95th percentile {tr.gps_route_dist_p95_m:.1f} m); student_test median {st.gps_route_dist_median_m:.1f} m "
        f"(95th percentile {st.gps_route_dist_p95_m:.1f} m).",
        f"- GPS points more than 50 m from the reference route: {q('GPS points over 50 m from the reference route')}. "
        f"Of these, points that lie on another road of the network (the reference route skips part of the drive): "
        f"{q('  of which on another road (route incomplete)')}.",
        f"- GPS points more than 50 m from any road in the supplied network: "
        f"{q('GPS points over 50 m from any road in the network')}.",
        "- Consequence for labels (part 1.2): forcing far-off points onto the nearest route arc creates wrong labels. "
        "Points far from the route should be marked unlabelled and left out of training and scoring, and trips where "
        "most points are off the network should be reviewed before use.",
        "",
        "## Trips needing attention",
    ]
    if anomalies.empty:
        lines.append("- None.")
    for row in anomalies.itertuples():
        lines.append(f"- `{row.trip_id}` ({row.partition}, {row.n_gps_points:,} points): {row.reasons}.")
    lines += [
        "",
        "## Preprocessing decisions",
        "- Each trip is projected into its own UTM zone (`x_m`, `y_m`); raw positions are never altered.",
        "- Rows are never dropped or reordered; `obs_idx` stays the join key for every later stage.",
        "- Problems are flagged per point instead: `is_large_gap`, `is_time_reversed`, `is_speed_outlier`, `is_stationary`.",
        f"- Speed and heading use a {SMOOTH_WINDOW}-point rolling mean of positions (`x_smooth_m`, `y_smooth_m`) and a "
        f"±{HEADING_HALF_WINDOW}-point central difference. Smoothing never crosses a gap over 5 s (`segment_id`).",
        "- Heading counts only where the vehicle moved at least 2 m over the window; otherwise the previous heading "
        "is kept (`heading_valid` = False).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    args = parser.parse_args(argv)

    tables = args.results_dir / "tables"
    figures = args.results_dir / "figures"
    maps = args.results_dir / "maps"
    for d in (tables, figures, maps):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats, pooled = collect(args.data_dir)
    print(f"Measured {len(stats)} trips in {time.time() - t0:.1f}s")

    summary = partition_summary(stats, pooled)
    quality = quality_table(stats)
    anomalies = anomaly_table(stats)
    intervals = plots.sampling_interval_table({p: pooled[p]["dt"] for p in DEV_PARTITIONS})
    headings = heading_table(pooled)

    stats.to_csv(tables / "eda_trip_inventory.csv", index=False)
    summary.to_csv(tables / "eda_partition_summary.csv", index=False)
    quality.to_csv(tables / "eda_quality_checks.csv", index=False)
    anomalies.to_csv(tables / "eda_anomalies.csv", index=False)
    intervals.to_csv(tables / "eda_sampling_intervals.csv", index=False)
    headings.to_csv(tables / "eda_heading_smoothing.csv", index=False)
    write_notes(tables / "eda_data_quality_notes.md", stats, summary, quality, anomalies, headings)

    plots.plot_trip_locations(stats, figures / "eda_trip_locations.png")
    plots.plot_trip_sizes(stats, figures / "eda_trip_sizes.png")
    plots.plot_quality_checks(quality, figures / "eda_quality_checks.png")
    plots.plot_sampling_intervals(intervals, figures / "eda_sampling_intervals.png")
    plots.plot_gaps_per_trip(stats, figures / "eda_gaps_per_trip.png")
    plots.plot_speed_by_partition({p: pooled[p]["speed_kmh"] for p in DEV_PARTITIONS},
                                  figures / "eda_speed_distribution.png")
    plots.plot_gps_route_distance({p: pooled[p]["route_dist_m"] for p in DEV_PARTITIONS}, stats,
                                  figures / "eda_gps_to_route_distance.png")
    plots.plot_route_vs_track(stats, figures / "eda_route_vs_track.png")
    plots.plot_network_connectivity(stats, figures / "eda_network_connectivity.png")
    plots.plot_heading_smoothing(headings, stats, figures / "eda_heading_smoothing.png")

    picks = pick_example_trips(stats)
    trips = [load_projected_trip(args.data_dir / part / tid, part) for part, tid, _ in picks]
    captions = [f"{t.trip_id} ({t.partition}): {why}\n{len(t.track):,} GPS points, "
                f"{0 if t.route is None else len(t.route):,} route arcs, {t.crs}"
                for t, (_, _, why) in zip(trips, picks)]
    plots.plot_trip_examples(trips, captions, figures / "eda_trip_examples.png")

    plots.make_locations_map(stats, maps / "eda_trip_locations.html")
    for trip, (_, _, why) in zip(trips, picks):
        plots.make_trip_map(trip, maps / f"eda_trip_{trip.trip_id}.html")

    plots.plt.close("all")
    print(f"Example trips: {', '.join(f'{tid} ({why})' for _, tid, why in picks)}")
    print(f"Wrote tables to {tables}, figures to {figures}, maps to {maps} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
