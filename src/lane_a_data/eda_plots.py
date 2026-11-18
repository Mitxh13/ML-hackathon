"""Exploratory Data Analysis (EDA) plotting utilities for PathSnap.

Matplotlib figures for the report (PNG) and Folium maps (HTML). Every
function takes numbers computed in ``eda_stats.py``; nothing here reads raw
files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

PARTITION_COLORS = {"train": "#1F6FB2", "student_test": "#E07A1F"}
PARTITION_LABELS = {"train": "Training (80 trips)", "student_test": "Student test (10 trips)"}
GPS_COLOR = "#2B6CB0"
ROUTE_COLOR = "#1A9850"
NETWORK_COLOR = "#B5BCC5"
GAP_COLOR = "#D7301F"
FIG_DPI = 150


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E3E6EA",
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, save_path: Optional[str | Path]) -> plt.Figure:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    return fig


def _partitions(stats: pd.DataFrame) -> list[str]:
    return [p for p in PARTITION_COLORS if p in set(stats["partition"])]


# ---------------------------------------------------------------------------
# Original helpers (kept for other parts that already import them)
# ---------------------------------------------------------------------------

def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Generate missing values bar chart for a DataFrame."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df.isnull().sum()
    missing.plot(kind="bar", color=PARTITION_COLORS["train"], ax=ax)
    ax.set_title("Missing values per column")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_speed_distribution(track_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot distribution of GPS speed observations."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    speeds = track_df["speed_mps"].dropna() if "speed_mps" in track_df.columns else []
    sns.histplot(speeds, bins=30, kde=True, color=ROUTE_COLOR, ax=ax)
    ax.set_title("GPS speed distribution (m/s)")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_candidate_distances(candidates_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot candidate distance distribution separated by target vs non-target."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    if "is_target" in candidates_df.columns:
        sns.boxplot(data=candidates_df, x="is_target", y="dist_to_arc_m", hue="is_target",
                    palette="Set2", legend=False, ax=ax)
        ax.set_xticks([0, 1], ["Non-target (0)", "Target (1)"])
    else:
        sns.histplot(candidates_df["dist_to_arc_m"], bins=30, ax=ax)
    ax.set_title("Candidate distance to road segment (m)")
    ax.set_ylabel("Distance (m)")
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Dataset-level figures (part 1.1)
# ---------------------------------------------------------------------------

def plot_trip_locations(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """World scatter of each trip's median GPS position, sized by number of points."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for part in _partitions(stats):
        sub = stats[stats["partition"] == part]
        ax.scatter(sub["centroid_lon"], sub["centroid_lat"], s=8 + np.sqrt(sub["n_gps_points"]) * 0.8,
                   color=PARTITION_COLORS[part], alpha=0.75, edgecolor="white", linewidth=0.6,
                   label=PARTITION_LABELS[part])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 81, 20))
    ax.axhline(0, color="#9AA3AF", linewidth=0.8)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Where the trips are: median GPS position per trip (marker size ~ points)")
    ax.legend(loc="lower left")
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_trip_sizes(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Distributions of GPS points, network arcs and route arcs per trip."""
    _style()
    cols = [("n_gps_points", "GPS points per trip"), ("n_arcs", "Road-network arcs per trip"),
            ("n_route_arcs", "Reference-route arcs per trip")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (col, title) in zip(axes, cols):
        values = stats[col].clip(lower=1)
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 18)
        for part in _partitions(stats):
            ax.hist(stats.loc[stats["partition"] == part, col].clip(lower=1), bins=bins,
                    color=PARTITION_COLORS[part], alpha=0.75, label=PARTITION_LABELS[part])
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Count (log scale)")
        ax.set_ylabel("Trips")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_quality_checks(checks: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Horizontal bars: how many trips are affected by each data-quality check.

    ``checks`` has columns check, partition, total, trips_affected.
    """
    _style()
    order = list(dict.fromkeys(checks["check"]))
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(order) + 1.4))
    y = np.arange(len(order))
    parts = [p for p in PARTITION_COLORS if p in set(checks["partition"])]
    height = 0.8 / max(len(parts), 1)
    for i, part in enumerate(parts):
        sub = checks[checks["partition"] == part].set_index("check").reindex(order)
        n_trips = int(sub["n_trips"].iloc[0])
        share = sub["trips_affected"] / n_trips
        bars = ax.barh(y + i * height, share, height=height, color=PARTITION_COLORS[part],
                       label=PARTITION_LABELS[part])
        for bar, affected, total in zip(bars, sub["trips_affected"], sub["total"]):
            unit = "case" if int(total) == 1 else "cases"
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(affected)}/{n_trips} trips · {int(total):,} {unit}",
                    va="center", fontsize=7.5, color="#3D4652")
    ax.set_yticks(y + height * (len(parts) - 1) / 2, order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of trips affected")
    ax.set_title("Missing-value and data-quality checks")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save_path)


DT_BIN_EDGES = [0, 1.001, 2.001, 5.001, 30.001, 300.001, np.inf]
DT_BIN_LABELS = ["1 s", "2 s", "3-5 s", "6-30 s", "31-300 s", "> 300 s"]


def sampling_interval_table(dt_by_partition: dict[str, np.ndarray]) -> pd.DataFrame:
    """Count GPS time steps per interval bin and partition."""
    rows = []
    for part, dt in dt_by_partition.items():
        counts, _ = np.histogram(dt, bins=DT_BIN_EDGES)
        for label, count in zip(DT_BIN_LABELS, counts):
            rows.append({"partition": part, "interval": label, "steps": int(count),
                         "share": float(count / max(len(dt), 1))})
    return pd.DataFrame(rows)


def plot_sampling_intervals(table: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Grouped bars of the share of time steps in each interval bin (log scale)."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    parts = [p for p in PARTITION_COLORS if p in set(table["partition"])]
    x = np.arange(len(DT_BIN_LABELS))
    width = 0.8 / len(parts)
    for i, part in enumerate(parts):
        sub = table[table["partition"] == part].set_index("interval").reindex(DT_BIN_LABELS)
        bars = ax.bar(x + i * width, sub["share"].clip(lower=1e-6), width=width,
                      color=PARTITION_COLORS[part], label=PARTITION_LABELS[part])
        for bar, share, steps in zip(bars, sub["share"], sub["steps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.25,
                    f"{share:.2%}\n({int(steps):,})", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 8)
    ax.set_xticks(x + width * (len(parts) - 1) / 2, DT_BIN_LABELS)
    ax.set_xlabel("Time between consecutive GPS points")
    ax.set_ylabel("Share of steps (log scale)")
    ax.set_title("Sampling intervals: most steps are 1 s, with a long tail of gaps")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_gaps_per_trip(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Gaps over 5 s per trip and the longest gap per trip, trips sorted."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, col, title, log in [
        (axes[0], "gaps_over_5s", "Gaps longer than 5 s, per trip", False),
        (axes[1], "dt_max_s", "Longest gap per trip (s, log scale)", True),
    ]:
        sub = stats.sort_values(col, ascending=False).reset_index(drop=True)
        ax.bar(sub.index, sub[col], color=[PARTITION_COLORS[p] for p in sub["partition"]], width=0.85)
        if log:
            ax.set_yscale("log")
            ax.axhline(5, color=GAP_COLOR, linewidth=0.9, linestyle="--")
            ax.text(len(sub) - 1, 5 * 1.15, "5 s gap threshold", ha="right", fontsize=7.5, color=GAP_COLOR)
        ax.set_title(title)
        ax.set_xlabel("Trips, sorted")
        ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PARTITION_COLORS.values()]
    axes[0].legend(handles, PARTITION_LABELS.values())
    fig.tight_layout()
    return _save(fig, save_path)


def plot_speed_by_partition(speed_by_partition: dict[str, np.ndarray],
                            save_path: Optional[str | Path] = None) -> plt.Figure:
    """Histogram of moving speeds (km/h) per partition, as density."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(0, 262, 5)
    for part, speeds in speed_by_partition.items():
        ax.hist(np.clip(speeds, 0, 259), bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                color=PARTITION_COLORS[part])
        ax.hist(np.clip(speeds, 0, 259), bins=bins, density=True, histtype="step", linewidth=1.4,
                color=PARTITION_COLORS[part],
                label=f"{PARTITION_LABELS[part]}: median {np.median(speeds):.0f} km/h")
    ax.set_xlabel("Smoothed speed (km/h); stationary points and raw jumps over 252 km/h removed")
    ax.set_ylabel("Density")
    ax.set_title("Vehicle speed distribution")
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_gps_route_distance(dist_by_partition: dict[str, np.ndarray], stats: pd.DataFrame,
                            save_path: Optional[str | Path] = None) -> plt.Figure:
    """How far GPS points lie from the reference route, pooled and per trip."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bins = np.arange(0, 302, 4)
    for part, dist in dist_by_partition.items():
        d = dist[np.isfinite(dist)]
        ax.hist(np.clip(d, 0, 299), bins=bins, histtype="step", linewidth=1.4, color=PARTITION_COLORS[part],
                label=f"{PARTITION_LABELS[part]}: median {np.median(d):.1f} m, 95th pct {np.percentile(d, 95):.0f} m")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from GPS point to nearest reference-route arc (m; 300+ binned at 300)")
    ax.set_ylabel("GPS points (log scale)")
    ax.set_title("GPS noise relative to the reference route")
    ax.legend(loc="upper right")

    ax = axes[1]
    sub = stats.sort_values("gps_route_dist_median_m", ascending=False).reset_index(drop=True)
    ax.bar(sub.index, sub["gps_route_dist_median_m"], color=[PARTITION_COLORS[p] for p in sub["partition"]],
           width=0.85)
    ax.axhline(50, color=GAP_COLOR, linestyle="--", linewidth=0.9)
    for i, row in sub[sub["gps_route_dist_median_m"] > 50].iterrows():
        ax.text(i + 0.8, row["gps_route_dist_median_m"], row["trip_id"], fontsize=7.5, va="center")
    ax.set_title("Median distance to the reference route, per trip")
    ax.set_ylabel("Median distance (m)")
    ax.set_xlabel("Trips, sorted")
    ax.set_xticks([])
    fig.tight_layout()
    return _save(fig, save_path)


def plot_route_vs_track(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """GPS points vs route arcs, and GPS path length vs route length, per trip."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for part in _partitions(stats):
        sub = stats[stats["partition"] == part]
        axes[0].scatter(sub["n_route_arcs"], sub["n_gps_points"], s=22, color=PARTITION_COLORS[part],
                        alpha=0.8, label=PARTITION_LABELS[part])
        axes[1].scatter(sub["route_length_km"], sub["path_length_km"], s=22, color=PARTITION_COLORS[part],
                        alpha=0.8, label=PARTITION_LABELS[part])
    lo = min(stats["n_gps_points"].min(), stats["n_route_arcs"].min()) * 0.7
    hi = max(stats["n_gps_points"].max(), stats["n_route_arcs"].max()) * 1.4
    axes[0].plot([lo, hi], [lo, hi], color="#7A8491", linewidth=0.9, linestyle="--", label="equal counts")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Arcs in the reference route")
    axes[0].set_ylabel("GPS points")
    axes[0].set_title("Every trip has more GPS points than route arcs")
    axes[0].legend()

    axes[1].scatter(stats["route_length_km"], stats["path_length_raw_km"], s=22, facecolor="none",
                    edgecolor="#7A8491", linewidth=0.9, label="Raw GPS path (jitter included)")
    lim = [0, max(stats["path_length_raw_km"].max(), stats["route_length_km"].max()) * 1.05]
    axes[1].plot(lim, lim, color="#7A8491", linewidth=0.9, linestyle="--", label="same length")
    ratio = stats["path_length_km"] / stats["route_length_km"].replace(0, np.nan)
    for _, row in stats[(ratio > 1.3) | (ratio < 0.77)].iterrows():
        axes[1].annotate(row["trip_id"], (row["route_length_km"], row["path_length_km"]), fontsize=7.5,
                         xytext=(4, 2), textcoords="offset points")
    axes[1].set_xlabel("Reference route length (km)")
    axes[1].set_ylabel("GPS path length (km)")
    axes[1].set_title("GPS path length vs route length (filled = smoothed, hollow = raw)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_network_connectivity(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> plt.Figure:
    """Largest strongly connected component share and two-way arc share per trip."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, col, title in [
        (axes[0], "largest_strong_share", "Nodes in the largest strongly connected component"),
        (axes[1], "two_way_arc_share", "Arcs whose reverse arc also exists (two-way roads)"),
    ]:
        sub = stats.sort_values(col).reset_index(drop=True)
        ax.bar(sub.index, sub[col], color=[PARTITION_COLORS[p] for p in sub["partition"]], width=0.85)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title)
        ax.set_xlabel("Trips, sorted")
        ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PARTITION_COLORS.values()]
    axes[0].legend(handles, PARTITION_LABELS.values(), loc="lower right")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_heading_smoothing(headings: pd.DataFrame, stats: pd.DataFrame,
                           save_path: Optional[str | Path] = None) -> plt.Figure:
    """Why speed and heading are smoothed: heading error per method, and raw jitter per trip."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(headings))
    colors = [ROUTE_COLOR if c else "#9AA3AF" for c in headings["chosen"]]
    bars = ax.barh(y, headings["share_wrong_direction"], color=colors)
    for bar, row in zip(bars, headings.itertuples()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{row.share_wrong_direction:.1%} wrong way · median error {row.median_error_deg:.1f}°",
                va="center", fontsize=7.8)
    ax.set_yticks(y, headings["method"])
    ax.invert_yaxis()
    ax.set_xlim(0, headings["share_wrong_direction"].max() * 1.75)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Training points whose heading is over 90° off the nearest route arc")
    ax.set_title("Heading error by method (training trips)")

    ax = axes[1]
    sub = stats.sort_values("jitter_reversal_share", ascending=False).reset_index(drop=True)
    ax.bar(sub.index, sub["jitter_reversal_share"], color=[PARTITION_COLORS[p] for p in sub["partition"]], width=0.85)
    ax.axhline(0.15, color=GAP_COLOR, linestyle="--", linewidth=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Raw steps that reverse direction (>150° turn in 1 s), per trip")
    ax.set_xlabel("Trips, sorted")
    ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PARTITION_COLORS.values()]
    ax.legend(handles, PARTITION_LABELS.values(), loc="upper right")
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Per-trip views
# ---------------------------------------------------------------------------

def _crop_arcs(trip, margin_m: float) -> np.ndarray:
    """Indices of arcs with at least one end inside the track's bounding box plus a margin."""
    ax_, ay, bx, by = trip.arc_xy
    t = trip.track
    x0, x1 = t["x_m"].min() - margin_m, t["x_m"].max() + margin_m
    y0, y1 = t["y_m"].min() - margin_m, t["y_m"].max() + margin_m
    inside_a = (ax_ >= x0) & (ax_ <= x1) & (ay >= y0) & (ay <= y1)
    inside_b = (bx >= x0) & (bx <= x1) & (by >= y0) & (by <= y1)
    return np.flatnonzero(inside_a | inside_b)


def _arcs_near_track(trip, radius_m: float) -> np.ndarray:
    """Indices of arcs whose start, middle or end lies within ``radius_m`` of a GPS point."""
    idx = _crop_arcs(trip, margin_m=radius_m)
    ax_, ay, bx, by = (a[idx] for a in trip.arc_xy)
    tree = cKDTree(np.column_stack([trip.track["x_m"].to_numpy(), trip.track["y_m"].to_numpy()]))
    near = np.zeros(idx.size, dtype=bool)
    for px, py in ((ax_, ay), (bx, by), ((ax_ + bx) / 2, (ay + by) / 2)):
        d, _ = tree.query(np.column_stack([px, py]), distance_upper_bound=radius_m)
        near |= np.isfinite(d)
    return idx[near]


def plot_trip_examples(trips: Sequence, captions: Sequence[str],
                       save_path: Optional[str | Path] = None) -> plt.Figure:
    """Up to four trips: road network (grey), reference route (green), GPS points (blue), gaps (red)."""
    _style()
    n = len(trips)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.6 * rows), squeeze=False)
    for ax, trip, caption in zip(axes.flat, trips, captions):
        x0, y0 = trip.track["x_m"].min(), trip.track["y_m"].min()
        ax_, ay, bx, by = trip.arc_xy
        idx = _crop_arcs(trip, margin_m=300.0)
        segs = np.stack([np.column_stack([ax_[idx] - x0, ay[idx] - y0]),
                         np.column_stack([bx[idx] - x0, by[idx] - y0])], axis=1) / 1000.0
        ax.add_collection(LineCollection(segs, colors=NETWORK_COLOR, linewidths=0.4, zorder=1))
        if trip.route is not None:
            r = trip.route["arc_id"].to_numpy()
            rsegs = np.stack([np.column_stack([ax_[r] - x0, ay[r] - y0]),
                              np.column_stack([bx[r] - x0, by[r] - y0])], axis=1) / 1000.0
            ax.add_collection(LineCollection(rsegs, colors=ROUTE_COLOR, linewidths=2.2, zorder=2))
        t = trip.track
        ax.scatter((t["x_m"] - x0) / 1000.0, (t["y_m"] - y0) / 1000.0, s=2.5, color=GPS_COLOR, zorder=3,
                   linewidths=0)
        gaps = t[t["is_large_gap"]]
        ax.scatter((gaps["x_m"] - x0) / 1000.0, (gaps["y_m"] - y0) / 1000.0, s=26, facecolor="none",
                   edgecolor=GAP_COLOR, linewidths=1.2, zorder=4)
        pad = 0.05 * max(t["x_m"].max() - x0, t["y_m"].max() - y0) / 1000.0
        ax.set_xlim(-pad, (t["x_m"].max() - x0) / 1000.0 + pad)
        ax.set_ylim(-pad, (t["y_m"].max() - y0) / 1000.0 + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
        ax.set_title(caption, fontsize=9.5)
        ax.grid(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=NETWORK_COLOR, linewidth=1.5, label="Road network"),
        plt.Line2D([], [], color=ROUTE_COLOR, linewidth=2.5, label="Reference route"),
        plt.Line2D([], [], marker="o", linestyle="", color=GPS_COLOR, markersize=4, label="GPS points"),
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=GAP_COLOR,
                   markersize=7, label="Point after a gap > 5 s"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Folium maps
# ---------------------------------------------------------------------------

def _legend_html(title: str, items: Sequence[tuple[str, str, str]]) -> str:
    """Fixed-position legend; items are (kind, colour, label) with kind 'line' or 'dot'."""
    rows = []
    for kind, color, label in items:
        if kind == "line":
            swatch = f'<span style="display:inline-block;width:22px;height:0;border-top:4px solid {color};' \
                     f'vertical-align:middle;margin-right:8px"></span>'
        else:
            swatch = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;' \
                     f'background:{color};vertical-align:middle;margin:0 14px 0 6px"></span>'
        rows.append(f"<div style='margin:3px 0'>{swatch}{label}</div>")
    return (
        "<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;"
        "padding:10px 14px;border:1px solid #c9ced6;border-radius:6px;font:12px/1.4 Arial,sans-serif;"
        f"box-shadow:0 1px 4px rgba(0,0,0,.15)'><b>{title}</b>{''.join(rows)}</div>"
    )


def make_trip_map(trip, save_path: Optional[str | Path] = None, max_points: int = 2500) -> folium.Map:
    """Interactive map of one trip: GPS points, nearby road network, reference route, gaps."""
    t = trip.track
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    lon, lat = trip.nodes["lon"].to_numpy().round(5), trip.nodes["lat"].to_numpy().round(5)
    u = trip.arcs["from_node_id"].to_numpy()
    v = trip.arcs["to_node_id"].to_numpy()

    idx = _arcs_near_track(trip, radius_m=150.0)
    network = [[[lat[u[i]], lon[u[i]]], [lat[v[i]], lon[v[i]]]] for i in idx]
    fg_net = folium.FeatureGroup(name=f"Road network within 150 m ({len(idx):,} arcs)")
    folium.PolyLine(network, color="#8C96A3", weight=1.2, opacity=0.7).add_to(fg_net)
    fg_net.add_to(m)

    if trip.route is not None:
        r = trip.route["arc_id"].to_numpy()
        route = [[[lat[u[a]], lon[u[a]]], [lat[v[a]], lon[v[a]]]] for a in r]
        fg_route = folium.FeatureGroup(name=f"Reference route ({len(r):,} arcs)")
        folium.PolyLine(route, color=ROUTE_COLOR, weight=5, opacity=0.85).add_to(fg_route)
        fg_route.add_to(m)

    step = max(1, int(np.ceil(len(t) / max_points)))
    shown = t.iloc[::step]
    label = f"GPS points ({len(t):,})" if step == 1 else f"GPS points (every {step}th of {len(t):,})"
    fg_gps = folium.FeatureGroup(name=label)
    for row in shown.itertuples():
        folium.CircleMarker([row.gps_lat, row.gps_lon], radius=2.2, color=GPS_COLOR, weight=0,
                            fill=True, fill_opacity=0.8,
                            tooltip=f"obs {row.obs_idx} · t={row.timestamp:.0f}s · {row.speed_mps * 3.6:.0f} km/h"
                            ).add_to(fg_gps)
    fg_gps.add_to(m)

    gaps = t[t["is_large_gap"]]
    fg_gap = folium.FeatureGroup(name=f"Points after a gap > 5 s ({len(gaps)})")
    for row in gaps.itertuples():
        folium.CircleMarker([row.gps_lat, row.gps_lon], radius=7, color=GAP_COLOR, weight=2, fill=False,
                            tooltip=f"obs {row.obs_idx}: {row.delta_t:.0f} s since previous point").add_to(fg_gap)
    fg_gap.add_to(m)

    first, last = t.iloc[0], t.iloc[-1]
    folium.Marker([first.gps_lat, first.gps_lon], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([last.gps_lat, last.gps_lon], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[t["gps_lat"].min(), t["gps_lon"].min()], [t["gps_lat"].max(), t["gps_lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    title = f"Trip {trip.trip_id} ({trip.partition} partition)"
    m.get_root().html.add_child(folium.Element(_legend_html(title, [
        ("dot", GPS_COLOR, "GPS observation"),
        ("line", ROUTE_COLOR, "Reference route"),
        ("line", "#8C96A3", "Road network"),
        ("dot", GAP_COLOR, "Point after a gap > 5 s"),
    ])))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
    return m


def make_locations_map(stats: pd.DataFrame, save_path: Optional[str | Path] = None) -> folium.Map:
    """World map with one marker per trip at its median GPS position."""
    m = folium.Map(location=[25, 10], zoom_start=2, tiles="OpenStreetMap", control_scale=True)
    for part in _partitions(stats):
        fg = folium.FeatureGroup(name=PARTITION_LABELS[part])
        for row in stats[stats["partition"] == part].itertuples():
            folium.CircleMarker(
                [row.centroid_lat, row.centroid_lon], radius=4 + np.sqrt(row.n_gps_points) / 18,
                color=PARTITION_COLORS[part], weight=1, fill=True, fill_opacity=0.7,
                tooltip=(f"Trip {row.trip_id} ({part}) · {row.n_gps_points:,} GPS points · "
                         f"{row.path_length_km:.1f} km · {row.crs}"),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html("Trip locations", [
        ("dot", PARTITION_COLORS[p], PARTITION_LABELS[p]) for p in _partitions(stats)
    ])))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
    return m
