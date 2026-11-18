"""Section 3.3: HMM / Viterbi experiments, decoder tuning, comparison tables, figures and maps.

Run from the repository root (after the Section 1 tables and the Section 2 model exist):

    python -m src.lane_c_hmm.run_hmm_experiments

Inputs
    data/processed/{train,student_test}_candidates.parquet   (src.lane_a_data.build_candidate_tables)
    artifacts/models/best_model.joblib                         (src.lane_b_models.train_section22)
    data/train, data/student_test raw trips                    (road network + reference route)

Protocol (no evaluator-test file is read)
    1. Emissions for the 80 training trips are *out-of-fold* probabilities: the selected
       model is re-fitted 5 times with GroupKFold by trip, and every trip is scored by the
       fit that never saw it. Student-test trips are scored by the exported model.
    2. sigma (GPS noise) and beta (transition scale) are first estimated from the training
       reference alignment (Newson & Krumm, 2009): sigma = 1.4826 * median distance of a fix
       to its reference arc, beta = median |d_route - d_gc| / ln 2 along reference paths.
    3. beta and the disconnected penalty are then tuned by grid search on the training trips
       (maximising observation-level Top-1). Every trial is logged.
    4. The tuned decoder is evaluated once on the 10 student-test trips and compared with
       the nearest-road and classifier-only predictions.

Outputs
    configs/hmm_params.json                         tuned decoder settings used by the API
    results/tables/hmm_*.csv, hmm_summary.md        tables for Report Section 5
    results/figures/hmm_*.png                       figures for Report Section 5
    results/maps/hmm_route_<trip>.html              Folium maps for Report Section 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.lane_a_data.ground_truth import assign_point_ground_truth
from src.lane_a_data.loaders import list_trip_dirs, load_trip
from src.lane_a_data.preprocessing import preprocess_track
from src.lane_b_models.export_model import load_model_bundle
from src.lane_c_hmm.decoder import HMMParams, MODES, ViterbiRouteDecoder, collapse_consecutive_duplicates
from src.lane_c_hmm.folium_maps import create_route_map
from src.lane_c_hmm.metrics import (
    compute_log_loss_safe,
    compute_macro_f1,
    continuity_flags,
    evaluate_predictions,
)
from src.lane_c_hmm.road_graph import RoadNetworkGraph

PARTITIONS = ("train", "student_test")  # evaluator_test is never read here
KEY = ["trip_id", "obs_idx", "candidate_arc_id"]
MODE_LABEL = {"ml_hmm": "Classifier + HMM", "classifier_only": "Classifier only",
              "distance_hmm": "Distance HMM", "nearest": "Nearest road"}
# Colour follows the decoder (same in every figure); validated categorical slots 1-4.
MODE_COLOUR = {"ml_hmm": "#2a78d6", "classifier_only": "#eb6834", "distance_hmm": "#1baf7a", "nearest": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BETA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PENALTIES = (0.0, 5.0, 15.0, 40.0, 100.0)
BACKWARD = (0.0, 1.0)


# ---------------------------------------------------------------------------- data
def load_probabilities(bundle: dict, train: pd.DataFrame, test: pd.DataFrame, cache: Path,
                       folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """pred_prob for every candidate row: out-of-fold on train, exported model on test."""
    features = bundle["feature_names"]
    name = bundle.get("metadata", {}).get("best_model_name", type(bundle["model"]).__name__)
    t0 = time.perf_counter()
    test = test.copy()
    test["pred_prob"] = bundle["model"].predict_proba(test[features].fillna(0.0).to_numpy())[:, 1]
    timing = {"student_test_scoring_sec": time.perf_counter() - t0}

    meta_path = cache.with_suffix(".json")
    if cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model") == name and meta.get("rows") == len(train) and meta.get("folds") == folds:
            oof = pd.read_parquet(cache)
            print(f"[oof] reusing cached out-of-fold probabilities ({name}, {len(oof):,} rows)")
            train = train.merge(oof, on=KEY, how="left", validate="one_to_one")
            timing["train_oof_sec"] = meta.get("seconds", float("nan"))
            return train, test, timing

    print(f"[oof] {folds}-fold GroupKFold out-of-fold probabilities with {name} ...")
    t0 = time.perf_counter()
    X = train[features].fillna(0.0).to_numpy()
    y = train["is_target"].to_numpy()
    groups = train["trip_id"].to_numpy()
    oof = np.full(len(train), np.nan)
    for k, (fit_idx, held_idx) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups), 1):
        fit_idx = fit_idx[np.isin(y[fit_idx], (0, 1))]  # unlabeled rows never enter training
        model = clone(bundle["model"]).fit(X[fit_idx], y[fit_idx])
        oof[held_idx] = model.predict_proba(X[held_idx])[:, 1]
        print(f"[oof] fold {k}/{folds} done ({time.perf_counter() - t0:.0f}s)")
    seconds = time.perf_counter() - t0
    out = train[KEY].copy()
    out["pred_prob"] = oof
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    meta_path.write_text(json.dumps({"model": name, "rows": len(train), "folds": folds, "seconds": seconds}))
    train = train.copy()
    train["pred_prob"] = oof
    timing["train_oof_sec"] = seconds
    return train, test, timing


def load_trips(partition: str, candidates: pd.DataFrame, decoder: ViterbiRouteDecoder) -> List[dict]:
    """Raw trip, reference labels, road graph and trellis for every trip of a partition."""
    trips = []
    by_trip = dict(tuple(candidates.groupby("trip_id", sort=False)))
    for d in list_trip_dirs(Path("data") / partition):
        if d.name not in by_trip:
            print(f"[{partition}] {d.name}: no candidate rows, skipped")
            continue
        raw = load_trip(d)
        track = preprocess_track(raw["track"])
        gt = assign_point_ground_truth(d.name, track, raw["route"], raw["nodes"], raw["arcs"])
        cands = by_trip[d.name]
        graph = RoadNetworkGraph(raw["nodes"], raw["arcs"])
        trellis = decoder.build_trellis(cands, graph, track["obs_idx"].to_numpy())
        trips.append({"trip_id": d.name, "partition": partition, "raw": raw, "track": track, "gt": gt,
                      "cands": cands, "graph": graph, "trellis": trellis})
    print(f"[{partition}] {len(trips)} trips prepared")
    return trips


# ---------------------------------------------------------------------------- estimation
def estimate_sigma_beta(trips: List[dict]) -> Dict[str, float]:
    """Newson & Krumm style estimates from the training reference alignment."""
    dists, gaps = [], []
    for trip in trips:
        tr, gt = trip["trellis"], trip["gt"].set_index("obs_idx")["true_arc_id"]
        truth = gt.reindex(tr.step_obs).to_numpy()
        true_state = np.full(len(tr.arcs), -1)
        for t, (arcs, ref) in enumerate(zip(tr.arcs, truth)):
            if pd.isna(ref):
                continue
            hit = np.flatnonzero(arcs == int(ref))
            if len(hit):
                true_state[t] = hit[0]
                dists.append(tr.dist[t][hit[0]])
        for t in range(1, len(tr.arcs)):
            i, j = true_state[t - 1], true_state[t]
            if i >= 0 and j >= 0 and tr.route_gap[t] is not None and tr.dt[t] <= 5.0:
                g = tr.route_gap[t][i, j]
                if np.isfinite(g):
                    gaps.append(g)
    dists, gaps = np.asarray(dists), np.asarray(gaps)
    return {"sigma_m": float(1.4826 * np.median(dists)), "beta_m": float(np.median(gaps) / np.log(2)),
            "n_points": int(len(dists)), "n_transitions": int(len(gaps)),
            "reference_gap_p95_m": float(np.quantile(gaps, 0.95))}


# ---------------------------------------------------------------------------- evaluation
def trip_metrics(trip: dict, preds: np.ndarray, mode: str, seconds: float) -> dict:
    gt, graph, cands = trip["gt"], trip["graph"], trip["cands"]
    m = evaluate_predictions(gt, preds, trip["raw"]["arcs"], cands if mode in ("classifier_only", "ml_hmm") else
                             cands.drop(columns=["pred_prob"], errors="ignore"),
                             inference_time_sec=seconds, graph=graph)
    labelled = gt["true_arc_id"].notna().to_numpy()
    truth = gt["true_arc_id"].fillna(-1).to_numpy(np.int64)
    flags = continuity_flags(preds, trip["raw"]["arcs"], graph, gt["timestamp"].to_numpy(float))
    m.update({"trip_id": trip["trip_id"], "mode": mode, "n_labelled": int(labelled.sum()),
              "n_correct": int(((preds == truth) & labelled).sum()), "n_transitions": int(len(flags)),
              "n_continuous": int(flags.sum()), "route_arcs": len(collapse_consecutive_duplicates(preds))})
    return m


def decode_partition(trips: List[dict], decoders: Dict[str, ViterbiRouteDecoder]) -> tuple[pd.DataFrame, Dict]:
    rows, preds = [], {}
    for trip in trips:
        for mode in MODES:
            dec = decoders.get(mode, decoders["ml_hmm"])
            t0 = time.perf_counter()
            out = dec.predict_arcs(trip["trellis"], mode)
            seconds = time.perf_counter() - t0 + (trip["trellis"].prep_time_sec if mode.endswith("hmm") else 0.0)
            p = np.asarray(out["point_predictions"], dtype=np.int64)
            preds[(trip["trip_id"], mode)] = p
            row = trip_metrics(trip, p, mode, seconds)
            row["disconnected_steps"] = out["disconnected_steps"]
            row["chain_restarts"] = out["chain_restarts"]
            rows.append(row)
    return pd.DataFrame(rows), preds


def summarise(per_trip: pd.DataFrame, trips: List[dict]) -> pd.DataFrame:
    """Pooled / averaged partition-level numbers per mode."""
    cand = pd.concat([t["cands"] for t in trips], ignore_index=True)
    labelled = cand[cand["is_target"].isin([0, 1])]
    logloss = compute_log_loss_safe(labelled["is_target"], labelled["pred_prob"])
    cand_f1 = compute_macro_f1(labelled["is_target"], (labelled["pred_prob"] >= 0.5).astype(int))
    best = cand.groupby(["trip_id", "obs_idx"])["is_target"].max()
    miss = float((best[best >= 0] == 0).mean())
    out = []
    for mode in MODES:
        g = per_trip[per_trip["mode"] == mode]
        prob_mode = mode in ("classifier_only", "ml_hmm")
        out.append({
            "mode": mode, "decoder": MODE_LABEL[mode], "trips": len(g),
            "top1_accuracy": g["n_correct"].sum() / g["n_labelled"].sum(),
            "macro_f1_obs_mean_per_trip": g["macro_f1"].mean(),
            "route_continuity_pct": 100.0 * g["n_continuous"].sum() / g["n_transitions"].sum(),
            "route_precision_mean": g["route_precision"].mean(),
            "route_recall_mean": g["route_recall"].mean(),
            "route_f1_mean": g["route_f1"].mean(),
            "candidate_macro_f1": cand_f1 if prob_mode else None,
            "log_loss": logloss if prob_mode else None,
            "candidate_miss_rate": miss,
            "disconnected_steps": int(g["disconnected_steps"].sum()),
            "decode_time_total_sec": g["inference_time_sec"].sum(),
            "decode_time_per_trip_sec": g["inference_time_sec"].mean(),
            "labelled_points": int(g["n_labelled"].sum()),
        })
    return pd.DataFrame(out)


def tune(trips: List[dict], mode: str, base: HMMParams, est: Dict[str, float]) -> tuple[HMMParams, pd.DataFrame]:
    """Grid search on training trips: beta x disconnected penalty x backward handling.

    Selection: highest observation Top-1, then adjacent-arc continuity, then the smaller beta.
    """
    rows = []
    for factor, penalty, backward in itertools.product(BETA_FACTORS, PENALTIES, BACKWARD):
        params = HMMParams(**{**base.to_dict(), "beta_m": est["beta_m"] * factor, "disconnected_penalty": penalty,
                              "backward_tolerant": backward})
        dec = ViterbiRouteDecoder(params)
        t0 = time.perf_counter()
        correct = labelled = cont = trans = 0
        for trip in trips:
            p = dec.predict_arcs(trip["trellis"], mode)["point_predictions"]
            truth = trip["gt"]["true_arc_id"]
            ok = truth.notna().to_numpy()
            correct += int((p[ok] == truth[ok].to_numpy(np.int64)).sum())
            labelled += int(ok.sum())
            flags = continuity_flags(p, trip["raw"]["arcs"])  # adjacency-only: fast, same for all trials
            cont += int(flags.sum())
            trans += len(flags)
        rows.append({"mode": mode, "beta_factor": factor, "beta_m": params.beta_m, "disconnected_penalty": penalty,
                     "backward_tolerant": backward, "sigma_m": params.sigma_m, "train_top1": correct / labelled,
                     "train_adjacent_continuity_pct": 100.0 * cont / trans, "seconds": time.perf_counter() - t0})
        print(f"[tune {mode}] beta {params.beta_m:6.2f} penalty {penalty:4.0f} backward_tolerant {backward:.0f} "
              f"-> top1 {correct / labelled:.4f}")
    trials = pd.DataFrame(rows)
    best = trials.sort_values(["train_top1", "train_adjacent_continuity_pct", "beta_factor"],
                              ascending=[False, False, True]).iloc[0]
    tuned = HMMParams(**{**base.to_dict(), "beta_m": float(best["beta_m"]),
                         "disconnected_penalty": float(best["disconnected_penalty"]),
                         "backward_tolerant": float(best["backward_tolerant"])})
    return tuned, trials


# ---------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(summary: pd.DataFrame, per_trip: pd.DataFrame, trials: pd.DataFrame,
                 change: pd.DataFrame, fig_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    order = ["nearest", "classifier_only", "distance_hmm", "ml_hmm"]
    paths = []

    # 1. Mode comparison on student test: three metrics, one panel each (no shared/dual axis).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    s = summary.set_index("mode").loc[order]
    panels = [("top1_accuracy", "Observation Top-1 accuracy", 100), ("route_continuity_pct", "Route continuity", 1),
              ("route_f1_mean", "Route-level F1 (mean per trip)", 100)]
    for ax, (col, title, scale) in zip(axes, panels):
        _style(ax)
        vals = s[col].to_numpy(float) * scale
        x = np.arange(len(order))
        ax.bar(x, vals, width=0.62, color=[MODE_COLOUR[m] for m in order], edgecolor=SURFACE, linewidth=2)
        decimals = 2 if col == "route_continuity_pct" else 1  # 99.97 must not print as 100.0
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.2, f"{v:.{decimals}f}%", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, [MODE_LABEL[m].replace(" + ", "\n+ ").replace(" ", "\n", 1) if len(MODE_LABEL[m]) > 12
                          else MODE_LABEL[m] for m in order], fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_ylabel("%", color=INK2, fontsize=9)
    fig.suptitle("Student-test trips (10): decoding strategies compared", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    paths.append(fig_dir / "hmm_mode_comparison.png")
    fig.savefig(paths[-1], dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # 2. Per-trip Top-1: classifier only vs classifier + HMM (dumbbell), distance HMM as reference dots.
    pt = per_trip.pivot(index="trip_id", columns="mode", values="top1_accuracy") * 100
    pt = pt.sort_values("ml_hmm")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pt) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(pt))
    ax.hlines(y, pt["classifier_only"], pt["ml_hmm"], color=GRID, linewidth=2, zorder=1)
    for mode, marker in (("distance_hmm", "D"), ("classifier_only", "o"), ("ml_hmm", "o")):
        ax.scatter(pt[mode], y, s=64 if mode != "distance_hmm" else 36, color=MODE_COLOUR[mode], marker=marker,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=MODE_LABEL[mode])
    for yi, v in zip(y, pt["ml_hmm"]):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, pt.index, fontsize=9)
    ax.set_xlabel("Observation Top-1 accuracy (%)", color=INK2, fontsize=9)
    lo = np.floor(np.nanmin(pt.to_numpy()) / 5) * 5
    ax.set_xlim(max(0, lo - 2), 102)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("Per student-test trip: effect of HMM decoding", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    paths.append(fig_dir / "hmm_per_trip_top1.png")
    fig.savefig(paths[-1], dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # 3. Tuning grid on training trips (sequential single-hue ramp).
    modes = [m for m in ("ml_hmm", "distance_hmm") if m in set(trials["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.4 * len(modes), 3.6), facecolor=SURFACE, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = trials[trials["mode"] == mode]
        backward = sub.loc[sub["train_top1"].idxmax(), "backward_tolerant"]  # show the slice holding the optimum
        grid = sub[sub["backward_tolerant"] == backward].pivot(
            index="disconnected_penalty", columns="beta_m", values="train_top1") * 100
        im = ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto", origin="lower")
        best = np.unravel_index(np.nanargmax(grid.to_numpy()), grid.shape)
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v > np.nanpercentile(grid.to_numpy(), 60) else INK,
                    fontweight="bold" if (i, j) == best else "normal")
        ax.set_xticks(range(grid.shape[1]), [f"{b:.1f}" for b in grid.columns], fontsize=8.5)
        ax.set_yticks(range(grid.shape[0]), [f"{p:.0f}" for p in grid.index], fontsize=8.5)
        ax.set_xlabel("beta (m)", color=INK2, fontsize=9)
        ax.set_ylabel("disconnected penalty", color=INK2, fontsize=9)
        ax.set_title(f"{MODE_LABEL[mode]}: training Top-1 (%), backward_tolerant = {backward:.0f}",
                     fontsize=10.5, color=INK, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=8, colors=INK2)
    fig.tight_layout()
    paths.append(fig_dir / "hmm_tuning_heatmap.png")
    fig.savefig(paths[-1], dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # 4. Points fixed vs broken by the HMM on each student-test trip (diverging around zero).
    c = change.set_index("trip_id").sort_values("net_gain")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(c) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(c))
    ax.barh(y, c["fixed_by_hmm"], color=MODE_COLOUR["ml_hmm"], height=0.6, label="Fixed by HMM (wrong -> right)")
    ax.barh(y, -c["broken_by_hmm"], color=MODE_COLOUR["classifier_only"], height=0.6,
            label="Broken by HMM (right -> wrong)")
    ax.axvline(0, color=INK2, linewidth=1)
    for yi, (f, b) in enumerate(zip(c["fixed_by_hmm"], c["broken_by_hmm"])):
        ax.text(f + 3, yi, str(int(f)), va="center", fontsize=8, color=INK2)
        ax.text(-b - 3, yi, str(int(b)), va="center", ha="right", fontsize=8, color=INK2)
    span = max(c["fixed_by_hmm"].max(), c["broken_by_hmm"].max(), 1) * 1.2
    ax.set_xlim(-span, span)
    ax.set_yticks(y, c.index, fontsize=9)
    ax.set_xlabel("GPS points", color=INK2, fontsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("Where the HMM changed the classifier's answer (student test)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    paths.append(fig_dir / "hmm_fixed_vs_broken.png")
    fig.savefig(paths[-1], dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model", default="artifacts/models/best_model.joblib")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--maps-dir", default="results/maps")
    parser.add_argument("--params-out", default="configs/hmm_params.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    t_start = time.perf_counter()
    tables, figures, maps = Path(args.tables_dir), Path(args.figures_dir), Path(args.maps_dir)
    for d in (tables, figures, maps):
        d.mkdir(parents=True, exist_ok=True)

    processed = Path(args.processed_dir)
    train = pd.read_parquet(processed / "train_candidates.parquet")
    test = pd.read_parquet(processed / "student_test_candidates.parquet")
    bundle = load_model_bundle(args.model)
    model_name = bundle.get("metadata", {}).get("best_model_name", type(bundle["model"]).__name__)
    print(f"[model] {model_name} with {len(bundle['feature_names'])} features")
    train, test, timing = load_probabilities(bundle, train, test, processed / "train_oof_probs.parquet", args.folds)

    base = HMMParams()
    prep = ViterbiRouteDecoder(base)
    train_trips = load_trips("train", train, prep)
    test_trips = load_trips("student_test", test, prep)

    est = estimate_sigma_beta(train_trips)
    print(f"[estimate] sigma {est['sigma_m']:.2f} m, beta {est['beta_m']:.2f} m "
          f"({est['n_points']:,} points, {est['n_transitions']:,} transitions)")
    base = HMMParams(**{**base.to_dict(), "sigma_m": est["sigma_m"], "beta_m": est["beta_m"]})

    tuned_ml, trials_ml = tune(train_trips, "ml_hmm", base, est)
    tuned_dist, trials_dist = tune(train_trips, "distance_hmm", base, est)
    trials = pd.concat([trials_ml, trials_dist], ignore_index=True)
    trials.to_csv(tables / "hmm_tuning_trials.csv", index=False)
    decoders = {"ml_hmm": ViterbiRouteDecoder(tuned_ml), "distance_hmm": ViterbiRouteDecoder(tuned_dist),
                "classifier_only": ViterbiRouteDecoder(tuned_ml), "nearest": ViterbiRouteDecoder(tuned_ml)}

    params_doc = {
        "ml_hmm": tuned_ml.to_dict(),
        "distance_hmm": tuned_dist.to_dict(),
        "provenance": {
            "model": model_name,
            "selected_on": "80 training trips, out-of-fold classifier probabilities (GroupKFold by trip)",
            "objective": "observation-level Top-1 accuracy (ties: adjacent-arc continuity)",
            "estimated_sigma_m": est["sigma_m"], "estimated_beta_m": est["beta_m"],
            "beta_factors": list(BETA_FACTORS), "penalties": list(PENALTIES), "backward_tolerant": list(BACKWARD),
            "evaluator_test_used": False,
        },
    }
    Path(args.params_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params_out).write_text(json.dumps(params_doc, indent=2))
    print(f"[params] wrote {args.params_out}")

    results = {}
    for name, trips in (("train_oof", train_trips), ("student_test", test_trips)):
        per_trip, preds = decode_partition(trips, decoders)
        summary = summarise(per_trip, trips)
        per_trip.to_csv(tables / f"hmm_per_trip_{name}.csv", index=False)
        summary.to_csv(tables / f"hmm_comparison_{name}.csv", index=False)
        results[name] = (per_trip, summary, preds, trips)
        print(f"\n[{name}]\n" + summary[["decoder", "top1_accuracy", "macro_f1_obs_mean_per_trip",
                                      "route_continuity_pct", "route_f1_mean"]].to_string(index=False))

    # Where did the HMM help or hurt? (student test)
    per_trip, summary, preds, trips = results["student_test"]
    change_rows = []
    for trip in trips:
        truth = trip["gt"]["true_arc_id"]
        ok = truth.notna().to_numpy()
        t = truth.fillna(-1).to_numpy(np.int64)
        c, h = preds[(trip["trip_id"], "classifier_only")], preds[(trip["trip_id"], "ml_hmm")]
        change_rows.append({"trip_id": trip["trip_id"], "labelled_points": int(ok.sum()),
                            "fixed_by_hmm": int((ok & (c != t) & (h == t)).sum()),
                            "broken_by_hmm": int((ok & (c == t) & (h != t)).sum()),
                            "wrong_both": int((ok & (c != t) & (h != t)).sum()),
                            "reference_not_in_candidates": int(
                                (trip["cands"].groupby("obs_idx")["is_target"].max() == 0).sum())})
    change = pd.DataFrame(change_rows)
    change["net_gain"] = change["fixed_by_hmm"] - change["broken_by_hmm"]
    change.to_csv(tables / "hmm_change_analysis_student_test.csv", index=False)

    fig_paths = make_figures(summary, per_trip, trials, change, figures)

    # Folium maps for every student-test trip (report Section 6).
    for trip in trips:
        truth = trip["gt"]["true_arc_id"]
        hmm = preds[(trip["trip_id"], "ml_hmm")]
        wrong = truth.notna().to_numpy() & (hmm != truth.fillna(-1).to_numpy(np.int64))
        graph = trip["graph"]
        fmap = create_route_map(
            trip["track"], trip["raw"]["nodes"], trip["raw"]["arcs"],
            predicted_arcs=graph.connect_route(collapse_consecutive_duplicates(hmm)),
            reference_route=trip["raw"]["route"]["arc_id"].tolist(),
            classifier_arcs=collapse_consecutive_duplicates(preds[(trip["trip_id"], "classifier_only")]),
            wrong_mask=wrong,
            title=f"Trip {trip['trip_id']} (student test)",
        )
        fmap.save(str(maps / f"hmm_route_{trip['trip_id']}.html"))

    tr_sum = results["train_oof"][1].set_index("mode")
    te_sum = summary.set_index("mode")
    lines = [
        "# Section 3.3: HMM / Viterbi results (generated by run_hmm_experiments)",
        "",
        f"Model: {model_name}. Decoder settings tuned on the 80 training trips with out-of-fold probabilities; "
        "evaluator-test trips were not read.",
        "",
        f"Estimated from training reference paths: sigma = {est['sigma_m']:.2f} m, beta = {est['beta_m']:.2f} m.",
        f"Tuned ml_hmm: beta = {tuned_ml.beta_m:.2f} m, disconnected penalty = {tuned_ml.disconnected_penalty:g}, "
        f"backward_tolerant = {tuned_ml.backward_tolerant:g}. "
        f"Tuned distance_hmm: beta = {tuned_dist.beta_m:.2f} m, disconnected penalty = {tuned_dist.disconnected_penalty:g}, "
        f"backward_tolerant = {tuned_dist.backward_tolerant:g}.",
        "",
        "| Decoder | Train Top-1 (OOF) | Student Top-1 | Student macro F1 (obs) | Student continuity | Student route F1 | Decode s/trip |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in ("nearest", "classifier_only", "distance_hmm", "ml_hmm"):
        a, b = tr_sum.loc[mode], te_sum.loc[mode]
        lines.append(f"| {MODE_LABEL[mode]} | {a.top1_accuracy:.4f} | {b.top1_accuracy:.4f} | "
                     f"{b.macro_f1_obs_mean_per_trip:.4f} | {b.route_continuity_pct:.2f}% | {b.route_f1_mean:.4f} | "
                     f"{b.decode_time_per_trip_sec:.2f} |")
    lines += ["", f"Candidate miss rate (student test): {te_sum.loc['ml_hmm', 'candidate_miss_rate']:.4f}; "
              f"classifier log loss (student test): {te_sum.loc['ml_hmm', 'log_loss']:.4f}.",
              f"HMM vs classifier-only on student test: {int(change.fixed_by_hmm.sum())} points fixed, "
              f"{int(change.broken_by_hmm.sum())} points broken, {int(change.wrong_both.sum())} wrong in both.",
              "", "Figures: " + ", ".join(p.name for p in fig_paths), f"Maps: {maps}/hmm_route_<trip>.html",
              f"Total run time: {time.perf_counter() - t_start:.0f} s"]
    (tables / "hmm_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
