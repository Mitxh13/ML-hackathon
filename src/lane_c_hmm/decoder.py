"""HMM + Viterbi route decoder (Section 3.2).

Hidden states
    At GPS observation t the hidden state is *which candidate arc the vehicle is on*. The
    states of t are exactly the candidate rows of t (at most k = 10 arcs within 100 m), so
    every state keeps its candidate-to-arc correspondence.

Emission score (log scale, per state)
    ``ml_hmm``        w * log(clip(pred_prob))  - the selected classifier's probability that
                      this candidate is the right arc (w = 1; only the ratio to the
                      transition scale matters, so w is not tuned separately).
    ``distance_hmm``  -0.5 * (dist_to_arc_m / sigma_m)^2 - a Gaussian GPS-error model
                      (the classifier-free baseline of Newson & Krumm, 2009).

Transition score (log scale, per pair of states at t-1 and t)
    The car's road distance between the two candidate positions should be close to the
    straight-line distance between the two GPS fixes:
        d_gc    = straight-line distance between GPS fixes t-1 and t (metres)
        d_route = distance along the road network from the projection of fix t-1 on arc i
                  to the projection of fix t on arc j (rest of arc i + shortest path + part
                  of arc j). On the same arc it is the signed distance between the two
                  projections, so moving backwards on a one-way arc is penalised, unless
                  ``backward_tolerant`` = 1 treats it as GPS noise (chosen by tuning).
        log T   = -|d_route - d_gc| / beta_t
    Road paths are searched only up to ``max_speed_mps * dt + search_slack_m`` metres (capped
    at ``max_search_m``). If arc j cannot be reached from arc i inside that bound, the pair is
    *disconnected*: log T = -limit / beta_t - disconnected_penalty. The penalty is finite, so
    the decoder never gets stuck when the data really jump (e.g. a GPS glitch or a missing
    road); it simply prefers any connected explanation.

GPS gaps
    dt <= large_gap_s (5 s): normal transitions. dt > large_gap_s: the search bound grows
    with dt and beta_t = beta * dt / large_gap_s, because the car could have taken more varied
    paths. dt > break_gap_s: the chain is restarted (log T = 0 for every pair).

Missing candidates
    An observation without candidate rows is skipped by the trellis and receives the arc
    decoded for the nearest previous (else next) observation; the output flags it.

Viterbi recurrence
    V[0, j] = E[0, j];  V[t, j] = E[t, j] + max_i (V[t-1, i] + log T[t, i, j]);
    the best final state is traced back through the stored arg-max pointers.

Modes: ``nearest`` (closest candidate), ``classifier_only`` (arg-max pred_prob per point),
``distance_hmm`` (alias ``distance_baseline``) and ``ml_hmm``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.contracts.schemas import DecoderOutput
from src.lane_c_hmm.road_graph import RoadNetworkGraph

MODE_ALIASES = {"distance_baseline": "distance_hmm"}
MODES = ("nearest", "classifier_only", "distance_hmm", "ml_hmm")
PROB_EPS = 1e-6


def collapse_consecutive_duplicates(seq: Sequence[int]) -> List[int]:
    """[10, 10, 15, 15, 22] -> [10, 15, 22]."""
    out: List[int] = []
    for item in seq:
        item = int(item)
        if not out or out[-1] != item:
            out.append(item)
    return out


@dataclass
class HMMParams:
    """Decoder settings. Defaults are replaced by the values tuned on training trips
    (``configs/hmm_params.json``, written by ``run_hmm_experiments``)."""

    sigma_m: float = 5.0
    beta_m: float = 5.0
    emission_weight: float = 1.0
    disconnected_penalty: float = 10.0
    max_speed_mps: float = 50.0
    search_slack_m: float = 200.0
    max_search_m: float = 3000.0
    large_gap_s: float = 5.0
    break_gap_s: float = 120.0
    backward_tolerant: float = 0.0  # 1: backwards movement on the same arc is GPS noise (|progress|)

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "HMMParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in (values or {}).items() if k in known})

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Trellis:
    """Everything the Viterbi pass needs for one trip, independent of the tunable scores."""

    trip_id: str
    obs_idx: np.ndarray                      # observations in time order (all of them)
    step_obs: np.ndarray                     # observations that have candidates (trellis steps)
    arcs: List[np.ndarray]                   # per step: candidate arc ids (K_t,)
    dist: List[np.ndarray]                   # per step: GPS-to-arc distance (K_t,)
    prob: List[Optional[np.ndarray]]         # per step: classifier probability or None
    dt: np.ndarray                           # per step: seconds since previous step
    route_gap: List[Optional[np.ndarray]]    # per step t>=1: |d_route - d_gc| (K_{t-1}, K_t), inf = unreachable
    same_arc: List[Optional[tuple]]          # per step: (i, j, |abs progress - d_gc|) for same-arc pairs
    limit: np.ndarray                        # per step: road-search bound used (metres)
    prep_time_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class ViterbiRouteDecoder:
    """Map-matching decoder. ``decode_trip`` keeps the signature used by the API pipeline."""

    def __init__(self, params: Optional[HMMParams] = None, **overrides: float) -> None:
        base = params or HMMParams()
        legacy = {"beta_transition_m": "beta_m", "sigma_emission_m": "sigma_m",
                  "large_gap_threshold_s": "large_gap_s"}
        values = base.to_dict()
        for key, val in overrides.items():
            key = legacy.get(key, key)
            if key == "unconnected_penalty":
                key, val = "disconnected_penalty", abs(float(val))
            if key in values:
                values[key] = float(val)
        self.params = HMMParams(**values)

    # ------------------------------------------------------------------ trellis
    def build_trellis(self, candidates_df: pd.DataFrame, graph: RoadNetworkGraph,
                      obs_index: Optional[Sequence[int]] = None) -> Trellis:
        """Group candidates by observation and pre-compute all road distances."""
        t0 = time.perf_counter()
        p = self.params
        trip_id = str(candidates_df["trip_id"].iloc[0]) if len(candidates_df) else "unknown"
        df = candidates_df.sort_values(["obs_idx", "dist_to_arc_m", "candidate_arc_id"], kind="stable")
        obs = df["obs_idx"].to_numpy(np.int64)
        all_obs = np.unique(np.asarray(obs_index, dtype=np.int64)) if obs_index is not None else np.unique(obs)
        step_obs, starts = np.unique(obs, return_index=True)
        ends = np.r_[starts[1:], len(obs)]

        arc_all = df["candidate_arc_id"].to_numpy(np.int64)
        dist_all = df["dist_to_arc_m"].to_numpy(float)
        prob_all = df["pred_prob"].to_numpy(float) if "pred_prob" in df.columns else None
        ts = df["timestamp"].to_numpy(float)[starts]
        lon = df["gps_lon"].to_numpy(float)[starts]
        lat = df["gps_lat"].to_numpy(float)[starts]
        pts = graph.project(lon, lat)

        # position of each GPS fix along each of its candidate arcs (metres from the arc start)
        row_step = np.repeat(np.arange(len(starts)), ends - starts)
        _, along_all = graph.project_onto_arcs(pts[row_step], arc_all) if len(arc_all) else (None, np.zeros(0))
        arcs = [arc_all[s:e] for s, e in zip(starts, ends)]
        dists = [dist_all[s:e] for s, e in zip(starts, ends)]
        probs = [prob_all[s:e] if prob_all is not None else None for s, e in zip(starts, ends)]
        pos = [along_all[s:e] for s, e in zip(starts, ends)]

        n = len(step_obs)
        dt = np.r_[0.0, np.diff(ts)] if n else np.zeros(0)
        gc = np.r_[0.0, np.hypot(*np.diff(pts, axis=0).T)] if n else np.zeros(0)
        limit = np.minimum(p.max_speed_mps * np.maximum(dt, 1.0) + p.search_slack_m, p.max_search_m)
        frm, to, length = graph.arc_from, graph.arc_to, graph.arc_length_m
        route_gap: List[Optional[np.ndarray]] = [None]
        same_arc: List[Optional[tuple]] = [None]
        for t in range(1, n):
            a_prev, a_cur = arcs[t - 1], arcs[t]
            if dt[t] > p.break_gap_s:
                route_gap.append(None)
                same_arc.append(None)
                continue
            rest = length[a_prev] - pos[t - 1]
            d_route = np.full((len(a_prev), len(a_cur)), np.inf)
            cur_from = frm[a_cur]
            for i, ai in enumerate(a_prev):
                reach = graph.node_distances(int(to[ai]), limit[t])
                sp = np.fromiter((reach.get(int(u), np.inf) for u in cur_from), dtype=float, count=len(a_cur))
                sp[sp > limit[t]] = np.inf  # same answer whatever an earlier, wider search cached
                d_route[i] = rest[i] + sp + pos[t]
                same = a_cur == ai
                if same.any():
                    # signed progress: going backwards on a one-way arc counts against the pair
                    # (|negative progress - d_gc| = backwards metres + GPS metres), so a fix that
                    # moves forward cannot sit on the reverse-direction twin for free
                    d_route[i, same] = pos[t][same] - pos[t - 1][i]
            route_gap.append(np.abs(d_route - gc[t]))
            ii, jj = np.nonzero(a_prev[:, None] == a_cur[None, :])
            same_arc.append((ii, jj, np.abs(np.abs(d_route[ii, jj]) - gc[t])) if len(ii) else None)
        return Trellis(trip_id=trip_id, obs_idx=all_obs, step_obs=step_obs, arcs=arcs, dist=dists,
                       prob=probs, dt=dt, route_gap=route_gap, same_arc=same_arc, limit=limit,
                       prep_time_sec=time.perf_counter() - t0)

    # ------------------------------------------------------------------ scores
    def _emission(self, trellis: Trellis, t: int, mode: str) -> np.ndarray:
        p = self.params
        if mode == "distance_hmm":
            return -0.5 * (trellis.dist[t] / p.sigma_m) ** 2
        prob = trellis.prob[t]
        return p.emission_weight * np.log(np.clip(prob, PROB_EPS, 1.0))

    def _transition(self, trellis: Trellis, t: int) -> Optional[np.ndarray]:
        gap = trellis.route_gap[t]
        if gap is None:
            return None  # chain restart after a very long gap
        p = self.params
        if p.backward_tolerant and trellis.same_arc[t] is not None:
            ii, jj, tolerant = trellis.same_arc[t]
            gap = gap.copy()
            gap[ii, jj] = tolerant
        beta = p.beta_m * max(1.0, trellis.dt[t] / p.large_gap_s)
        disconnected = -trellis.limit[t] / beta - p.disconnected_penalty
        return np.where(np.isfinite(gap), -gap / beta, disconnected)

    # ------------------------------------------------------------------ decoding
    def decode_states(self, trellis: Trellis, mode: str) -> Dict[str, Any]:
        """Return the chosen candidate index per step plus diagnostics."""
        mode = MODE_ALIASES.get(mode, mode)
        if mode not in MODES:
            raise ValueError(f"Unknown decoder mode '{mode}'. Use one of {MODES} or 'distance_baseline'.")
        n = len(trellis.arcs)
        if mode in ("classifier_only", "ml_hmm") and n and trellis.prob[0] is None:
            raise ValueError(f"Mode '{mode}' needs model probabilities (column 'pred_prob').")
        if n == 0:
            return {"states": np.zeros(0, dtype=np.int64), "disconnected_steps": 0, "chain_restarts": 0}
        if mode == "nearest":
            return {"states": np.array([int(np.argmin(d)) for d in trellis.dist]),
                    "disconnected_steps": 0, "chain_restarts": 0}
        if mode == "classifier_only":
            return {"states": np.array([int(np.argmax(pr)) for pr in trellis.prob]),
                    "disconnected_steps": 0, "chain_restarts": 0}

        score = self._emission(trellis, 0, mode)
        back: List[Optional[np.ndarray]] = [None]
        restarts = 0
        for t in range(1, n):
            trans = self._transition(trellis, t)
            emit = self._emission(trellis, t, mode)
            if trans is None:
                restarts += 1
                back.append(np.full(len(emit), int(np.argmax(score))))
                score = float(score.max()) + emit
                continue
            total = score[:, None] + trans
            best = np.argmax(total, axis=0)
            back.append(best)
            score = total[best, np.arange(total.shape[1])] + emit
        states = np.empty(n, dtype=np.int64)
        states[-1] = int(np.argmax(score))
        for t in range(n - 1, 0, -1):
            states[t - 1] = back[t][states[t]]
        disconnected = sum(
            1 for t in range(1, n)
            if trellis.route_gap[t] is not None and not np.isfinite(trellis.route_gap[t][states[t - 1], states[t]])
        )
        return {"states": states, "disconnected_steps": disconnected, "chain_restarts": restarts}

    def predict_arcs(self, trellis: Trellis, mode: str) -> Dict[str, Any]:
        """Decoded arc for every observation of the trip (gaps filled from neighbours)."""
        out = self.decode_states(trellis, mode)
        step_arcs = np.array([trellis.arcs[t][s] for t, s in enumerate(out["states"])], dtype=np.int64)
        all_obs = trellis.obs_idx
        if len(step_arcs) == 0:
            out.update(point_predictions=np.full(len(all_obs), -1, dtype=np.int64), filled=len(all_obs))
            return out
        pos = np.searchsorted(trellis.step_obs, all_obs, side="right") - 1
        exact = (pos >= 0) & (trellis.step_obs[np.clip(pos, 0, None)] == all_obs)
        pos = np.where(pos < 0, 0, pos)  # observations before the first step take the first arc
        out["point_predictions"] = step_arcs[pos]
        out["filled"] = int((~exact).sum())
        return out

    def decode_trip(
        self,
        candidates_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        mode: str = "ml_hmm",
        graph: Optional[RoadNetworkGraph] = None,
        obs_index: Optional[Sequence[int]] = None,
    ) -> DecoderOutput:
        """Decode one trip and return Contract 4 output.

        ``point_predictions[i]`` is the arc for the i-th observation in ``obs_index``
        (default: the sorted ``obs_idx`` values present in ``candidates_df``).
        """
        start = time.perf_counter()
        canonical = MODE_ALIASES.get(mode, mode)
        if candidates_df is None or candidates_df.empty:
            n = 0 if obs_index is None else len(obs_index)
            return DecoderOutput(trip_id="unknown", mode=canonical, point_predictions=[-1] * n,
                                 reconstructed_route=[], inference_time_sec=0.0)
        if graph is None:
            graph = RoadNetworkGraph(nodes_df, arcs_df)
        trellis = self.build_trellis(candidates_df, graph, obs_index)
        result = self.predict_arcs(trellis, canonical)
        preds = [int(a) for a in result["point_predictions"]]
        route = collapse_consecutive_duplicates([a for a in preds if a >= 0])
        elapsed = time.perf_counter() - start
        return DecoderOutput(
            trip_id=trellis.trip_id,
            mode=canonical,
            point_predictions=preds,
            reconstructed_route=route,
            inference_time_sec=elapsed,
            obs_idx=[int(o) for o in trellis.obs_idx],
            connected_route=graph.connect_route(route),
            disconnected_steps=int(result["disconnected_steps"]),
            chain_restarts=int(result["chain_restarts"]),
            observations_without_candidates=int(result["filled"]),
            hmm_params=self.params.to_dict(),
        )


def load_tuned_decoders(path: str = "configs/hmm_params.json") -> Dict[str, ViterbiRouteDecoder]:
    """Decoder per mode from the tuned settings file (defaults when the file is missing)."""
    import json
    from pathlib import Path

    doc = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    ml = HMMParams.from_dict(doc.get("ml_hmm"))
    dist = HMMParams.from_dict(doc.get("distance_hmm")) if "distance_hmm" in doc else ml
    return {"ml_hmm": ViterbiRouteDecoder(ml), "classifier_only": ViterbiRouteDecoder(ml),
            "nearest": ViterbiRouteDecoder(ml), "distance_hmm": ViterbiRouteDecoder(dist)}
