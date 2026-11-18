# Section 3: road graph, metrics, HMM/Viterbi decoding and maps

| Part | File | What it does |
|---|---|---|
| 3.1 | `road_graph.py` | Directed road graph per trip, node coordinates in metres (trip UTM zone), arc lengths and directions, bounded and cached Dijkstra, shortest connecting arcs. |
| 3.1 | `metrics.py` | The shared metrics every report table uses: Top-1, macro F1 (candidate and observation level), route continuity, candidate miss rate, safe log loss, route-level precision/recall/F1. |
| 3.2 | `decoder.py` | `ViterbiRouteDecoder`: modes `nearest`, `classifier_only`, `distance_hmm` (alias `distance_baseline`) and `ml_hmm`; `HMMParams`; `load_tuned_decoders`. |
| 3.3 | `compare_hmm.py` | All decoding modes on one trip, same candidates and graph. |
| 3.3 | `folium_maps.py` | Interactive maps: nearby road network, reference route, classifier-only route, HMM route, GPS points, wrong points, legend and layer control. |
| 3.3 | `run_hmm_experiments.py` | Out-of-fold emissions, sigma/beta estimation, tuning, student-test comparison, tables, figures, maps and `configs/hmm_params.json`. |

## How to run

From the repository root, in order:

```bash
python -m src.lane_a_data.build_candidate_tables      # Section 1 tables -> data/processed/
python -m src.lane_b_models.train_section22           # Section 2 model -> artifacts/models/best_model.joblib
python -m src.lane_c_hmm.run_hmm_experiments          # Section 3 results -> results/, configs/hmm_params.json
python -m pytest tests/test_lane_c_hmm.py -q          # Section 3 tests
```

The API (`src/lane_d_app/pipeline.py`) loads `configs/hmm_params.json` at start-up and decodes
with the tuned settings. No evaluator-test file is read by any Section 3 script.

## Model in one paragraph

Hidden state at GPS fix *t* = which of its candidate arcs (at most 10 within 100 m) the vehicle is
on. Emission: `log P(candidate is right)` from the selected classifier (`ml_hmm`), or a Gaussian
on the GPS-to-arc distance (`distance_hmm`). Transition: `-|d_route - d_gc| / beta`, where `d_gc` is
the straight-line distance between consecutive fixes and `d_route` the distance along the road
network between the two candidate positions (Newson & Krumm, 2009). Arcs that cannot be reached
within `max_speed * dt + 200 m` get a finite "disconnected" score, so the decoder never stalls.
Gaps over 5 s widen the search and beta; gaps over 120 s restart the chain. Fixes without
candidates take the arc of the previous fix. sigma and beta are first estimated from training
reference paths, then beta and the disconnected penalty are tuned on the 80 training trips with
out-of-fold probabilities, and the result is evaluated once on the 10 student-test trips.
