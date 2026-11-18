# PathSnap Development Guidelines & Agent Rules
# Distributed Map-Matching & Route Reconstruction

This document specifies the architecture, data contracts, technical definitions, anti-leakage policies, lane ownership, environment rules, and Git branching workflows governing all development in the PathSnap repository.

---

## 1. System Architecture & Challenge Context

### Objective
PathSnap reconstructs realistic vehicle routes by matching noisy $\approx 1\text{ Hz}$ GPS trajectories to a road network using:
1. Spatial candidate generation (KD-Tree metric radius search).
2. Supervised machine learning classification (scoring candidate road segments).
3. Hidden Markov Model (HMM) + Viterbi decoding (integrating ML emission probabilities with road-network topological transition constraints).

### Two-Laptop Deployment Architecture
PathSnap is designed and demonstrated as a distributed two-node system:
- **Laptop 1 (Model & API Server)**:
  - Hosts the FastAPI backend (`src/lane_d_app/api_server.py`).
  - Stores trained model weights (`artifacts/models/best_model.joblib`), scalers, and road network topology (`.nodes`, `.arcs`).
  - Runs the HMM/Viterbi decoder and performs server-side evaluation against internal ground-truth `.route` files.
- **Laptop 2 (Evaluator UI)**:
  - Hosts the Streamlit evaluator dashboard (`src/lane_d_app/ui_client.py`).
  - Displays the 10 reserved evaluation trips for selection.
  - Sends the selected `trip_id` over HTTP/REST to Laptop 1.
  - Receives and renders the interactive Folium route map and performance metrics.

---

## 2. Raw Dataset Specifications

Each trip folder contains four whitespace-separated text files without headers:
- `.track`: 3 columns (`longitude`, `latitude`, `timestamp` in seconds). Nominally $1\text{ Hz}$ GPS trajectory ($\approx 97\%$ of steps are $1\text{ s}$ apart).
- `.nodes`: 2 columns (`longitude`, `latitude`). The 0-based row index corresponds directly to `node_id`.
- `.arcs`: 2 integer columns (`from_node_id`, `to_node_id`). The 0-based row index corresponds directly to `arc_id`.
- `.route`: 1 integer column (`arc_id`). Chronological sequence of ground-truth road segments traversed by the vehicle.

### Geographical Rules & CRS Projection
- Longitude and latitude are angular coordinates on the WGS84 ellipsoid.
- **NEVER** compute raw Euclidean distances in degrees ($\sqrt{\Delta \text{lon}^2 + \Delta \text{lat}^2}$).
- Always project coordinates to a local metric CRS (e.g., UTM / Web Mercator in meters via `pyproj`) or compute geodesic distances using the Haversine formula in meters.
- **Coordinate Order Convention**:
  - Raw `.track` and `.nodes` files store `[longitude, latitude]`.
  - Folium and Leaflet maps expect `[latitude, longitude]`.
  - Always check coordinate order carefully to avoid swapped axis bugs.

---

## 3. Up-Front Shared Technical Definitions

### Point-to-Segment Ground Truth Assignment (`.route` $\rightarrow$ `.track`)
- `.route` is a deduplicated chronological sequence of traversed road segments (e.g., 525 segments), whereas `.track` contains raw GPS observations (e.g., 1,315 points).
- To assign each GPS observation $t$ its `true_arc_id`, Lane A aligns the sequence of GPS points monotonically along the chronological `.route` arc sequence by minimizing point-to-segment metric distance while respecting the forward order along `.route`.

### Shared Evaluation Metrics (`src/lane_c_hmm/metrics.py`)
All evaluation tables across Report Section 3 (Model Comparison), Section 4 (Bayesian Tuning), and Section 5 (HMM Reconstruction) **MUST** compute metrics exclusively through `src/lane_c_hmm/metrics.py`:
1. **Observation-level Top-1 Accuracy**:
   $$\text{Top-1} = \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{I}(\hat{a}_t = a_t^*)$$
   Fraction of GPS observations where the predicted segment $\hat{a}_t$ matches the ground-truth segment $a_t^*$.
2. **Macro F1-Score**:
   - Binary candidate-level Macro F1: unweighted average of F1 for class 0 (non-target) and class 1 (target) across candidate pairs.
   - Observation-level multi-class Macro F1: macro-averaged F1 across active `arc_id` classes per trip.
   - Both definitions must be explicitly logged to maintain total consistency.
3. **Route Continuity**:
   Percentage of consecutive observation pairs $(t-1, t)$ where the predicted road segments $(\hat{a}_{t-1}, \hat{a}_t)$ are either:
   - Identical ($\hat{a}_{t-1} = \hat{a}_t$).
   - Directly connected in `.arcs` ($\text{to\_node}(\hat{a}_{t-1}) = \text{from\_node}(\hat{a}_t)$).
   - Topologically connected via a short path in the road graph within the elapsed time step.
4. **Candidate Miss Rate**:
   Fraction of GPS observations where the ground-truth `true_arc_id` is absent from the top-$k$ candidate set.
5. **Safe Log Loss**:
   Binary cross-entropy computed on calibrated candidate probabilities `pred_prob` with safe probability clipping ($[\epsilon, 1-\epsilon]$). Reported as N/A for non-probabilistic models.
6. **Inference Time**:
   Wall-clock execution time in seconds per trip or partition.

---

## 4. Strict Data Partition & Anti-Leakage Policy

### Partitions
- `data/train/` (80 trips):
  - Used for preprocessing fitting, candidate feature engineering, training $\ge 5$ model families, trip-grouped cross-validation (`GroupKFold` grouped by `trip_id`), and Bayesian hyperparameter optimization.
- `data/student_test/` (10 trips):
  - Used for development-stage model evaluation, cross-method comparison tables, and HMM decoder validation.
- `data/evaluator_test/` (10 trips):
  - **UNTOUCHED during development.** Reserved strictly for final demonstration and grading via the Two-Laptop demo.

### Mandatory Integrity Rules
1. **No Group Leakage**: Never split observations from the same trip across train and validation folds. Always group by `trip_id`.
2. **No Test Peeking**: Never use `evaluator_test` for feature engineering, model training, hyperparameter search, or threshold tuning.
3. **Server-Side Isolation**: Evaluator ground-truth `.route` files remain strictly server-side on Laptop 1 for metric calculation and must **NEVER** enter the prediction pipeline or be sent to Laptop 2 UI.
4. **Strict Honesty**: Hardcoded predictions, fabricated metrics, or artificial training curves for non-iterative models are strictly prohibited.

---

## 5. Lane Ownership & Parallel Work Rules

To eliminate file collisions and enable concurrent development:
- **Lane A (`src/lane_a_data/`)**:
  - Owns file loaders (`loaders.py`), preprocessing and projection (`preprocessing.py`), point-to-segment ground truth (`ground_truth.py`), candidate generation (`candidates.py`), feature extraction (`features.py`), EDA charts (`eda_plots.py`), and Report Section 2.
- **Lane B (`src/lane_b_models/`)**:
  - Owns $\ge 5$ model family implementations (`train_models.py`), trip-grouped cross-validation, Optuna Bayesian hyperparameter search with full trial history logging (`bayesian_opt.py`), model evaluation and ROC/confusion plots (`evaluate_models.py`), artifact export (`export_model.py`), and Report Sections 3 & 4.
- **Lane C (`src/lane_c_hmm/`)**:
  - Owns standardized metrics (`metrics.py`), road network graph (`road_graph.py`), distance baseline & ML+HMM Viterbi decoder (`decoder.py`), decoder comparison (`compare_hmm.py`), interactive Folium maps (`folium_maps.py`), and Report Sections 5 & 6.
- **Lane D (`src/lane_d_app/`)**:
  - Owns end-to-end pipeline (`pipeline.py`), FastAPI server (`api_server.py`), Evaluator Streamlit UI (`ui_client.py`), two-laptop networking setup, `README.md`, final PDF assembly, and Report Section 7.

*Rule*: Do not modify code outside your assigned `src/lane_*` directory without prior coordination on `dev/integration`.

---

## 6. Python Virtual Environment (.venv) Rules

- Always activate and use the dedicated project environment located at `.venv/`.
- On Linux/macOS: `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest`.
- On Windows: `.venv\Scripts\python.exe`, `.venv\Scripts\pip.exe`, `.venv\Scripts\pytest.exe`.
- **Never install packages globally.** Whenever a new dependency is introduced, add it immediately to `requirements.txt`.

---

## 7. Git Workflow & Branching Rules

### Protected Branches
- `main`: Production-ready, fully integrated code only. Never commit directly to `main`.
- `dev/integration`: Staging branch where Lanes A, B, C, and D merge at designated join checkpoints.

### Mandatory Lane Branches
- Lane A: `feat/lane-a-data-features`
- Lane B: `feat/lane-b-models-tuning`
- Lane C: `feat/lane-c-hmm-viterbi`
- Lane D: `feat/lane-d-api-ui`

### Three Integration Checkpoints
- **Join 1**: Lane A merges candidate table generator into `dev/integration` so Lane B and Lane C can run on real candidate DataFrames.
- **Join 2**: Lane B merges trained model artifacts and out-of-fold probability predictions into `dev/integration` so Lane C can execute full ML+HMM decoding.
- **Join 3**: Lanes A, B, and C merge into `dev/integration` so Lane D replaces the mock pipeline with the full production pipeline.

### Commit Convention
Use conventional commits:
- `feat(lane-a): add spatial k-d tree candidate search`
- `feat(lane-b): train xgboost and lightgbm with groupkfold`
- `feat(lane-c): implement viterbi trellis with gap handling`
- `feat(lane-d): build two-laptop fastapi evaluation endpoint`
- `fix(lane-c): handle disconnected component penalties in road graph`
- `chore: update requirements and schema validation`
