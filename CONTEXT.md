# PATHSNAP: COMPLETE PROJECT CONTEXT & TECHNICAL REFERENCE

**Project**: PathSnap — Distributed Map-Matching & Route Reconstruction  
**Domain**: Applied Machine Learning & Spatial Route Reconstruction  
**Evaluation Milestone Note**: There is a first-phase evaluation before the lunch break, followed by the final Two-Laptop evaluation and viva.

---

## 1. Challenge Overview & Core Problem

GPS tracking systems in urban environments frequently suffer from signal interference, urban canyons (tall buildings), multipath reflections, GPS dropouts, and low or irregular sampling frequencies. As a result, raw GPS observations often appear to:
* Drift onto parallel streets,
* Jump across buildings or intersections, or
* Deviate significantly from the actual road travelled by the vehicle.

**PathSnap Objective**: Build an end-to-end distributed machine learning and sequential route-reconstruction system that matches noisy GPS trajectories to a road network, reconstructs realistic continuous vehicle routes, evaluates predictions against reference routes, and runs across a Two-Laptop deployment architecture (Laptop 1: FastAPI Model & Decoder Server; Laptop 2: Evaluator User Interface).

---

## 2. Raw Dataset Architecture & Geographical Rules

The dataset is organized by **trip** (e.g., `00000001`). Each trip folder contains up to four whitespace-separated raw text files without headers and using $0$-based indexing:

| File Extension | Columns (No Headers, Whitespace-Separated) | Indexing & Meaning |
| :--- | :--- | :--- |
| `.track` | `longitude latitude timestamp` ($3$ float columns) | Chronological GPS observations recorded by the vehicle. Timestamp is in seconds. Nominally $1\text{ Hz}$ ($\approx 97\%$ of consecutive steps have $\Delta t = 1\text{ s}$). Row index $t \in \{0, \dots, T-1\}$ is `obs_idx`. |
| `.nodes` | `longitude latitude` ($2$ float columns) | Geographical vertices in the road network. The $0$-based row index is `node_id`. |
| `.arcs` | `from_node_id to_node_id` ($2$ integer columns) | Directed road segments connecting `from_node_id` to `to_node_id`. The $0$-based row index is `arc_id`. |
| `.route` | `arc_id` ($1$ integer column) | Reference route: chronological sequence of ground-truth `arc_id` values traversed by the vehicle. |

### Critical Data Realities & Geographical Considerations
1. **Row Count Mismatch (`.track` vs. `.route`)**:
   * A row in `.track` does **not** correspond $1$-to-$1$ with a row in `.route`.
   * `.track` records GPS observations at $\approx 1\text{ Hz}$ (for example, Trip `00000001` has $1,315$ GPS points), whereas `.route` lists the deduplicated sequence of road segments in the order driven (for example, $525$ route segments). A vehicle typically records multiple GPS points while traversing a single road arc, or may skip short arcs during a GPS gap.
2. **Coordinate System & Distance Calculations**:
   * Longitude and latitude are **angular coordinates** in degrees.
   * **Never** use ordinary Euclidean distance in degrees ($\sqrt{\Delta \text{lon}^2 + \Delta \text{lat}^2}$), as $1^\circ$ of longitude varies with latitude ($\cos(\text{lat})$).
   * Always project coordinates into an appropriate metric Coordinate Reference System (e.g., local UTM zone via `pyproj` or local tangent-plane equirectangular meters) or use geodesic/Haversine formulas in meters.
   * **Coordinate Order Warning**: Raw `.track` and `.nodes` files use `(longitude, latitude)` order $(x, y)$, whereas `folium` expects `[latitude, longitude]` order.
3. **Data Quality & Reference Route Limitations**:
   * Raw trajectories may contain GPS noise, irregular timestamp gaps ($\Delta t > 1\text{ s}$), stationary points, or disconnected road-network components.
   * Supplied `.route` files are reference routes for the task and should not be assumed to be error-free physical ground truth. Teams must document any anomalies or limitations discovered during exploratory analysis.

---

## 3. Dataset Partitions & Strict Anti-Leakage Rules

The dataset of $100$ total trips is divided into three strict partitions:

| Partition | Trip Count | Permitted Usage |
| :--- | :--- | :--- |
| **Training** (`data/train/`) | $80$ trips | Exploratory analysis, fitting preprocessing transformers/scalers, feature engineering, training $\ge 5$ models, trip-grouped cross-validation (`GroupKFold`), and Bayesian hyperparameter optimization. |
| **Student Testing** (`data/student_test/`) | $10$ trips | Development-stage evaluation, comparing the $\ge 5$ trained models, evaluating untuned vs. Bayesian-tuned models, and tuning/evaluating HMM + Viterbi decoding against classifier-only baselines. |
| **Evaluator Testing** (`data/evaluator_test/`) | $10$ trips | **Final evaluation only** during the live Two-Laptop demonstration. |

### Mandatory Evaluation Integrity & Anti-Leakage Rules
* **Trip-Level Grouping**: Observations from the same trip must never be split across training and validation folds. Always use `GroupKFold(n_splits=...)` with `groups=trip_id`.
* **Untouched Evaluator Set**: The $10$ evaluator-testing trips must remain completely untouched during development. They must not be used for training, feature selection, model selection, hyperparameter optimization, or HMM parameter tuning.
* **Server-Side Ground Truth Isolation**: During final evaluation, the evaluator ground-truth `.route` labels must remain strictly on Laptop 1 (inside the server-side evaluation module) and must **never** enter the student prediction pipeline or be sent to Laptop 2 UI as an input.
* **No Hardcoding**: Hardcoded predictions, lookup tables, or manual route adjustments for specific trips are strictly prohibited and repositories will be inspected for them.

---

## 4. Team Division of Labor (4-Lane Architecture)

To ensure all four team members work in parallel without blocking one another and join cleanly at the end:

| Lane | Owner Responsibilities | Code Directory | Report Sections |
| :--- | :--- | :--- | :--- |
| **Lane A: Data & Features** | File loaders for `.track`, `.nodes`, `.arcs`, `.route`; Exploratory Data Analysis (plots, data-quality checks, initial Folium map); assigning a reference `true_arc_id` to each GPS point; generating candidate road segments near each GPS point; feature extraction. | `src/lane_a_data/` | Section 2 (§2.2) |
| **Lane B: Models & Tuning** | Training and comparing at least $5$ distinct models across different algorithm families; `GroupKFold` cross-validation grouped by trip; model comparison plots and confusion matrices; Bayesian hyperparameter optimization of the chosen model; exporting final model weights. | `src/lane_b_models/` | Sections 3 & 4 (§3, §4) |
| **Lane C: HMM, Viterbi & Maps** | Building the directed road graph; shared evaluation metrics module (`metrics.py`); route decoding (Phase 1 distance-only baseline Viterbi + Phase 2 ML-emission HMM/Viterbi); comparing Classifier-only vs. Classifier + HMM predictions; interactive Folium route maps. | `src/lane_c_hmm/` | Sections 5 & 6 (§5, §6) |
| **Lane D: API, UI & Packaging** | End-to-end prediction pipeline integration; FastAPI server on Laptop 1; Evaluator UI on Laptop 2; Two-Laptop networking setup; `README.md`; compiling the final consolidated PDF report. | `src/lane_d_app/` | Section 7 (§7) + Full PDF Assembly |

### Parallel Execution Strategy ("Nobody Waits on Anyone")
1. **Hour 1 Alignment**: Lock in the exact column schemas (`src/contracts/schemas.py`) and the two foundational definitions (Point-to-Segment Ground Truth Assignment and Macro F1 calculation).
2. **Immediate Unblocking**:
   * **Lane A** produces a fast initial candidate table (using point-to-segment distance + basic geometric features) so Lanes B and C have real data within the first hour, then iterates on richer features.
   * **Lane C** builds `metrics.py`, `road_graph.py`, and a **distance-only Viterbi decoder** first. This requires zero ML models from Lane B and provides a strong baseline for Report Section 5.
   * **Lane D** builds the FastAPI server and Evaluator UI around a mock prediction step that returns valid schema payloads, then swaps in the real pipeline once Lanes A, B, and C merge.
3. **Three Clean Integration Joins**:
   * **Join 1**: Lane A's candidate tables pass to Lane B (for model training) and Lane C (for distance-based HMM decoding).
   * **Join 2**: Lane B's exported model and predicted candidate probabilities pass to Lane C's HMM/Viterbi decoder.
   * **Join 3**: Lane A + B + C's unified pipeline plugs into Lane D's FastAPI endpoint on Laptop 1.

---

## 5. Up-Front Shared Definitions & Handoff Contracts

### Definition 1: Point-to-Segment Ground Truth Assignment (`.route` $\rightarrow$ `.track`)
Because `.route` is an ordered sequence of $M$ traversed road segments $(r_0, r_1, \dots, r_{M-1})$ and `.track` is a sequence of $T$ GPS points $(p_0, p_1, \dots, p_{T-1})$ with $T > M$:
* Each GPS point $p_t$ is assigned its ground-truth segment `true_arc_id` by aligning `.track` monotonically along the sequence of arcs in `.route`.
* Specifically, using dynamic programming (sequence alignment / monotonic projection), each GPS observation $p_t$ is matched to a route index $m(t) \in \{0, \dots, M-1\}$ such that $m(0) = 0$, $m(t) \ge m(t-1)$ (allowing stays on the same arc or forward steps along `.route`), minimizing total perpendicular metric distance $\sum_{t=0}^{T-1} d_{\perp}(p_t, r_{m(t)})$.
* Then $\text{true\_arc\_id}(t) = r_{m(t)}$.

### Definition 2: Shared Evaluation Metrics (`src/lane_c_hmm/metrics.py`)
All tables in Report Sections 3, 4, and 5 must call `src/lane_c_hmm/metrics.py` so every number is computed identically:

1. **Observation-Level Top-1 Accuracy**:
   $$\text{Top-1 Accuracy} = \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{I}\left(\hat{a}_t = a_t^{\text{true}}\right)$$
   where $\hat{a}_t$ is the predicted `arc_id` for GPS point $t$ (for classifier-only: the candidate $k$ with highest `pred_prob` at observation $t$; for HMM: the Viterbi-decoded state at observation $t$).
2. **Macro F1-Score**:
   * **Candidate-Level Binary Macro F1** (used during candidate classification evaluation): Unweighted average of $F_1(\text{class } 0)$ and $F_1(\text{class } 1)$ across all `(obs_idx, candidate_arc_id)` rows.
   * **Observation-Level Route Macro F1** (used when comparing point-level decoded `arc_id` predictions): Unweighted mean of per-segment $F_1$ scores across the unique ground-truth and predicted `arc_id` classes present in the evaluated trips:
     $$\text{Macro } F_1 = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \frac{2 \cdot \text{TP}_a}{2 \cdot \text{TP}_a + \text{FP}_a + \text{FN}_a}$$
   * *Note*: `metrics.py` computes and logs both so tables remain $100\%$ consistent between Classifier-only and HMM-decoded outputs.
3. **Route Continuity**:
   Percentage of consecutive GPS observation transitions $(t-1 \rightarrow t)$ where the predicted road segments $(\hat{a}_{t-1}, \hat{a}_t)$ form a topologically valid continuation in the road network:
   $$\text{Route Continuity} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{I}\left(\hat{a}_{t-1} = \hat{a}_t \;\lor\; \text{to\_node}(\hat{a}_{t-1}) = \text{from\_node}(\hat{a}_t) \;\lor\; d_{\text{graph}}(\hat{a}_{t-1}, \hat{a}_t) \le d_{\max}(\Delta t_t)\right)$$
4. **Candidate Miss Rate**:
   Fraction of GPS observations $t$ where $a_t^{\text{true}}$ is not present in the generated candidate set $\mathcal{C}_t$ (establishes the theoretical upper bound on Top-1 Accuracy).
5. **Log Loss**:
   Binary cross-entropy loss over candidate probability estimates $p_{t,k} = P(y_{t,k}=1 \mid x_{t,k})$:
   $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
   (Reported as `N/A` with explanation if a model/decoder does not output calibrated candidate probabilities).
6. **Inference Time**:
   Wall-clock runtime in seconds required to generate predictions for the test partition (or per trip).

### Handoff Data Contracts (`src/contracts/schemas.py`)
* **Contract A $\rightarrow$ B & C (Candidate Table)**:
  `[trip_id, obs_idx, timestamp, gps_lon, gps_lat, candidate_arc_id, from_node_id, to_node_id, dist_to_arc_m, is_target, feat_*]`
* **Contract B $\rightarrow$ C (Scored Candidate Table)**:
  All Candidate Table columns plus `[pred_prob, pred_label]`.
* **Contract C $\rightarrow$ D (Decoder Output Dictionary)**:
  `{"trip_id": str, "mode": str, "point_predictions": list[int], "reconstructed_route": list[int], "metrics": dict, "inference_time_sec": float, "folium_map_html": str}`

---

## 6. Stage-by-Stage Technical Requirements

### Stage 1: Exploratory Data Analysis & Feature Engineering (Lane A — Report §2.2)
* **Dataset Inventory**: Trip counts per partition ($80 / 10 / 10$), GPS observation counts per trip, road node/arc counts, file sizes, and missing/invalid value checks.
* **Anomaly & Gap Detection**: Distribution of sampling intervals $\Delta t = t_i - t_{i-1}$, identification of trips with large GPS dropouts ($\Delta t > 5\text{ s}$), isolated nodes, and disconnected graph components.
* **Candidate Generation**: For each GPS point $p_t$, use a spatial index (`scipy.spatial.cKDTree` or `shapely.STRtree` on projected metric coordinates) to select the top-$K$ nearest road segments within radius $R$ meters (e.g., $K = 8\text{--}10$, $R = 100\text{ m}$).
* **Engineered Features (`feat_*`)**:
  * Perpendicular distance from GPS point to candidate segment ($d_{\perp}$ in meters).
  * Distance from GPS point to segment start node (`from_node`) and end node (`to_node`).
  * Normalized projection position along the segment ($\alpha \in [0, 1]$).
  * Heading alignment: absolute angular difference $|\theta_{\text{gps}} - \theta_{\text{arc}}|$ and $\cos(\theta_{\text{gps}} - \theta_{\text{arc}})$ between GPS movement vector and directed road segment bearing.
  * Instantaneous vehicle speed ($v_t = \Delta d_{\text{gps}} / \Delta t$) and acceleration.
  * Segment length ($L_{\text{arc}}$) and node degree (in-degree / out-degree of `from_node` and `to_node` at intersections).
  * Rank and relative distance ratio ($d_{\perp, k} / d_{\perp, \min}$) among candidates at timestamp $t$.
  * Temporal context: distance/connectivity to nearest candidates at $t-1$ and $t+1$.

### Stage 2: Model Experimentation & Selection (Lane B — Report §3)
* **Minimum 5 Models Requirement**: Must train, evaluate, and document **at least $5$ distinct models** chosen across different algorithm families:
  1. *Linear / Probabilistic*: Logistic Regression, Ridge Classifier, SGD Classifier, Gaussian Naive Bayes.
  2. *Instance-Based*: KNN, Distance-Weighted KNN, Radius Neighbors.
  3. *Tree-Based*: Decision Tree, Extra Trees.
  4. *Ensemble / Bagging*: Random Forest, Extra Trees Classifier, Bagging Classifier.
  5. *Boosting*: LightGBM, XGBoost, CatBoost, HistGradientBoosting, Gradient Boosting, AdaBoost.
  6. *Kernel-Based*: Linear SVM, RBF SVM (with probability calibration if needed).
  7. *Neural Networks*: Shallow MLP, Deep MLP (`MLPClassifier` or PyTorch MLP with epoch-wise loss history).
  8. *Unsupervised / Clustering*: K-Means, Mini-Batch K-Means, DBSCAN (if used, must explicitly map clusters to supervised candidate probabilities and never treat raw cluster IDs as ground-truth labels).
* **Compulsory Table (§3.1)**: Consolidated comparison table with columns:
  `Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time`
* **Compulsory Per-Model Questions (§3.2)**: Every model trained must have its own subsection answering:
  1. Why was this algorithm considered suitable for PathSnap?
  2. What are its major strengths and limitations for this task?
  3. What were the important hyperparameters and their initial values?
  4. What loss function or optimization objective was used?
  5. What was the model's performance on the student-testing dataset (Top-1 Accuracy, Macro F1-score, Log Loss)?
  6. Did you observe overfitting or underfitting, and what experimental evidence supports this conclusion?
* **Compulsory Model Visualisations (§3.3)**:
  * Cross-validation Macro F1 comparison plot across all models.
  * Student-testing Top-1 Accuracy and Macro F1 comparison plots.
  * Log Loss comparison plot across applicable models.
  * Confusion matrices (diagnostic visualizations).
  * Epoch-wise training vs. validation loss curves **only** for models that naturally produce epoch histories (e.g., MLP / boosting rounds)—**never** fabricate artificial training curves for algorithms that do not train iteratively.

### Stage 3: Bayesian Hyperparameter Optimization (Lane B — Report §4)
* Select the best-performing model from Stage 2 based on CV Macro F1, probability quality (Log Loss), Route Continuity, and inference speed.
* Perform Bayesian hyperparameter optimization (e.g., using `optuna` Tree-structured Parzen Estimator `TPESampler`) using trip-grouped cross-validation (`GroupKFold`) on the $80$ training trips.
* Retain and commit the complete trial search log (`artifacts/logs/bayesian_trials.csv`).
* Answer all $6$ mandatory questions in Report §4 and include a direct comparison table of **Untuned vs. Tuned Model** across CV Macro F1, Student-Test Top-1 Accuracy, Student-Test Macro F1, Log Loss, and Inference Time.

### Stage 4: HMM & Viterbi Route Reconstruction (Lane C — Report §5)
Independent pointwise classification often predicts disconnected road segments at intersections or parallel streets. Lane C resolves this with a Hidden Markov Model (HMM) decoded via the Viterbi algorithm:
1. **Hidden States ($\mathcal{S}_t$)**: At each GPS observation $t$, the hidden states are the candidate road segments $\mathcal{C}_t = \{a_{t,1}, a_{t,2}, \dots, a_{t,K_t}\}$ generated near $p_t$.
2. **Emission Probabilities ($B_t(a_{t,k})$)**:
   * *Phase 1 (Distance-Only Baseline)*: Gaussian radial likelihood based on point-to-segment distance $d_{\perp}(p_t, a_{t,k})$:
     $$\log P(p_t \mid a_{t,k}) = -\frac{d_{\perp}(p_t, a_{t,k})^2}{2\sigma_e^2}$$
   * *Phase 2 (ML Classifier HMM)*: Uses the tuned classifier's calibrated probability output $p_{t,k} = P(\text{is\_target}=1 \mid x_{t,k})$ converted to log-likelihood:
     $$\log B_t(a_{t,k}) = \log\left(\max(p_{t,k}, \epsilon)\right)$$
3. **Transition Probabilities ($A(a_{t-1, i} \rightarrow a_{t, j})$)**:
   * Governed by road-network topology in `road_graph.py` and the discrepancy between GPS step distance $d_{\text{gps}}(p_{t-1}, p_t)$ and shortest road-network path distance $d_{\text{net}}(a_{t-1, i}, a_{t, j})$:
     $$\log P(a_{t,j} \mid a_{t-1,i}) = -\frac{\left| d_{\text{gps}}(p_{t-1}, p_t) - d_{\text{net}}(a_{t-1,i}, a_{t,j}) \right|}{\beta \cdot \max(\Delta t_t, 1.0)} - \lambda_{\text{turn}}$$
   * Same segment ($a_{t-1,i} = a_{t,j}$) or directly connected segments ($\text{to\_node}(a_{t-1,i}) = \text{from\_node}(a_{t,j})$) receive high transition probability.
   * Disconnected segments (unreachable within reasonable driving speed $v_{\max} \cdot \Delta t_t$) receive a fallback penalty $\log P_{\text{unconnected}}$ so the chain does not crash on minor graph imperfections.
4. **Viterbi Recurrence (in Log-Space to Prevent Underflow)**:
   $$V_t(j) = \log B_t(a_{t,j}) + \max_{i \in \mathcal{C}_{t-1}} \left[ V_{t-1}(i) + \log A(a_{t-1,i} \rightarrow a_{t,j}) \right]$$
   with backpointers $\text{bp}_t(j) = \arg\max_{i} [V_{t-1}(i) + \log A(a_{t-1,i} \rightarrow a_{t,j})]$ traced backward from $\hat{k}_{T-1} = \arg\max_j V_{T-1}(j)$.
5. **Handling Edge Cases**:
   * *Missing Candidates*: If $\mathcal{C}_t = \emptyset$, expand search radius dynamically or carry forward the previous state $\mathcal{C}_{t-1}$.
   * *Large GPS Gaps ($\Delta t_t > \tau_{\text{gap}}$)*: Scale transition tolerance $\beta \cdot \Delta t_t$ proportionally to elapsed time, or reset transition penalties if $\Delta t_t$ exceeds a break threshold (e.g., $30\text{ s}$) and bridge intermediate arcs via shortest path on `road_graph`.

### Stage 5: Geographical Visualisation Using Folium (Lane C — Report §6)
Folium maps are mandatory in both the repository (`results/maps/*.html`) and the PDF report (high-resolution screenshots). Every route map must include:
1. **Original GPS Observations**: Plotted as distinguishable markers/dots (e.g., blue circle markers).
2. **Relevant Road-Network Geometry**: Road segments rendered from `.nodes` and `.arcs` (e.g., light grey/slate polylines).
3. **Reference Route**: Ground-truth `.route` highlighted clearly (e.g., solid green polyline).
4. **Predicted Route**: Classifier-only and/or HMM-reconstructed route in a contrasting colour (e.g., dashed red/orange polyline).
5. **Map Usability**: Built-in HTML legend, appropriate auto-fitted bounding box (`fit_bounds`), and interactive `folium.LayerControl()` to toggle layers on and off.

### Stage 6: Two-Laptop Deployment Architecture (Lane D — Report §7)
During the final evaluation, the system must operate live across two physical laptops over a shared local network (LAN / Wi-Fi hotspot):

| Machine | Component | Responsibilities |
| :--- | :--- | :--- |
| **Laptop 1** | **FastAPI Backend (`api_server.py`)** | Loads the exported ML model, preprocessing pipeline, HMM/Viterbi decoder, and evaluator trip files. Exposes endpoints (e.g., `GET /trips`, `POST /predict/{trip_id}`). Runs the complete prediction pipeline on the requested trip, computes accuracy/metrics against server-side ground truth, logs the request to the terminal (CMD), and returns the predicted route, metrics, and Folium HTML map. |
| **Laptop 2** | **Evaluator UI (`ui_client.py`)** | Displays the dropdown/selector for the $10$ reserved evaluator-testing trips. Sends the evaluator's selected `trip_id` over HTTP to Laptop 1's IP address (`http://<LAPTOP_1_IP>:8000/predict/{trip_id}`). Renders the returned evaluation metrics, reconstructed route summary, and interactive Folium map. |

**Critical Demo Rule**: Students must show the evaluator the live API terminal logs on Laptop 1 receiving and processing the HTTP requests from Laptop 2 for any of the $10$ reserved evaluation trips.

---

## 7. Final Deliverables & Report Structure Checklist

### Deliverable 1: Public GitHub Repository
Must contain all $11$ required components in clearly named folders:
1. Data processing code (`src/lane_a_data/`)
2. Feature engineering code (`src/lane_a_data/features.py`)
3. Model training scripts for all $\ge 5$ models (`src/lane_b_models/train_models.py`)
4. Model evaluation, metrics, plots, and confusion matrices (`src/lane_b_models/evaluate_models.py`, `src/lane_c_hmm/metrics.py`)
5. Bayesian hyperparameter optimization code & trial logs (`src/lane_b_models/bayesian_opt.py`, `artifacts/logs/`)
6. HMM and Viterbi decoding implementation (`src/lane_c_hmm/road_graph.py`, `src/lane_c_hmm/decoder.py`)
7. FastAPI backend code (`src/lane_d_app/api_server.py`)
8. Evaluator UI code (`src/lane_d_app/ui_client.py`)
9. Deployment configuration, `requirements.txt`, and `README.md`
10. Folium map-generation scripts and exported `.html` maps (`src/lane_c_hmm/folium_maps.py`, `results/maps/`)
11. Supporting CSV tables, experiment logs, and saved model weights (`results/tables/`, `artifacts/models/`)

### Deliverable 2: Consolidated Technical Report (PDF)
Must strictly follow the required numbering and sections:
* **Section 2.2 — Dataset Understanding and Exploratory Data Analysis**: Preprocessing steps, observed patterns, feature extraction justification, missing-value analysis, GPS trajectory plots, road-network visualizations, at least one Folium map, and explicit mention of which dataset partition was used.
* **Section 3 — Model Experimentation and Comparison**:
  * **§3.1 Overall Model Comparison**: Consolidated table of $\ge 5$ models across all required metrics.
  * **§3.2 Compulsory Questions for EVERY Model**: Separate subsection per model answering all questions (suitability, strengths/limitations, hyperparameters, loss function, student-test metrics, overfitting/underfitting evidence).
  * **§3.3 Model Visualisations**: CV Macro F1 plot, Student-Test Top-1 & Macro F1 plots, Log Loss plot, confusion matrices, and valid training curves.
* **Section 4 — Bayesian Hyperparameter Optimisation**: Search space, distributions, objective function, trial count, best parameters, and Untuned vs. Tuned comparison table.
* **Section 5 — HMM and Viterbi Route Reconstruction**: Answers to all $6$ questions (hidden states, emission probabilities, transition probabilities, missing candidates/large gaps, disconnected segments, and experimental proof of improvement comparing Classifier-only vs. Classifier + HMM).
* **Section 6 — Geographical Visualisation Using Folium**: Screenshots of interactive Folium maps showing original GPS, road geometry, reference route, and predicted route with legends and layer controls.
* **Section 7 — API, UI and Deployment**: Screenshots of the working Evaluator UI on Laptop 2 and screenshots of successful API responses in the terminal/CMD of Laptop 1.

---

## 8. Viva Voce, Academic Integrity & Plagiarism Rules

* **Viva Readiness**: Every single team member must be able to explain and defend all parts of the system during evaluation: dataset structure, preprocessing, feature engineering, model comparisons, Bayesian optimization, HMM + Viterbi math/code, FastAPI backend, UI, and experimental metrics.
* **Strict Independent Work**: Sharing code, engineered datasets, model weights, hyperparameter configs, or experimental results across teams is strictly prohibited. Submissions are checked across all teams and sections for code, weight, configuration, and report similarity.
* **AI Tool Policy**:
  * Never submit AI-generated text claiming experiments that were not actually run.
  * Never fabricate metrics, graphs, citations, or artificial training curves.
  * All numbers and plots in the report must be directly reproducible from the scripts and logs committed to the GitHub repository.

---

## 9. Development Environment (`.venv`) & Git Workflow Summary

* **Virtual Environment**: Always execute Python and tests via the project virtual environment (`.venv/bin/python`, `.venv/bin/pytest`). Keep `requirements.txt` updated whenever adding packages.
* **Git Branching**:
  * `main`: Protected release branch (never commit directly to `main`).
  * `dev/integration`: Shared integration branch for Join 1, Join 2, and Join 3.
  * `feat/lane-a-data-features`: Lane A working branch.
  * `feat/lane-b-models-tuning`: Lane B working branch.
  * `feat/lane-c-hmm-viterbi`: Lane C working branch.
  * `feat/lane-d-api-ui`: Lane D working branch.