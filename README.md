# PathSnap: Distributed Map-Matching & Route Reconstruction
**Distributed Map-Matching & Route Reconstruction**

PathSnap reconstructs realistic vehicle routes by matching noisy $\approx 1\text{ Hz}$ GPS trajectories to an underlying directed road network. It combines spatial KD-Tree candidate generation, supervised classification across $\ge 5$ model families, and Hidden Markov Model (HMM) + Viterbi decoding with road topological continuity constraints.

---

## 📁 Repository Structure

```
pathsnap/
├── AGENTS.md                  # Development rules, metrics, anti-leakage policies, and lane ownership
├── CLAUDE.md                  # Development guidelines (mirrors AGENTS.md)
├── README.md                  # Setup instructions, architecture overview, and execution guide
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excludes raw data, virtualenv, caches, and models
├── configs/
│   ├── __init__.py
│   └── default_config.yaml    # Configurable candidate radii, HMM params, and API ports
├── data/
│   ├── train/                 # 80 training trips (.track, .nodes, .arcs, .route)
│   ├── student_test/          # 10 student-testing trips
│   ├── evaluator_test/        # 10 reserved evaluator-testing trips (UNTOUCHED during dev)
│   └── processed/             # Candidate tables & extracted feature parquet/csv files
├── src/
│   ├── __init__.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   └── schemas.py         # Shared column names, dataclasses, Pydantic models, validators
│   ├── lane_a_data/           # LANE A: Data loaders, preprocessing, ground truth, candidates, features, EDA
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   ├── ground_truth.py
│   │   ├── candidates.py
│   │   ├── features.py
│   │   └── eda_plots.py
│   ├── lane_b_models/         # LANE B: >=5 models, GroupKFold CV, Bayesian tuning, evaluation, export
│   │   ├── __init__.py
│   │   ├── train_models.py
│   │   ├── bayesian_opt.py
│   │   ├── evaluate_models.py
│   │   └── export_model.py
│   ├── lane_c_hmm/            # LANE C: Road graph, HMM/Viterbi decoder, shared metrics, Folium maps
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── road_graph.py
│   │   ├── decoder.py
│   │   ├── compare_hmm.py
│   │   └── folium_maps.py
│   └── lane_d_app/            # LANE D: End-to-end pipeline, FastAPI server (Laptop 1), Evaluator UI (Laptop 2)
│       ├── __init__.py
│       ├── pipeline.py
│       ├── api_server.py
│       └── ui_client.py
├── artifacts/
│   ├── models/                # Exported model weights & scalers (.joblib)
│   └── logs/                  # Bayesian search records & training histories (.csv)
├── results/
│   ├── tables/                # CSV comparison tables for Report Sections 3, 4, 5
│   ├── figures/               # Saved PNG plots for technical report
│   └── maps/                  # Exported interactive Folium HTML maps
└── tests/
    ├── __init__.py
    └── test_contracts.py      # Smoke tests verifying A -> B -> C -> D handoff schemas & metrics
```

---

## 🐍 Environment Setup (.venv)

PathSnap uses a dedicated Python virtual environment (`.venv/`). Never install packages globally.

### macOS / Linux
```bash
# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running Smoke Tests

Verify that all handoff contracts and shared evaluation metrics pass:

### Linux / macOS
```bash
.venv/bin/pytest tests/test_contracts.py -v
```

### Windows
```powershell
.\.venv\Scripts\pytest.exe tests/test_contracts.py -v
```

---

## 💻 Two-Laptop Execution Setup

PathSnap deploys across two networked laptops during final evaluation:

```
┌─────────────────────────────────┐           HTTP REST           ┌─────────────────────────────────┐
│            LAPTOP 1             │◄─────────────────────────────►│            LAPTOP 2             │
│   FastAPI Server (Port 8000)    │       /evaluate/{trip_id}     │      Streamlit Evaluator UI     │
│   - Hosts ML Models & Graph     │                               │   - Evaluator Selects Trip      │
│   - Executes Viterbi Decoder    │                               │   - Renders Route Maps (Folium) │
│   - Calculates Server-Side GT   │                               │   - Displays Benchmark Metrics  │
└─────────────────────────────────┘                               └─────────────────────────────────┘
```

### Laptop 1 (Backend Server)
Find your local IP address (`ipconfig` on Windows or `ifconfig` on Linux/macOS), then run:
```powershell
# Windows
.\.venv\Scripts\python.exe -m uvicorn src.lane_d_app.api_server:app --host 0.0.0.0 --port 8000 --reload
```
```bash
# Linux / macOS
.venv/bin/python -m uvicorn src.lane_d_app.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Laptop 2 (Evaluator UI) - web UI (used for the demo)
Static web page in `src/lane_d_app/web_ui/` (no Python packages needed on Laptop 2):
```bash
python -m http.server 5500 --directory src/lane_d_app/web_ui
```
Open `http://localhost:5500/?api=http://<laptop-1-ip>:8000`. The address can also be changed from the
connection button in the top-right corner. Trips are listed per partition (Evaluator / Student test).
Run a single trip, or **Run all trips**, then **Export CSV** for report tables. Needs internet access for
the map tiles and Leaflet.

The UI receives GPS points, the predicted route and metric values only. The map returned by the API is
built without the reference route, so ground truth stays on Laptop 1.

### Laptop 2 (alternative) - Streamlit UI
Connect to Laptop 1 by entering its IP address in the Streamlit sidebar (e.g. `http://192.168.1.50:8000`):
```powershell
# Windows
.\.venv\Scripts\streamlit.exe run src/lane_d_app/ui_client.py
```
```bash
# Linux / macOS
.venv/bin/streamlit run src/lane_d_app/ui_client.py
```

---

## 🌿 4-Lane Git Workflow & Branching Strategy

To eliminate code collisions across team members, all development occurs in isolated lane branches:

### Branches
- `main`: Production-ready code only.
- `dev/integration`: Staging branch where Lanes A, B, C, and D merge.
- `feat/lane-a-data-features`: Lane A (loaders, preprocessing, candidates, features, EDA)
- `feat/lane-b-models-tuning`: Lane B (>=5 models, GroupKFold, Optuna Bayesian search)
- `feat/lane-c-hmm-viterbi`: Lane C (road graph, HMM/Viterbi decoder, metrics, Folium maps)
- `feat/lane-d-api-ui`: Lane D (FastAPI, Streamlit UI, networking, documentation)

### 3 Integration Checkpoints
1. **Join 1**: Lane A merges candidate table generator into `dev/integration`.
2. **Join 2**: Lane B merges trained model artifacts and out-of-fold probability predictions into `dev/integration`.
3. **Join 3**: Lanes A, B, and C merge into `dev/integration`, and Lane D swaps the mock pipeline for the full production pipeline.
