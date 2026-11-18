"""FastAPI server running on Laptop 1 for PathSnap model hosting and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yaml

from src.contracts.schemas import EvaluationResponse
from src.lane_d_app.pipeline import PathSnapPipeline
from src.lane_b_models.export_model import load_model_bundle

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Backend API serving ML-probability HMM map matching and evaluation.",
    version="1.0.0",
)

# Enable CORS for Laptop 2 Evaluator UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline_instance: PathSnapPipeline = PathSnapPipeline()


def load_config() -> dict[str, Any]:
    """Load default yaml configuration."""
    config_path = Path("configs/default_config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@app.on_event("startup")
def startup_event():
    """Initialize model and pipeline on API startup."""
    global pipeline_instance
    config = load_config()

    # Attempt to load trained model bundle if available
    model_path = config.get("paths", {}).get("models_dir", "artifacts/models") + "/best_model.joblib"
    model = None
    feature_names = None

    if Path(model_path).exists():
        try:
            bundle = load_model_bundle(model_path)
            model = bundle["model"]
            feature_names = bundle["feature_names"]
            print(f"[PathSnap API] Loaded model bundle from {model_path}")
        except Exception as e:
            print(f"[PathSnap API] Warning: Failed to load model bundle ({e}). Using baseline mode.")

    pipeline_instance = PathSnapPipeline(model=model, feature_names=feature_names)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "PathSnap API"}


@app.get("/trips")
def list_trips() -> dict[str, list[str]]:
    """List available test trips in student_test and evaluator_test directories."""
    trips: dict[str, list[str]] = {"student_test": [], "evaluator_test": []}

    student_dir = Path("data/student_test")
    if student_dir.exists():
        trips["student_test"] = sorted(
            [d.name for d in student_dir.iterdir() if d.is_dir()]
        )

    eval_dir = Path("data/evaluator_test")
    if eval_dir.exists():
        trips["evaluator_test"] = sorted(
            [d.name for d in eval_dir.iterdir() if d.is_dir()]
        )

    return trips


@app.get("/evaluate/{trip_id}", response_model=EvaluationResponse)
def evaluate_trip(
    trip_id: str,
    partition: str = Query("student_test", description="'student_test' or 'evaluator_test'"),
    mode: str = Query("ml_hmm", description="'ml_hmm', 'distance_baseline', or 'classifier_only'"),
) -> EvaluationResponse:
    """Run route reconstruction on a selected trip and compute evaluation metrics.

    Ground-truth reference route files remain strictly server-side on Laptop 1.
    """
    trip_dir = Path(f"data/{partition}/{trip_id}")
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip directory '{trip_dir}' not found.")

    try:
        response = pipeline_instance.process_trip(trip_id=trip_id, trip_dir=trip_dir, mode=mode)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    host = cfg.get("api", {}).get("api_host", "0.0.0.0")
    port = cfg.get("api", {}).get("api_port", 8000)
    uvicorn.run("src.lane_d_app.api_server:app", host=host, port=port, reload=True)
