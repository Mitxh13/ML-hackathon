"""Model and artifact serialization utilities for PathSnap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib


def export_model_bundle(
    model: Any,
    feature_names: list[str],
    save_path: str = "artifacts/models/best_model.joblib",
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Export trained model, feature names, and metadata into a joblib artifact.

    Args:
        model: Trained classifier.
        feature_names: List of feature names used during training.
        save_path: Destination path.
        metadata: Optional dictionary with hyperparameters, training dates, metrics.

    Returns:
        Path to exported artifact.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "metadata": metadata or {},
    }
    joblib.dump(bundle, path)
    return path


def load_model_bundle(load_path: str = "artifacts/models/best_model.joblib") -> dict[str, Any]:
    """Load exported model bundle.

    Returns:
        Dictionary with keys 'model', 'feature_names', 'metadata'.
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {load_path}")
    return joblib.load(path)
