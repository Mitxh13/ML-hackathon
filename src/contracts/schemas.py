"""Shared handoff contracts, schemas, dataclasses, and validation helpers for PathSnap.

Enforces clear schema boundaries between:
- Lane A (Data & Feature Engineering) -> Lane B (Model Training) & Lane C (HMM Decoding)
- Lane B (Model Output Scores) -> Lane C (ML-Probability HMM Decoding)
- Lane A (Ground-Truth Point Table) -> Lane B & Lane C (Evaluation)
- Lane C (Decoder Output) -> Lane D (Inference Pipeline & Server)
- Lane D (FastAPI Response) -> Laptop 2 Evaluator UI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

# ==============================================================================
# Contract 1: Candidate Table Schema (Lane A -> Lane B & Lane C)
# ==============================================================================
CANDIDATE_BASE_COLS: list[str] = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "gps_lon",
    "gps_lat",
    "candidate_arc_id",
    "from_node_id",
    "to_node_id",
    "dist_to_arc_m",
    "is_target",
]


@dataclass
class CandidateRecord:
    """Single candidate arc candidate for a GPS observation."""

    trip_id: str
    obs_idx: int
    timestamp: float
    gps_lon: float
    gps_lat: float
    candidate_arc_id: int
    from_node_id: int
    to_node_id: int
    dist_to_arc_m: float
    is_target: int = -1  # 1: target, 0: non-target, -1: unlabeled / blind inference
    features: dict[str, float] = field(default_factory=dict)


def validate_candidate_df(
    df: pd.DataFrame,
    require_target: bool = True,
    allowed_extra_cols: Optional[set[str]] = None,
) -> bool:
    """Validate that candidate DataFrame adheres to Contract 1.

    Args:
        df: DataFrame containing candidate records.
        require_target: If True, requires 'is_target' column with values in {0, 1}
            (or -1 for unlabeled inference).
        allowed_extra_cols: Optional set of allowed column names that do not need
            to start with 'feat_' (e.g. 'pred_prob', 'pred_label').

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a valid pandas DataFrame.")

    missing_cols = [c for c in CANDIDATE_BASE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Candidate DataFrame missing required columns: {missing_cols}")

    if df.empty:
        return True

    # Validate types
    if not pd.api.types.is_numeric_dtype(df["obs_idx"]):
        raise TypeError("'obs_idx' must be integer numeric.")
    if not pd.api.types.is_numeric_dtype(df["candidate_arc_id"]):
        raise TypeError("'candidate_arc_id' must be integer numeric.")
    if not pd.api.types.is_numeric_dtype(df["dist_to_arc_m"]):
        raise TypeError("'dist_to_arc_m' must be float numeric.")

    # Validate feature naming convention
    ignored = set(CANDIDATE_BASE_COLS)
    if allowed_extra_cols:
        ignored.update(allowed_extra_cols)

    feature_cols = [c for c in df.columns if c not in ignored]
    invalid_features = [c for c in feature_cols if not c.startswith("feat_")]
    if invalid_features:
        raise ValueError(
            f"All engineered feature columns must be prefixed with 'feat_'. Found: {invalid_features}"
        )

    if require_target:
        if not pd.api.types.is_numeric_dtype(df["is_target"]):
            raise TypeError("'is_target' must be numeric (1, 0, or -1).")

    return True


# ==============================================================================
# Contract 2: Model Output Scores Schema (Lane B -> Lane C)
# ==============================================================================
MODEL_OUTPUT_COLS: list[str] = CANDIDATE_BASE_COLS + ["pred_prob", "pred_label"]


@dataclass
class ModelOutputRecord(CandidateRecord):
    """Candidate record enriched with model prediction scores."""

    pred_prob: float = 0.0
    pred_label: int = 0


def validate_model_output_df(df: pd.DataFrame) -> bool:
    """Validate that model output DataFrame adheres to Contract 2.

    Args:
        df: DataFrame containing candidate records with model predictions.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    validate_candidate_df(
        df, require_target=False, allowed_extra_cols={"pred_prob", "pred_label"}
    )

    for col in ["pred_prob", "pred_label"]:
        if col not in df.columns:
            raise ValueError(f"Model output DataFrame missing required column '{col}'.")

    if not df.empty:
        if not pd.api.types.is_numeric_dtype(df["pred_prob"]):
            raise TypeError("'pred_prob' must be float numeric.")
        if ((df["pred_prob"] < 0.0) | (df["pred_prob"] > 1.0)).any():
            raise ValueError("All 'pred_prob' values must be bounded within [0.0, 1.0].")
        if not pd.api.types.is_numeric_dtype(df["pred_label"]):
            raise TypeError("'pred_label' must be numeric (0 or 1).")

    return True


# ==============================================================================
# Contract 3: Ground-Truth Point Table Schema (Lane A -> Lane B & Lane C)
# ==============================================================================
POINT_GT_COLS: list[str] = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "gps_lon",
    "gps_lat",
    "true_arc_id",
]


@dataclass
class PointGTRecord:
    """Ground-truth point-to-arc assignment for a GPS observation."""

    trip_id: str
    obs_idx: int
    timestamp: float
    gps_lon: float
    gps_lat: float
    true_arc_id: int


def validate_point_gt_df(df: pd.DataFrame) -> bool:
    """Validate that point ground-truth DataFrame adheres to Contract 3.

    Args:
        df: DataFrame containing point-to-segment ground truth.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a valid pandas DataFrame.")

    missing_cols = [c for c in POINT_GT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Point GT DataFrame missing required columns: {missing_cols}")

    if not df.empty:
        if not pd.api.types.is_numeric_dtype(df["obs_idx"]):
            raise TypeError("'obs_idx' must be integer numeric.")
        if not pd.api.types.is_numeric_dtype(df["true_arc_id"]):
            raise TypeError("'true_arc_id' must be integer numeric.")

    return True


# ==============================================================================
# Contract 4: Decoder Input & Output Schema (Lane C -> Lane D)
# ==============================================================================
class DecoderOutput(BaseModel):
    """Output contract produced by ViterbiRouteDecoder."""

    trip_id: str
    mode: str = Field(..., description="'classifier_only', 'distance_hmm', or 'ml_hmm'")
    point_predictions: list[int] = Field(
        ..., description="Predicted arc_id for every obs_idx (t = 0 ... T-1)"
    )
    reconstructed_route: list[int] = Field(
        ...,
        description="Chronological sequence of traversed arc_id values with consecutive duplicates collapsed",
    )
    inference_time_sec: float = Field(default=0.0, ge=0.0)

    model_config = {"extra": "allow"}


# ==============================================================================
# Contract 5: FastAPI Response Schema (Laptop 1 API -> Laptop 2 UI)
# ==============================================================================
class EvaluationMetrics(BaseModel):
    """Standard evaluation metrics schema."""

    macro_f1: Optional[float] = None
    top1_accuracy: Optional[float] = None
    route_continuity: Optional[float] = None
    log_loss: Optional[float] = None
    candidate_miss_rate: Optional[float] = None
    inference_time_sec: Optional[float] = None


class EvaluationResponse(BaseModel):
    """JSON-serializable response payload for Laptop 2 Evaluator UI."""

    trip_id: str
    num_gps_points: int
    point_predictions: list[int]
    reconstructed_route: list[int]
    metrics: dict[str, Optional[float]]
    folium_map_html: str

    model_config = {"extra": "allow"}
