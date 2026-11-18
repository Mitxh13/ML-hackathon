"""Model evaluation, comparison plotting, and prediction scoring adhering to Contract 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.contracts.schemas import MODEL_OUTPUT_COLS, validate_model_output_df


def predict_candidates(
    model: Any,
    candidates_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Score candidate segments with model predicted probabilities and binary labels.

    Args:
        model: Trained classifier supporting predict_proba or predict.
        candidates_df: Candidate DataFrame with 'feat_*' columns.
        feature_cols: Optional explicit list of feature column names.

    Returns:
        DataFrame adhering to Contract 2 (MODEL_OUTPUT_COLS).
    """
    df = candidates_df.copy()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("feat_")]

    if not feature_cols:
        raise ValueError("No feature columns available to score candidates.")

    X = df[feature_cols].fillna(0.0).values
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    df["pred_prob"] = np.clip(probs, 0.0, 1.0)
    df["pred_label"] = (probs >= 0.5).astype(int)

    validate_model_output_df(df)
    return df


def plot_model_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot cross-validation comparison of candidate classification models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = [c for c in ["cv_macro_f1", "cv_top1_accuracy"] if c in results_df.columns]

    melted = results_df.melt(
        id_vars=["model_family"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score"
    )
    sns.barplot(data=melted, x="model_family", y="Score", hue="Metric", ax=ax, palette="Blues_r")
    ax.set_title("Cross-Validated Model Comparison (GroupKFold by Trip)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix for candidate classification."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Candidate Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Non-Target (0)", "Target (1)"])
    ax.set_yticklabels(["Non-Target (0)", "Target (1)"])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
