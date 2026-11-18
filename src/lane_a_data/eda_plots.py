"""Exploratory Data Analysis (EDA) plotting utilities for PathSnap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Generate missing values bar chart for a DataFrame."""
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df.isnull().sum()
    missing.plot(kind="bar", color="#3B82F6", ax=ax)
    ax.set_title("Missing Values Count per Column")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_speed_distribution(track_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot distribution of GPS speed observations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    speeds = track_df["speed_mps"].dropna() if "speed_mps" in track_df.columns else []
    sns.histplot(speeds, bins=30, kde=True, color="#10B981", ax=ax)
    ax.set_title("GPS Speed Distribution (m/s)")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_candidate_distances(candidates_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot candidate distance distribution separated by target vs non-target."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if "is_target" in candidates_df.columns:
        sns.boxplot(data=candidates_df, x="is_target", y="dist_to_arc_m", palette="Set2", ax=ax)
        ax.set_xticklabels(["Non-Target (0)", "Target (1)"])
    else:
        sns.histplot(candidates_df["dist_to_arc_m"], bins=30, ax=ax)
    ax.set_title("Candidate Distance to Road Segment (meters)")
    ax.set_ylabel("Distance (m)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
