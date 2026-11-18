"""CLI runner and diagnostic plotting script for Section 2.2 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.lane_b_models.models_2_2 import Section22Trainer
from src.lane_b_models.export_model import export_model_bundle


def plot_cv_and_test_metrics(
    consolidated_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Plot cross-validation and test comparison bar charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. CV Macro F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = sns.barplot(
        data=consolidated_df,
        x="Model",
        y="CV Macro F1",
        hue="Model",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Cross-Validated Candidate Macro F1 (GroupKFold by Trip)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1-Score")
    for bar in bars.patches:
        ax.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    fig.savefig(figures_dir / "model_cv_macro_f1.png", dpi=300)
    plt.close(fig)

    # 2. Student Test Top-1 & Macro F1 Comparison
    melted = consolidated_df.melt(
        id_vars=["Model"],
        value_vars=["Student-Test Top-1 Accuracy", "Student-Test Macro F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="crest", ax=ax)
    ax.set_title("Student-Test Performance Across Section 2.2 Models", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    plt.tight_layout()
    fig.savefig(figures_dir / "model_student_test_top1_f1.png", dpi=300)
    plt.close(fig)

    # 3. Log Loss Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = sns.barplot(
        data=consolidated_df,
        x="Model",
        y="Student-Test Log Loss",
        hue="Model",
        palette="Reds_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Candidate Probability Calibration: Safe Binary Log Loss", fontsize=13, fontweight="bold")
    ax.set_ylabel("Log Loss (Lower is Better)")
    for bar in bars.patches:
        ax.annotate(
            f"{bar.get_height():.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(figures_dir / "model_log_loss.png", dpi=300)
    plt.close(fig)


def plot_all_confusion_matrices(
    scored_dfs: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Plot a 2x3 grid of confusion matrices for all evaluated models."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(scored_dfs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (name, df) in enumerate(scored_dfs.items()):
        ax = axes[idx]
        y_true = df["is_target"].values
        y_pred = df["pred_label"].values
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{name} (Confusion Matrix)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Ground Truth")
        ax.set_xticklabels(["Non-Target (0)", "Target (1)"])
        ax.set_yticklabels(["Non-Target (0)", "Target (1)"])

    # Hide extra subplot
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    fig.savefig(figures_dir / "model_confusion_matrices.png", dpi=300)
    plt.close(fig)


def plot_mlp_loss_curve(
    loss_history: list[float],
    figures_dir: Path,
) -> None:
    """Plot actual epoch-wise training loss curve for the MLPClassifier."""
    if not loss_history:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color="#2563EB", lw=2)
    ax.set_title("MLPClassifier: Epoch-Wise Training Loss Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(figures_dir / "mlp_training_loss_curve.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Section 2.2 Models")
    parser.add_argument("--train-parquet", default="data/processed/train_candidates.parquet", help="Train candidates parquet")
    parser.add_argument("--test-parquet", default="data/processed/student_test_candidates.parquet", help="Test candidates parquet")
    parser.add_argument("--output-tables-dir", default="results/tables", help="Directory for tables")
    parser.add_argument("--output-figures-dir", default="results/figures", help="Directory for figures")
    parser.add_argument("--models-dir", default="artifacts/models", help="Directory to save best model")
    parser.add_argument("--logs-dir", default="artifacts/logs", help="Directory for training history logs")
    parser.add_argument("--sample-trips", type=int, default=None, help="Optional trip limit for fast execution")

    args = parser.parse_args()

    train_path = Path(args.train_parquet)
    test_path = Path(args.test_parquet)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed candidate parquet files not found at {train_path} or {test_path}. "
            "Please run 'python -m src.lane_a_data.build_candidate_tables' first."
        )

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    print(f"Loading student-test data from {test_path}...")
    test_df = pd.read_parquet(test_path)

    if args.sample_trips:
        sampled_trips = train_df["trip_id"].unique()[: args.sample_trips]
        train_df = train_df[train_df["trip_id"].isin(sampled_trips)].reset_index(drop=True)
        print(f"Subsampled train data to {len(sampled_trips)} trips ({len(train_df):,} rows).")

    trainer = Section22Trainer()
    cv_df, oof_dict = trainer.run_group_kfold_cv(train_df, n_splits=5)
    test_df_results, scored_dfs = trainer.evaluate_student_test(test_df)

    # Consolidated Results Table matching Report §3.1
    consolidated_df = pd.merge(cv_df, test_df_results, on="Model")
    consolidated_df = consolidated_df[[
        "Model",
        "Algorithm Family",
        "Key Configuration",
        "CV Macro F1",
        "CV Top-1 Accuracy",
        "Student-Test Macro F1",
        "Student-Test Top-1 Accuracy",
        "Student-Test Log Loss",
        "Student-Test Inference Time (s)",
    ]]

    tables_dir = Path(args.output_tables_dir)
    figures_dir = Path(args.output_figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV tables
    section22_csv = tables_dir / "models_2_2_comparison.csv"
    section3_csv = tables_dir / "models_comparison_section3.csv"
    consolidated_df.to_csv(section22_csv, index=False)
    consolidated_df.to_csv(section3_csv, index=False)
    print(f"\n Saved consolidated model comparison table to:\n   - {section22_csv}\n   - {section3_csv}")

    # Generate Figures
    print("\n Generating publication figures...")
    plot_cv_and_test_metrics(consolidated_df, figures_dir)
    plot_all_confusion_matrices(scored_dfs, figures_dir)
    if "MLP" in trainer.learning_curves_:
        plot_mlp_loss_curve(trainer.learning_curves_["MLP"], figures_dir)
    print(f" Saved all diagnostic figures to {figures_dir}")

    # Determine and export best performing model
    # Ranked primarily by Student-Test Top-1 Accuracy, then Macro F1
    best_row = consolidated_df.sort_values(
        by=["Student-Test Top-1 Accuracy", "Student-Test Macro F1"], ascending=False
    ).iloc[0]
    best_name = best_row["Model"]
    best_model = trainer.fitted_models[best_name]

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    export_path = models_dir / "best_model.joblib"

    metadata = {
        "best_model_name": best_name,
        "algorithm_family": best_row["Algorithm Family"],
        "cv_macro_f1": float(best_row["CV Macro F1"]),
        "student_test_top1": float(best_row["Student-Test Top-1 Accuracy"]),
        "student_test_f1": float(best_row["Student-Test Macro F1"]),
        "feature_count": len(trainer.feature_cols),
    }

    export_model_bundle(
        model=best_model,
        feature_names=trainer.feature_cols,
        save_path=str(export_path),
        metadata=metadata,
    )
    print(f"\n Exported Best Model '{best_name}' to {export_path}")

    # Save training logs & learning curves
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_json = logs_dir / "training_history_section22.json"
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics_table": consolidated_df.to_dict(orient="records"),
                "learning_curves": trainer.learning_curves_,
                "metadata": metadata,
            },
            f,
            indent=2,
        )
    print(f" Saved training history and loss logs to {log_json}")

    print("\n Section 2.2 Model Training, Evaluation & Plotting Complete!\n")
    print(consolidated_df.to_string(index=False))


if __name__ == "__main__":
    main()
