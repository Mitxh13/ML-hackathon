"""CLI and orchestration runner for Section 2.3 & Report Section 4.

Executes:
1. Optuna Bayesian hyperparameter optimization on LightGBM.
2. Trial history logging to artifacts/logs/bayesian_trials.csv.
3. Diagnostic plotting (optimization history, parameter importance, ROC, tuning delta).
4. Direct comparative benchmarking (Default vs Tuned LightGBM) on CV and student_test.
5. Export of production champion model bundle to artifacts/models/best_model.joblib.
6. Generation of Section 4 table results/tables/bayesian_tuning_comparison.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.contracts.schemas import validate_candidate_df, validate_model_output_df
from src.lane_b_models.bayesian_opt import (
    evaluate_tuning_improvement,
    optimize_lightgbm,
    plot_optimization_history,
    plot_param_importances,
    plot_roc_comparison,
    plot_tuning_comparison,
)
from src.lane_b_models.export_model import export_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_section23")


def run_section23(
    train_parquet: str = "data/processed/train_candidates.parquet",
    student_test_parquet: str = "data/processed/student_test_candidates.parquet",
    n_trials: int = 25,
    n_splits: int = 5,
    random_state: int = 42,
    export_path: str = "artifacts/models/best_model.joblib",
    trials_log_path: str = "artifacts/logs/bayesian_trials.csv",
    summary_log_path: str = "artifacts/logs/tuning_summary_section23.json",
    comparison_table_path: str = "results/tables/bayesian_tuning_comparison.csv",
) -> dict[str, Any]:
    """Execute complete Section 2.3 Bayesian optimization, evaluation, and export workflow."""
    start_total = time.time()
    logger.info("==================================================================")
    logger.info("Starting Section 2.3: Bayesian Hyperparameter Optimization & Export")
    logger.info("==================================================================")

    # 1. Load Parquet Candidate Tables
    logger.info("Loading training candidates from %s...", train_parquet)
    train_df = pd.read_parquet(train_parquet)
    validate_candidate_df(train_df)

    logger.info("Loading student_test candidates from %s...", student_test_parquet)
    student_test_df = pd.read_parquet(student_test_parquet)
    validate_candidate_df(student_test_df)

    feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
    logger.info("Dataset loaded: Train=%d rows (%d trips), StudentTest=%d rows (%d trips), %d features",
                len(train_df), train_df["trip_id"].nunique(),
                len(student_test_df), student_test_df["trip_id"].nunique(),
                len(feature_cols))

    # 2. Run Optuna Bayesian Search
    logger.info("Running Bayesian Optimization (%d trials, %d CV folds)...", n_trials, n_splits)
    best_params, best_score, study = optimize_lightgbm(
        candidates_df=train_df,
        n_trials=n_trials,
        n_splits=n_splits,
        log_file=trials_log_path,
        random_state=random_state,
        study_name="lightgbm_bayesian_tuning",
    )
    logger.info("Bayesian Optimization Complete! Best Composite Score: %.4f", best_score)
    logger.info("Best Hyperparameters: %s", json.dumps(best_params, indent=2))

    # 3. Save Diagnostic Figures
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    Path("artifacts/logs").mkdir(parents=True, exist_ok=True)

    logger.info("Generating optimization history plot...")
    plot_optimization_history(study, save_path="results/figures/bayesian_opt_history.png")

    logger.info("Generating parameter importance plot...")
    plot_param_importances(study, save_path="results/figures/bayesian_param_importances.png")

    # 4. Compare Default vs Tuned LightGBM
    logger.info("Benchmarking Default LightGBM vs Bayesian-Tuned LightGBM...")
    comparison_df, default_clf, tuned_clf = evaluate_tuning_improvement(
        train_df=train_df,
        test_df=student_test_df,
        best_params=best_params,
        n_splits=n_splits,
    )

    logger.info("Generating tuning comparison bar chart...")
    plot_tuning_comparison(comparison_df, save_path="results/figures/tuning_before_after_comparison.png")

    # 5. Generate ROC Curve on Student Test
    X_test = student_test_df[feature_cols].fillna(0.0).values
    y_test = student_test_df["is_target"].values
    def_probs = default_clf.predict_proba(X_test)[:, 1]
    tun_probs = tuned_clf.predict_proba(X_test)[:, 1]

    logger.info("Generating ROC curve...")
    plot_roc_comparison(y_test, def_probs, tun_probs, save_path="results/figures/roc_curve_champion.png")

    # 6. Save Comparison Table
    comparison_df.to_csv(comparison_table_path, index=False)
    logger.info("Saved tuning comparison table to %s", comparison_table_path)
    print("\n" + "=" * 90)
    print("SECTION 4: BAYESIAN TUNING BENCHMARK TABLE")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90 + "\n")

    # 7. Update Section 3 Table (Adding Tuned LightGBM)
    sec3_path = Path("results/tables/models_comparison_section3.csv")
    if sec3_path.exists():
        sec3_df = pd.read_csv(sec3_path)
        tuned_row = comparison_df[comparison_df["configuration"] == "Bayesian-Tuned LightGBM"].iloc[0]

        # Detect column naming convention used in the existing table
        model_col = "Model" if "Model" in sec3_df.columns else "model_family"
        new_row = {
            model_col: "Bayesian-Tuned LightGBM",
        }
        # Map tuned metrics to whichever column names the table uses
        col_map = {
            "CV Macro F1": "cv_macro_f1", "CV Top-1 Accuracy": "cv_top1_accuracy",
            "Student-Test Macro F1": "student_test_macro_f1",
            "Student-Test Top-1 Accuracy": "student_test_top1_accuracy",
            "Student-Test Log Loss": "student_test_log_loss",
            "Student-Test Inference Time (s)": "inference_time_sec",
        }
        for sec3_col, comp_key in col_map.items():
            target_col = sec3_col if sec3_col in sec3_df.columns else comp_key
            if target_col in sec3_df.columns:
                new_row[target_col] = tuned_row.get(comp_key, tuned_row.get(f"student_test_{comp_key}", ""))

        # Avoid duplicate rows
        sec3_df = sec3_df[sec3_df[model_col] != "Bayesian-Tuned LightGBM"]
        sec3_df = pd.concat([sec3_df, pd.DataFrame([new_row])], ignore_index=True)
        sec3_df.to_csv(sec3_path, index=False)
        logger.info("Updated Section 3 comparison table at %s", sec3_path)

    # 8. Export Champion Model Artifact
    metadata = {
        "model_family": "LightGBM",
        "tuning_method": "Optuna TPE Bayesian Search",
        "n_trials": n_trials,
        "n_splits": n_splits,
        "best_params": best_params,
        "best_cv_score": best_score,
        "metrics": comparison_df[comparison_df["configuration"] == "Bayesian-Tuned LightGBM"].to_dict(orient="records")[0],
    }

    logger.info("Exporting champion model bundle to %s...", export_path)
    bundle_path = export_model_bundle(
        model=tuned_clf,
        feature_names=feature_cols,
        save_path=export_path,
        metadata=metadata,
    )
    logger.info("Successfully exported bundle to %s", bundle_path)

    # 9. Verify Contract 2 on Test Predictions
    scored_test = student_test_df.copy()
    scored_test["pred_prob"] = np.clip(tun_probs, 0.0, 1.0)
    scored_test["pred_label"] = (tun_probs >= 0.5).astype(int)
    validate_model_output_df(scored_test)
    logger.info("Contract 2 validation passed on scored student_test dataset!")

    # 10. Save Detailed JSON Log
    total_time = round(time.time() - start_total, 2)
    summary_data = {
        "status": "SUCCESS",
        "total_runtime_seconds": total_time,
        "best_params": best_params,
        "best_cv_composite_score": best_score,
        "benchmark": comparison_df.to_dict(orient="records"),
        "exported_bundle": str(bundle_path),
    }

    with open(summary_log_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved tuning summary to %s", summary_log_path)
    logger.info("Section 2.3 completed successfully in %.2f seconds!", total_time)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Section 2.3 Bayesian Optimization & Export.")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--splits", type=int, default=5, help="Number of GroupKFold splits")
    args = parser.parse_args()

    run_section23(n_trials=args.trials, n_splits=args.splits)
