"""Leakage-safe shared training setup and Section 2.1 baseline models.

This module is additive: the existing multi-family Section 2.2/2.3 interfaces in
``train_models.py`` remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.contracts.schemas import CANDIDATE_BASE_COLS, validate_candidate_df
from src.lane_c_hmm.metrics import compute_log_loss_safe, compute_macro_f1

LOGGER = logging.getLogger(__name__)
IDENTIFIER_COLUMNS = ["trip_id", "obs_idx", "candidate_arc_id", "from_node_id", "to_node_id"]
MODEL_DESCRIPTIONS = {
    "LogisticRegression": (
        "Linear probabilistic classifier for candidate-road relevance; class_weight='balanced' "
        "compensates for the target imbalance."
    ),
    "GaussianNB": (
        "Gaussian Naive Bayes estimates empirical class priors and assumes conditional feature "
        "independence; correlated geometric and kinematic features can limit its fit."
    ),
    "KNN": (
        "Distance-weighted KNN identifies similar candidate-road examples in standardized "
        "feature space; prediction cost grows with the stored training set."
    ),
}


@dataclass
class TrainingData:
    """Prepared labelled examples plus identifiers kept outside the model matrix."""
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    identifiers: pd.DataFrame
    feature_columns: list[str]
    report: dict[str, Any]


def prepare_training_data(candidates: pd.DataFrame) -> TrainingData:
    """Validate and split candidates, excluding unlabeled ``is_target == -1`` rows."""
    if candidates is None or not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if "is_target" not in candidates:
        raise ValueError("candidate table requires an is_target target column")
    if "trip_id" not in candidates or candidates.trip_id.isna().any():
        raise ValueError("trip_id must be present for every candidate row")
    feature_columns = [c for c in candidates.columns if c.startswith("feat_")]
    if not feature_columns:
        raise ValueError("candidate table has no feat_* numerical columns")
    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(candidates[c])]
    if non_numeric:
        raise TypeError(f"feat_* columns must be numeric: {non_numeric}")
    # Shared candidate validation is applied to its declared contract plus features;
    # Section 1.2 diagnostic columns are not part of the shared model input schema.
    base = [c for c in CANDIDATE_BASE_COLS if c in candidates]
    validate_candidate_df(candidates[base + feature_columns])
    valid = candidates["is_target"].isin([0, 1])
    labelled = candidates.loc[valid].copy()
    if labelled.empty:
        raise ValueError("no labelled candidate rows (is_target in {0, 1})")
    if labelled.trip_id.isna().any():
        raise ValueError("trip_id cannot be missing for grouped validation")
    X = labelled[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    y = labelled["is_target"].astype(int).reset_index(drop=True)
    groups = labelled["trip_id"].astype(str).reset_index(drop=True)
    ids = labelled[[c for c in IDENTIFIER_COLUMNS if c in labelled]].reset_index(drop=True)
    missing = X.isna().sum()
    return TrainingData(
        X=X.reset_index(drop=True), y=y, groups=groups, identifiers=ids,
        feature_columns=feature_columns,
        report={"candidate_rows": int(len(candidates)), "labelled_rows": int(len(labelled)),
                "unlabelled_rows": int((~valid).sum()), "trip_count": int(groups.nunique()),
                "candidate_table_trip_count": int(candidates.trip_id.nunique()),
                "trips_without_labelled_candidates": sorted(
                    set(candidates.trip_id.astype(str)) - set(groups.unique())
                ),
                "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
                "feature_columns": feature_columns,
                "missing_values_by_feature": {str(k): int(v) for k, v in missing.items()},
                "features_with_missing_values": int((missing > 0).sum())},
    )


def build_feature_table_from_candidate_geometry(
    geometry_csv: str | Path, output_csv: str | Path, *, data_dir: str | Path = "data",
    partition: str = "train", chunksize: int = 200_000,
) -> Path:
    """Enrich Section 1.2 candidate geometry with Lane A Section 1.3 features.

    The input is streamed by its trip-contiguous rows, so raw data and candidate
    geometry are loaded one trip at a time. Student targets are set to -1 in the
    feature table; student evaluation labels must be loaded from the separate
    ``candidate_labels_student_test.csv`` sidecar.
    """
    from src.lane_a_data.features import extract_candidate_features
    from src.lane_a_data.loaders import load_trip
    from src.lane_a_data.preprocessing import preprocess_track

    source, destination = Path(geometry_csv), Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    available = set(pd.read_csv(source, nrows=0).columns)
    usecols = [c for c in CANDIDATE_BASE_COLS if c in available]
    required = set(CANDIDATE_BASE_COLS) - ({"is_target"} if partition == "student_test" else set())
    if required - set(usecols):
        raise ValueError(f"candidate geometry CSV missing base columns: {sorted(required - set(usecols))}")
    carry = pd.DataFrame()
    seen: set[str] = set()
    write_header = True

    def process(trip_id: str, rows: pd.DataFrame) -> None:
        nonlocal write_header
        trip = load_trip(Path(data_dir) / partition / trip_id)
        needed = {"track", "nodes", "arcs"}
        if needed - trip.keys():
            raise ValueError(f"{partition}/{trip_id}: missing files {sorted(needed - trip.keys())}")
        track = preprocess_track(trip["track"])
        rows = rows.copy()
        if partition == "student_test":
            rows["is_target"] = -1
        features = extract_candidate_features(rows[CANDIDATE_BASE_COLS], trip["nodes"], trip["arcs"], track)
        features.to_csv(destination, mode="w" if write_header else "a", header=write_header, index=False)
        write_header = False
        LOGGER.info("Feature rows %s/%s: %d", partition, trip_id, len(features))

    for chunk in pd.read_csv(source, usecols=usecols, dtype={"trip_id": str}, chunksize=chunksize, low_memory=False):
        combined = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        boundaries = np.flatnonzero(combined.trip_id.ne(combined.trip_id.shift()).to_numpy())
        ends = np.r_[boundaries[1:], len(combined)]
        groups = [combined.iloc[start:end] for start, end in zip(boundaries, ends)]
        if len(groups) == 1:
            carry = groups[0].copy()
            continue
        for group in groups[:-1]:
            trip_id = str(group.trip_id.iloc[0])
            if trip_id in seen:
                raise ValueError(f"candidate rows for trip {trip_id} are not contiguous in {source}")
            seen.add(trip_id)
            process(trip_id, group)
        carry = groups[-1].copy()
    if not carry.empty:
        trip_id = str(carry.trip_id.iloc[0])
        if trip_id in seen:
            raise ValueError(f"candidate rows for trip {trip_id} are not contiguous in {source}")
        process(trip_id, carry)
    if write_header:
        raise ValueError(f"no candidate rows found in {source}")
    return destination


def make_group_folds(data: TrainingData, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize fixed GroupKFold indices and verify disjoint trip IDs."""
    n_groups = data.groups.nunique()
    if n_splits != 5:
        raise ValueError("Section 2.1 requires GroupKFold(n_splits=5)")
    if n_groups < n_splits:
        raise ValueError(f"5-fold GroupKFold requires at least 5 trips; found {n_groups}")
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(data.X, data.y, groups=data.groups))
    for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
        train_trips = set(data.groups.iloc[train_idx])
        val_trips = set(data.groups.iloc[val_idx])
        if train_trips & val_trips:
            raise RuntimeError(f"trip leakage in fold {fold_no}")
        LOGGER.info("Fold %d: %d training trips, %d validation trips; disjoint",
                    fold_no, len(train_trips), len(val_trips))
    return folds


def build_baseline_pipeline(model_name: str, *, n_neighbors: int = 15) -> Pipeline:
    """Build a fresh preprocessing/classifier Pipeline for a Section 2.1 model."""
    name = model_name.lower().replace(" ", "").replace("_", "")
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if name in {"logisticregression", "logistic"}:
        steps.extend([("scaler", StandardScaler()), ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs"))])
    elif name in {"gaussiannb", "naivebayes", "nb"}:
        steps.append(("classifier", GaussianNB(var_smoothing=1e-9, priors=None)))
    elif name in {"knearestneighbors", "knn"}:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        steps.extend([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="minkowski", p=2, n_jobs=-1))])
    else:
        raise ValueError(f"unknown Section 2.1 model: {model_name}")
    return Pipeline(steps)


def _positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = model.named_steps["classifier"].classes_
    positive_index = np.flatnonzero(classes == 1)
    if len(positive_index) != 1:
        raise ValueError(f"model does not expose positive class 1; fitted classes={classes.tolist()}")
    return np.asarray(probabilities[:, positive_index[0]], dtype=float)


def observation_top1_accuracy(
    frame: pd.DataFrame, probabilities: np.ndarray, *, target_column: str | None = None,
) -> float:
    """Count Top-1 correct only when a positive candidate exists and wins.

    Training folds use ``is_target``; the separately joined student label sidecar
    uses ``label``. Accept both so evaluation never has to leak labels into inputs.
    """
    if target_column is None:
        target_column = "is_target" if "is_target" in frame else "label"
    if target_column not in frame:
        raise ValueError(f"Top-1 frame is missing target column {target_column!r}")
    scored = frame[["trip_id", "obs_idx", target_column]].copy()
    scored["_probability"] = probabilities
    winners = scored.loc[scored.groupby(["trip_id", "obs_idx"], sort=False)["_probability"].idxmax()]
    return float(winners[target_column].eq(1).mean()) if len(winners) else 0.0


def run_baseline_cross_validation(
    candidates: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run identical trip-grouped folds for Logistic Regression, GaussianNB and KNN.

    Returns ``(fold_metrics, out_of_fold_predictions, fitted_full_data_pipelines)``.
    All preprocessing is fit within a model/fold pipeline; unlabeled rows are omitted.
    """
    np.random.seed(random_seed)
    data = prepare_training_data(candidates)
    folds = make_group_folds(data, n_splits=n_splits)
    labelled = candidates.loc[candidates.is_target.isin([0, 1])].reset_index(drop=True)
    names = ["LogisticRegression", "GaussianNB", "KNN"]
    records: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    full_models: dict[str, Pipeline] = {}
    for model_name in names:
        oof_prob = np.full(len(data.y), np.nan)
        oof_pred = np.full(len(data.y), -1, dtype=int)
        for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
            y_train, y_val = data.y.iloc[train_idx], data.y.iloc[val_idx]
            if y_train.nunique() < 2:
                raise ValueError(f"{model_name} fold {fold_no} training split has one class; refusing invalid fit")
            if y_val.nunique() < 2:
                raise ValueError(
                    f"{model_name} fold {fold_no} validation split has one class; "
                    "binary macro F1 would not cover both classes"
                )
            k_fold = min(15, len(train_idx))
            estimator = build_baseline_pipeline(model_name, n_neighbors=k_fold)
            start = perf_counter()
            estimator.fit(data.X.iloc[train_idx], y_train)
            train_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = _positive_probability(estimator, data.X.iloc[val_idx])
            inference_seconds = perf_counter() - start
            predictions = (probabilities >= 0.5).astype(int)
            oof_prob[val_idx], oof_pred[val_idx] = probabilities, predictions
            val_frame = labelled.iloc[val_idx]
            records.append({
                "fold": fold_no, "model_name": model_name,
                "training_time_sec": train_seconds, "inference_time_sec": inference_seconds,
                "candidate_macro_f1": compute_macro_f1(y_val, predictions),
                "candidate_accuracy": float(accuracy_score(y_val, predictions)),
                "precision": float(precision_score(y_val, predictions, zero_division=0)),
                "recall": float(recall_score(y_val, predictions, zero_division=0)),
                "log_loss": compute_log_loss_safe(y_val, probabilities),
                "observation_top1_accuracy": observation_top1_accuracy(val_frame, probabilities),
                "n_train_trips": int(data.groups.iloc[train_idx].nunique()),
                "n_validation_trips": int(data.groups.iloc[val_idx].nunique()),
                "n_train_candidates": int(len(train_idx)), "n_validation_candidates": int(len(val_idx)),
                "configuration": json.dumps(estimator.named_steps["classifier"].get_params(), default=str, sort_keys=True),
            })
        if np.isnan(oof_prob).any() or (oof_pred < 0).any():
            raise RuntimeError(f"incomplete OOF predictions for {model_name}")
        oof = labelled[IDENTIFIER_COLUMNS + ["is_target"]].copy()
        oof["model_name"] = model_name
        oof["pred_prob"] = oof_prob
        oof["pred_label"] = oof_pred
        oof_parts.append(oof)
        # Preserve a deployable preprocessing + estimator pipeline trained only on train.
        full_model = build_baseline_pipeline(model_name, n_neighbors=min(15, len(data.y)))
        full_model.fit(data.X, data.y)
        full_models[model_name] = full_model
    return pd.DataFrame(records), pd.concat(oof_parts, ignore_index=True), full_models


def score_student_candidates(
    model: Pipeline, student_inputs: pd.DataFrame, feature_columns: list[str],
    student_labels: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Score student inputs without labels, then optionally join labels for evaluation."""
    missing = [c for c in feature_columns if c not in student_inputs]
    if missing:
        raise ValueError(f"student input is missing training features: {missing}")
    validation_columns = CANDIDATE_BASE_COLS + [
        c for c in feature_columns if c not in CANDIDATE_BASE_COLS
    ]
    absent_contract_columns = [c for c in validation_columns if c not in student_inputs]
    if absent_contract_columns:
        raise ValueError(f"student candidate table is missing contract columns: {absent_contract_columns}")
    validate_candidate_df(student_inputs[validation_columns])
    identifiers = [c for c in IDENTIFIER_COLUMNS if c in student_inputs]
    scored = student_inputs[identifiers].copy()
    features = student_inputs[feature_columns].replace([np.inf, -np.inf], np.nan)
    scored["pred_prob"] = _positive_probability(model, features)
    scored["pred_label"] = (scored.pred_prob >= 0.5).astype(int)
    evaluated = None
    if student_labels is not None:
        join_keys = ["trip_id", "obs_idx", "candidate_arc_id"]
        if student_labels.duplicated(join_keys).any() or scored.duplicated(join_keys).any():
            raise ValueError("student candidate and label keys must be unique")
        evaluated = scored.merge(student_labels[join_keys + ["reference_arc_id", "label"]],
                                 on=join_keys, how="left", validate="one_to_one")
    return scored, evaluated


def run_training_from_csv(
    train_csv: str | Path,
    output_dir: str | Path,
    *,
    student_input_csv: str | Path | None = None,
    student_labels_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible command helper writing OOF metrics, predictions and fitted models."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(train_csv, dtype={"trip_id": str}, low_memory=False)
    metrics, oof, models = run_baseline_cross_validation(candidates)
    metrics.to_csv(out / "section21_fold_metrics.csv", index=False)
    oof.to_csv(out / "section21_oof_predictions.csv", index=False)
    data_report = prepare_training_data(candidates).report
    data_report.update({
        "cv_strategy": "GroupKFold(n_splits=5), grouped by trip_id; identical folds for all models",
        "random_seed": 42,
        "models": MODEL_DESCRIPTIONS,
    })
    (out / "section21_training_data_report.json").write_text(json.dumps(data_report, indent=2))
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")
    if student_input_csv:
        student_inputs = pd.read_csv(student_input_csv, dtype={"trip_id": str}, low_memory=False)
        labels = pd.read_csv(student_labels_csv, dtype={"trip_id": str}, low_memory=False) if student_labels_csv else None
        features = prepare_training_data(candidates).feature_columns
        student_metrics = []
        student_predictions = []
        for name, model in models.items():
            scored, evaluated = score_student_candidates(model, student_inputs, features, labels)
            scored["model_name"] = name
            student_predictions.append(scored)
            if evaluated is not None:
                evaluated["model_name"] = name
                student_metrics.append(_student_metrics(name, evaluated))
        pd.concat(student_predictions, ignore_index=True).to_csv(out / "section21_student_predictions.csv", index=False)
        if student_metrics:
            pd.DataFrame(student_metrics).to_csv(out / "section21_student_metrics.csv", index=False)
    return metrics, oof


def _student_metrics(model_name: str, evaluated: pd.DataFrame) -> dict[str, Any]:
    valid = evaluated.label.isin([0, 1])
    frame = evaluated.loc[valid]
    probabilities = frame.pred_prob.to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return {"model_name": model_name, "candidate_rows": len(frame),
            "candidate_macro_f1": compute_macro_f1(frame.label, predictions),
            "candidate_accuracy": float(accuracy_score(frame.label, predictions)) if len(frame) else float("nan"),
            "precision": float(precision_score(frame.label, predictions, zero_division=0)) if len(frame) else float("nan"),
            "recall": float(recall_score(frame.label, predictions, zero_division=0)) if len(frame) else float("nan"),
            "log_loss": compute_log_loss_safe(frame.label, probabilities),
            "observation_top1_accuracy": observation_top1_accuracy(
                frame, probabilities, target_column="label") if len(frame) else float("nan")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/processed/train_candidates.csv")
    parser.add_argument("--output-dir", default="artifacts/logs/section21")
    parser.add_argument("--student-input-csv")
    parser.add_argument("--student-labels-csv")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        train_geometry = Path(args.data_dir) / "intermediate" / "candidate_segments_train.csv"
        if not train_geometry.exists():
            raise FileNotFoundError(f"training features and candidate geometry missing: {train_csv}, {train_geometry}")
        build_feature_table_from_candidate_geometry(train_geometry, train_csv, data_dir=args.data_dir, partition="train")
    student_input = args.student_input_csv
    student_labels = args.student_labels_csv
    if not student_input:
        student_geometry = Path(args.data_dir) / "intermediate" / "candidate_segments_student_test.csv"
        if student_geometry.exists():
            student_feature_file = Path(args.output_dir) / "student_test_feature_inputs.csv"
            build_feature_table_from_candidate_geometry(student_geometry, student_feature_file,
                                                        data_dir=args.data_dir, partition="student_test")
            student_input = str(student_feature_file)
            sidecar = Path(args.data_dir) / "intermediate" / "candidate_labels_student_test.csv"
            student_labels = str(sidecar) if sidecar.exists() else None
    run_training_from_csv(train_csv, args.output_dir, student_input_csv=student_input,
                          student_labels_csv=student_labels)
