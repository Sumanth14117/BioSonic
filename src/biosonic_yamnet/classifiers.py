"""Leakage-safe classifier pipelines and training-only grouped CV."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def make_group_cv_splits(metadata: pd.DataFrame, n_splits: int = config.N_SPLITS, seed: int = config.SEED):
    if metadata.filename.duplicated().any():
        raise ValueError("Duplicate recording IDs in classifier metadata")
    if set(metadata.primary_label.astype(str)) != set(config.WORKING_LABELS):
        raise ValueError("Training CV metadata must contain all fixed eight classes")
    labels = metadata.primary_label.astype(str).to_numpy()
    strata = (metadata.primary_label.astype(str) + " × " + metadata.collection.astype(str)).to_numpy()
    groups = metadata.group_id.astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(metadata), dtype=np.uint8), strata, groups))
    all_classes, all_strata = set(labels), set(strata)
    for fold, (train_index, validation_index) in enumerate(splits, start=1):
        if set(groups[train_index]) & set(groups[validation_index]):
            raise AssertionError(f"Author/content group leakage in training fold {fold}")
        if set(labels[train_index]) != all_classes or set(labels[validation_index]) != all_classes:
            raise ValueError(f"Fold {fold} lacks one or more species classes")
        if set(strata[train_index]) != all_strata or set(strata[validation_index]) != all_strata:
            raise ValueError(f"Fold {fold} lacks one or more primary_label × collection strata")
    return splits


def pipeline_and_grid(model_name: str, seed: int = config.SEED):
    if model_name == "logistic_regression":
        return (Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(
            max_iter=4000, class_weight="balanced", random_state=seed))]),
            {"model__C": [0.1, 1.0, 10.0]})
    if model_name == "svm":
        return (Pipeline([("scaler", StandardScaler()), ("model", SVC(
            kernel="rbf", class_weight="balanced", decision_function_shape="ovr",
            probability=False, random_state=seed))]),
            {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale"]})
    if model_name == "random_forest":
        return (Pipeline([("model", RandomForestClassifier(
            class_weight="balanced_subsample", random_state=seed, n_jobs=4))]),
            {"model__n_estimators": [200, 300], "model__max_depth": [None, 24],
             "model__min_samples_leaf": [1, 2]})
    raise KeyError(f"Unknown M5 classifier: {model_name}")


def _score_matrix(estimator, x: np.ndarray) -> tuple[np.ndarray, str]:
    model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    if hasattr(model, "predict_proba"):
        scores, kind = estimator.predict_proba(x), "probability"
    else:
        scores, kind = estimator.decision_function(x), "decision_score"
    scores = np.asarray(scores)
    classes = [str(value) for value in model.classes_]
    if scores.ndim != 2 or set(classes) != set(config.WORKING_LABELS):
        raise ValueError("Estimator output shape or class map differs from the fixed eight-class mapping")
    scores = scores[:, [classes.index(label) for label in config.WORKING_LABELS]]
    if not np.isfinite(scores).all():
        raise ValueError("Classifier produced non-finite class scores")
    return scores, kind


def classification_metrics(estimator, x: np.ndarray, y_true: np.ndarray, predictions=None) -> dict:
    y_true = np.asarray(y_true).astype(str)
    pred = np.asarray(predictions).astype(str) if predictions is not None else estimator.predict(x).astype(str)
    if pred.shape != y_true.shape:
        raise ValueError("Prediction and target arrays are not aligned")
    scores, score_kind = _score_matrix(estimator, x)
    aucs = [float(roc_auc_score(y_true == label, scores[:, index]))
            for index, label in enumerate(config.WORKING_LABELS)]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(precision_score(y_true, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=list(config.WORKING_LABELS), average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(np.mean(aucs)), "score_kind": score_kind,
    }


def tune_with_training_cv(x: np.ndarray, metadata: pd.DataFrame, progress=None) -> dict:
    if x.shape != (len(metadata), config.EMBEDDING_DIM) or not np.isfinite(x).all():
        raise ValueError("Training embeddings have invalid shape or non-finite values")
    splits = make_group_cv_splits(metadata)
    y = metadata.primary_label.astype(str).to_numpy()
    candidate_rows, fold_rows = [], []
    started_all = time.perf_counter()
    total_candidates, completed = 14, 0
    for model_name in ("logistic_regression", "svm", "random_forest"):
        base, grid = pipeline_and_grid(model_name)
        for params in ParameterGrid(grid):
            candidate_started = time.perf_counter()
            fold_scores = []
            for fold, (train_index, validation_index) in enumerate(splits, start=1):
                estimator = base.set_params(**params)
                estimator.fit(x[train_index], y[train_index])
                metrics = classification_metrics(estimator, x[validation_index], y[validation_index])
                fold_rows.append({"model": model_name, "candidate_parameters": json.dumps(params, sort_keys=True),
                                  "fold": fold, "train_recordings": len(train_index),
                                  "validation_recordings": len(validation_index), **metrics})
                fold_scores.append(metrics)
            candidate_rows.append({
                "model": model_name, "candidate_parameters": json.dumps(params, sort_keys=True),
                "mean_macro_f1": float(np.mean([item["macro_f1"] for item in fold_scores])),
                "std_macro_f1": float(np.std([item["macro_f1"] for item in fold_scores], ddof=1)),
                "mean_accuracy": float(np.mean([item["accuracy"] for item in fold_scores])),
                "mean_precision_macro": float(np.mean([item["precision_macro"] for item in fold_scores])),
                "mean_recall_macro": float(np.mean([item["recall_macro"] for item in fold_scores])),
                "mean_weighted_f1": float(np.mean([item["weighted_f1"] for item in fold_scores])),
                "mean_roc_auc_ovr_macro": float(np.mean([item["roc_auc_ovr_macro"] for item in fold_scores])),
                "candidate_fit_seconds": time.perf_counter() - candidate_started, "folds": len(fold_scores),
            })
            completed += 1
            if progress:
                progress(completed, total_candidates, model_name, params, candidate_rows[-1]["mean_macro_f1"])
    summary = pd.DataFrame(candidate_rows).sort_values(
        ["mean_macro_f1", "model", "candidate_parameters"],
        ascending=[False, True, True], kind="stable").reset_index(drop=True)
    selected = summary.iloc[0].to_dict()
    selected_model = str(selected["model"])
    selected_params = json.loads(str(selected["candidate_parameters"]))
    estimator, _ = pipeline_and_grid(selected_model)
    estimator.set_params(**selected_params)
    final_start = time.perf_counter()
    estimator.fit(x, y)
    return {
        "estimator": estimator, "selected_model": selected_model, "selected_params": selected_params,
        "selected_cv": selected, "candidate_rows": candidate_rows, "fold_rows": fold_rows,
        "cv_total_seconds": time.perf_counter() - started_all,
        "final_fit_seconds": time.perf_counter() - final_start, "cv_splits": len(splits),
    }
