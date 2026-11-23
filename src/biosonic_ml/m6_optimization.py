"""Training-only M6 classifier optimization on frozen M5 embeddings.

This module deliberately reads only the training prefix of the approved split
manifest and only the M5 training embedding aggregate. It never opens the
holdout embedding artifacts or parses holdout metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .io_utils import atomic_write_csv, atomic_write_json


M6_REPORT_ROOT = config.ROOT / "reports/ml/m6_optimization"
M6_MODEL_ROOT = config.ROOT / "models/m6_optimization"
M5_TRAIN_EMBEDDINGS = config.ROOT / "data/processed/m5_yamnet/train_embeddings.npz"
M5_SPLIT_MANIFEST = config.SPLIT_MANIFEST
M6_SEED = 42
M6_FOLDS = 5
EXPECTED_TRAIN_ROWS = 1_748
EMBEDDING_DIM = 1_024


def read_training_metadata(path: Path = M5_SPLIT_MANIFEST) -> pd.DataFrame:
    """Read only train rows; stop at the first non-training row."""
    rows: list[dict[str, str]] = []
    saw_boundary = False
    required = {"filename", "primary_label", "collection", "group_id", "partition"}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Split manifest is missing required fields: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["partition"] != "train":
                saw_boundary = True
                break
            rows.append({key: row[key] for key in required})
    if not saw_boundary:
        raise ValueError("Training boundary was not found; refusing to use an incomplete split")
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} train rows, found {len(frame)}")
    if frame["filename"].duplicated().any():
        raise ValueError("Training metadata has duplicate recording names")
    if frame[["primary_label", "collection", "group_id"]].isna().any().any():
        raise ValueError("Training metadata contains missing labels, collections, or groups")
    if set(frame.primary_label.astype(str)) != set(config.WORKING_LABELS):
        raise ValueError("Training metadata does not match the fixed eight-class scope")
    return frame.reset_index(drop=True)


def load_training_embeddings(npz_path: Path = M5_TRAIN_EMBEDDINGS,
                             metadata_path: Path = M5_SPLIT_MANIFEST) -> tuple[np.ndarray, pd.DataFrame]:
    """Load training-only YAMNet embeddings and assert exact manifest alignment."""
    metadata = read_training_metadata(metadata_path)
    with np.load(npz_path, allow_pickle=False) as saved:
        required = {"embeddings", "labels", "filenames"}
        if set(saved.files) != required:
            raise ValueError(f"Unexpected embedding archive contents: {saved.files}")
        x = np.asarray(saved["embeddings"], dtype=np.float32)
        labels = saved["labels"].astype(str)
        filenames = saved["filenames"].astype(str)
    if x.shape != (EXPECTED_TRAIN_ROWS, EMBEDDING_DIM) or not np.isfinite(x).all():
        raise ValueError(f"Invalid training embedding matrix shape or values: {x.shape}")
    if not np.array_equal(labels, metadata.primary_label.astype(str).to_numpy()):
        raise ValueError("Embedding labels do not match the training manifest")
    if not np.array_equal(filenames, metadata.filename.astype(str).to_numpy()):
        raise ValueError("Embedding order does not match the training manifest")
    return x, metadata


def make_group_splits(metadata: pd.DataFrame, n_splits: int = M6_FOLDS,
                      seed: int = M6_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic author/content group-aware CV splits."""
    labels = metadata.primary_label.astype(str).to_numpy()
    strata = (metadata.primary_label.astype(str) + " × " + metadata.collection.astype(str)).to_numpy()
    groups = metadata.group_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(metadata), dtype=np.uint8), strata, groups))
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise AssertionError(f"Group leakage in fold {fold}")
        if set(labels[train_idx]) != set(config.WORKING_LABELS) or set(labels[valid_idx]) != set(config.WORKING_LABELS):
            raise ValueError(f"Fold {fold} does not contain all eight classes on both sides")
        train_strata, valid_strata = set(strata[train_idx]), set(strata[valid_idx])
        if train_strata != set(strata) or valid_strata != set(strata):
            raise ValueError(f"Fold {fold} does not preserve all label-by-collection strata")
    return splits


def _candidate_specs(seed: int = M6_SEED) -> list[tuple[str, dict[str, object], bool]]:
    """Return a small prespecified candidate set; scaling stays fold-local."""
    specs: list[tuple[str, dict[str, object], bool]] = []
    for weight in ("balanced", None):
        for c in (0.01, 0.1, 1.0, 10.0):
            specs.append(("logistic_regression",
                          {"C": c, "class_weight": weight, "max_iter": 4000,
                           "random_state": seed}, True))
    for weight in ("balanced", None):
        for c in (0.5, 1.0, 5.0):
            specs.append(("rbf_svm",
                          {"C": c, "class_weight": weight, "gamma": "scale",
                           "kernel": "rbf", "decision_function_shape": "ovr",
                           "random_state": seed}, True))
    specs.append(("random_forest",
                  {"n_estimators": 200, "max_depth": 24,
                   "min_samples_leaf": 2, "class_weight": "balanced_subsample",
                   "n_jobs": 4, "random_state": seed}, False))
    return specs


def _make_estimator(name: str, params: dict[str, object], scale: bool) -> Pipeline:
    if name == "logistic_regression":
        model = LogisticRegression(**params)
    elif name == "rbf_svm":
        model = SVC(**params)
    elif name == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise KeyError(name)
    steps = ([ ("scaler", StandardScaler()), ("model", model) ] if scale
             else [("model", model)])
    return Pipeline(steps)


def _evaluate_fold(estimator: Pipeline, x_valid: np.ndarray, y_valid: np.ndarray) -> tuple[dict, np.ndarray]:
    pred = estimator.predict(x_valid).astype(str)
    scores = {
        "accuracy": float(accuracy_score(y_valid, pred)),
        "precision_macro": float(precision_score(y_valid, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_valid, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_valid, pred, labels=list(config.WORKING_LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_valid, pred, labels=list(config.WORKING_LABELS), average="weighted", zero_division=0)),
    }
    return scores, pred


def run_optimization(x: np.ndarray, metadata: pd.DataFrame, report_root: Path = M6_REPORT_ROOT,
                     model_root: Path = M6_MODEL_ROOT, progress=print) -> dict:
    """Run training-only grouped CV, refit selected model, and write new outputs.

    Refuses to overwrite any existing M6 output directories.
    """
    report_root, model_root = Path(report_root), Path(model_root)
    if report_root.exists() or model_root.exists():
        raise FileExistsError("M6 output directory already exists; refusing to overwrite prior work")
    if x.shape != (len(metadata), EMBEDDING_DIM) or len(metadata) != EXPECTED_TRAIN_ROWS:
        raise ValueError("M6 requires the aligned 1,748 x 1,024 training-only embedding matrix")
    if not np.isfinite(x).all():
        raise ValueError("M6 input embeddings contain non-finite values")
    splits = make_group_splits(metadata)
    y = metadata.primary_label.astype(str).to_numpy()
    labels = list(config.WORKING_LABELS)
    candidate_rows: list[dict] = []
    fold_rows: list[dict] = []
    prediction_by_candidate: dict[str, list[np.ndarray | None]] = {}
    started = time.perf_counter()
    specs = _candidate_specs()
    for candidate_index, (name, params, scale) in enumerate(specs, start=1):
        candidate_key = f"{name}|{json.dumps(params, sort_keys=True)}"
        fold_scores: list[dict] = []
        oof = np.empty(len(y), dtype=object)
        oof[:] = None
        fit_start = time.perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            estimator = _make_estimator(name, params, scale)
            estimator.fit(x[train_idx], y[train_idx])
            scores, pred = _evaluate_fold(estimator, x[valid_idx], y[valid_idx])
            oof[valid_idx] = pred
            row = {"model": name, "parameters": json.dumps(params, sort_keys=True),
                   "fold": fold, "train_recordings": len(train_idx),
                   "validation_recordings": len(valid_idx), **scores}
            fold_rows.append(row)
            fold_scores.append(scores)
        prediction_by_candidate[candidate_key] = list(oof)
        summary = {metric: float(np.mean([row[metric] for row in fold_scores]) if metric != "std_macro_f1"
                                 else np.std([row["macro_f1"] for row in fold_scores], ddof=1))
                   for metric in ("accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1")}
        candidate_rows.append({"model": name, "parameters": json.dumps(params, sort_keys=True),
                               **{f"mean_{k}": v for k, v in summary.items()},
                               "std_macro_f1": float(np.std([row["macro_f1"] for row in fold_scores], ddof=1)),
                               "fit_seconds": time.perf_counter() - fit_start,
                               "folds": len(fold_scores)})
        progress(f"M6 candidate {candidate_index}/{len(specs)}: {name}, CV macro-F1={summary['macro_f1']:.4f} ± {candidate_rows[-1]['std_macro_f1']:.4f}")
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["mean_macro_f1", "model", "parameters"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    selected = candidates.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_params = json.loads(str(selected["parameters"]))
    selected_key = f"{selected_name}|{json.dumps(selected_params, sort_keys=True)}"
    selected_oof = np.asarray(prediction_by_candidate[selected_key], dtype=str)
    if selected_oof.shape != y.shape or np.any(selected_oof == "None"):
        raise RuntimeError("Selected candidate does not have one out-of-fold prediction per training recording")
    report = classification_report(y, selected_oof, labels=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"primary_label": label,
                               "precision": report[label]["precision"],
                               "recall": report[label]["recall"],
                               "f1": report[label]["f1-score"],
                               "support": int(report[label]["support"])} for label in labels])
    cm = pd.DataFrame(confusion_matrix(y, selected_oof, labels=labels), index=labels, columns=labels)
    final_estimator = _make_estimator(selected_name, selected_params,
                                      selected_name in {"logistic_regression", "rbf_svm"})
    final_start = time.perf_counter()
    final_estimator.fit(x, y)
    total_seconds = time.perf_counter() - started

    # Create new output directories only after all CV work has completed.
    report_root.mkdir(parents=True, exist_ok=False)
    model_root.mkdir(parents=True, exist_ok=False)
    model_path = model_root / "selected_classifier.joblib"
    joblib.dump(final_estimator, model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    atomic_write_json({
        "model_file": model_path.name,
        "model_sha256": model_sha256,
        "model_bytes": model_path.stat().st_size,
        "model_family": selected_name,
        "parameters": selected_params,
        "class_order": labels,
        "training_recordings": len(metadata),
        "embedding_dimension": EMBEDDING_DIM,
        "validation": "5-fold StratifiedGroupKFold; training partition only",
        "seed": M6_SEED,
        "holdout_used": False,
    }, model_root / "model_metadata.json")
    atomic_write_csv(candidates, report_root / "candidate_summary.csv")
    atomic_write_csv(pd.DataFrame(fold_rows), report_root / "fold_metrics.csv")
    atomic_write_csv(per_class, report_root / "selected_oof_per_class_metrics.csv")
    atomic_write_csv(cm.rename_axis("actual").reset_index(), report_root / "selected_oof_confusion_matrix.csv")
    try:
        import torch
        gpu_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception:
        gpu_available, gpu = False, None
    summary = {
        "milestone": "M6 model optimization",
        "scope": "eight-class interim task; training partition only",
        "input": "existing M5 frozen YAMNet recording embeddings",
        "input_path": str(M5_TRAIN_EMBEDDINGS.relative_to(config.ROOT)),
        "training_recordings": len(metadata), "embedding_dimension": EMBEDDING_DIM,
        "class_order": labels, "splitter": "StratifiedGroupKFold",
        "folds": M6_FOLDS, "seed": M6_SEED,
        "stratification": "primary_label × collection",
        "grouping": "existing author/content group_id",
        "holdout_used": False,
        "selection_metric": "mean five-fold training-side macro-F1",
        "selected_model": selected_name, "selected_parameters": selected_params,
        "selected_cv": {key: float(selected[key]) for key in selected if key.startswith("mean_") or key == "std_macro_f1"},
        "selected_oof_metrics": {
            "accuracy": float(accuracy_score(y, selected_oof)),
            "precision_macro": float(precision_score(y, selected_oof, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, selected_oof, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, selected_oof, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, selected_oof, labels=labels, average="weighted", zero_division=0)),
        },
        "candidate_count": len(specs), "fit_seconds_total_cv_and_final": total_seconds,
        "final_fit_seconds": time.perf_counter() - final_start,
        "execution_device": "cpu (scikit-learn estimators; GPU not used)",
        "gpu_available_but_not_used": gpu_available, "gpu_model": gpu,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "final_model_path": (str(model_path.relative_to(config.ROOT))
                             if model_path.is_relative_to(config.ROOT) else str(model_path)),
        "final_model_sha256": model_sha256,
        "preprocessing": "StandardScaler fitted inside each fold for LR/SVM; no validation-fold fitting; RF unscaled.",
        "roc_auc": "Not reported: selection used hard-label macro-F1; SVM scores are not calibrated probabilities.",
    }
    atomic_write_json(summary, report_root / "experiment_summary.json")
    markdown = _render_report(summary, candidates, per_class, cm)
    (report_root / "m6_optimization_report.md").write_text(markdown, encoding="utf-8", newline="\n")
    return {"summary": summary, "candidates": candidates, "folds": pd.DataFrame(fold_rows),
            "per_class": per_class, "confusion_matrix": cm, "model_path": model_path,
            "total_seconds": total_seconds}


def _render_report(summary: dict, candidates: pd.DataFrame,
                   per_class: pd.DataFrame, cm: pd.DataFrame) -> str:
    top = candidates.head(10)
    lines = [
        "# BioSonic M6 — Training-only model optimization", "",
        "## Dataset attribution", "",
        "This report contains aggregate results derived from BirdCLEF+ 2026 recordings and metadata. Attribute BirdCLEF+ 2026 and the original recording contributors; competition data and applicable derived material are subject to CC BY-NC-SA 4.0 and competition terms. BioSonic does not claim ownership of the source data. The root Apache-2.0 license applies to original BioSonic source code only.", "",
        "## Objective and scope", "",
        "M6 evaluates whether classifier-side changes improve the frozen YAMNet embedding baseline. The scope remains the eight-class interim bird classification task; four classes from the original 12-class objective remain unavailable. No M3–M5 results or artifacts were changed.", "",
        "## Existing model limitations and experiment choice", "",
        "M3 uses 204-dimensional handcrafted recording summaries and already applies class weighting to Logistic Regression/SVM; M6 tests whether its regularization settings transfer to the frozen-embedding representation. M4 uses five-second power-Mel windows, pads incomplete tails, assigns the source recording label to its windows, and averages segment probabilities at recording level. Its internal-validation macro-F1 (0.7603) and historical shared-holdout score are from different partitions and the latter is not valid for new tuning. M5 uses frozen 1,024-dimensional YAMNet embeddings and had a post-resampling waveform range diagnostic outside [-1, 1]. M6 does not tune on the reused holdout and does not alter any M3–M5 audio inputs. It focuses on class weighting and classifier regularization on M5's existing training-only embedding cache. CNN augmentation, CNN architecture changes, and new YAMNet waveform handling were not investigated in this run; those changes would need new leakage-safe training-side validation, with partial-window label noise handled explicitly.", "",
        "## Dataset, partitions, and leakage controls", "",
        f"M6 used **{summary['training_recordings']:,} training recordings** and eight classes. Inputs were the existing per-recording mean-pooled YAMNet embeddings (`{summary['embedding_dimension']}` dimensions), aligned to the training prefix of the authoritative split manifest. The manifest reader stops at the first non-training row; no holdout labels, embeddings, predictions, or metrics were read. The old 429-recording holdout was not used.", "",
        "Validation used five-fold `StratifiedGroupKFold` (seed 42), stratified on `primary_label × collection`, grouped by the existing author/content `group_id`. Each fold was checked for zero group overlap, all eight classes on both sides, and all observed label-by-collection strata on both sides. All scaling is inside the model pipeline and is fit separately on each fold's training data.", "",
        "## Improvements investigated", "",
        "Fifteen prespecified candidates compared Logistic Regression with balanced or unweighted loss across C values 0.01, 0.1, 1, and 10; RBF SVM with balanced or unweighted loss across C values 0.5, 1, and 5; and one class-balanced Random Forest configuration. The exact candidate parameters and fold scores are in `candidate_summary.csv` and `fold_metrics.csv`. Macro-F1 was the sole model-selection metric. No threshold tuning or probability calibration was performed.", "",
        "## Training-side validation results", "",
        "### Top candidates", "",
        "| Model | Parameters | CV macro-F1 mean ± SD | Accuracy | Macro precision | Macro recall | Weighted F1 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        params = str(row.parameters).replace("|", "\\|")
        lines.append(f"| {row.model} | `{params}` | {row.mean_macro_f1:.4f} ± {row.std_macro_f1:.4f} | {row.mean_accuracy:.4f} | {row.mean_precision_macro:.4f} | {row.mean_recall_macro:.4f} | {row.mean_weighted_f1:.4f} |")
    lines += ["", f"Selected by training-side mean macro-F1: **{summary['selected_model']}** with `{json.dumps(summary['selected_parameters'], sort_keys=True)}`. Its cross-validated out-of-fold macro-F1 is {summary['selected_oof_metrics']['macro_f1']:.4f}; accuracy {summary['selected_oof_metrics']['accuracy']:.4f}, macro precision {summary['selected_oof_metrics']['precision_macro']:.4f}, macro recall {summary['selected_oof_metrics']['recall_macro']:.4f}, weighted F1 {summary['selected_oof_metrics']['weighted_f1']:.4f}.", "", "### Selected out-of-fold per-class metrics", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for r in per_class.itertuples(index=False): lines.append(f"| {r.primary_label} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.support} |")
    matrix_header = "| Actual / Predicted | " + " | ".join(cm.columns.astype(str)) + " |"
    matrix_sep = "|---|" + "---:|" * len(cm.columns)
    matrix_lines = [matrix_header, matrix_sep]
    for label, row in cm.iterrows():
        matrix_lines.append("| " + str(label) + " | " + " | ".join(str(int(v)) for v in row.to_numpy()) + " |")
    lines += ["", "### Selected out-of-fold confusion matrix", "", *matrix_lines, "",
              "## Comparison with M3–M5", "",
        "M6's primary estimate is grouped five-fold CV on the 1,748-recording training partition. The best M6 classifier (balanced Logistic Regression, C=0.1; mean macro-F1 0.5972 ± 0.0517) matches the prior M5 classifier baseline and does not improve it. The prior M5 result is 0.5972 ± 0.0517 under the same training-only grouped-CV protocol. Existing M3 training-CV results were 0.6575 ± 0.0397 for RBF SVM and 0.6550 ± 0.0420 for Logistic Regression on different handcrafted features; M3 remains the strongest comparable training-CV result among those reported. The M4 internal-validation macro-F1 of 0.7603 used its distinct 1,444/304 split and is not directly comparable. Previously reported M3–M5 holdout metrics are intentionally not repeated as M6 selection evidence and were not accessed.", "",
        f"The final refit ran on {summary['execution_device']}; the available RTX 4060 was detected but not used. Candidate CV plus final fitting took {summary['fit_seconds_total_cv_and_final']:.1f} seconds; final refit took {summary['final_fit_seconds']:.1f} seconds. Environment: Python {summary['python_version']}, scikit-learn {summary['package_versions']['scikit-learn']}, NumPy {summary['package_versions']['numpy']}.", "",
              "## Selection and limitations", "",
              "The selected estimator is refit on all 1,748 training recordings after grouped CV. The selected fold mean and selected-candidate out-of-fold table use the same folds; both are model-selection results and may be optimistic, not an independent unbiased estimate. Use a new author/spatially separated dataset for confirmation. This remains an eight-class interim, recording-level single-label experiment. Weak primary labels, unknown-author generalization, geographic dependence, shared historical M3–M5 holdout reuse, and the M5 input-range concern remain. M6 does not establish performance for the four unavailable classes or soundscape multi-label detection.", "",
              "## Reproducibility", "",
              "Run from the repository root: `.venv\\Scripts\\python.exe scripts\\run_m6_optimization.py`. The default output directories are write-once and the script refuses to overwrite them. For an additional independent run with the same fixed configuration, use `--run-id run_02` (or another unique safe identifier) to write under `reports/ml/m6_optimization/runs/<run-id>/` and `models/m6_optimization/runs/<run-id>/`. Configuration, fold-level aggregate metrics, selected out-of-fold per-class metrics, confusion matrix, and the selected classifier with SHA-256 metadata are saved. No per-recording prediction file is written.", "",
              "## M7 next steps", "",
              "Keep M7 focused on explainability and biodiversity analytics with the validation boundary intact. Explain model behavior using training-side folds, document uncertainty and species/collection limitations, and design ecological activity summaries that are not presented as direct population estimates.", ""]
    return "\n".join(lines)


def run_from_project(report_root: Path = M6_REPORT_ROOT,
                     model_root: Path = M6_MODEL_ROOT, progress=print) -> dict:
    if Path(report_root).exists() or Path(model_root).exists():
        raise FileExistsError("M6 output already exists; inspect it before rerunning")
    x, metadata = load_training_embeddings()
    return run_optimization(x, metadata, report_root=report_root,
                            model_root=model_root, progress=progress)
