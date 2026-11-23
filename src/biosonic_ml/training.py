import json
import platform
import time
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from . import config
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def make_group_cv_splits(metadata: pd.DataFrame, n_splits=config.N_SPLITS, seed=config.SEED):
    y = metadata.primary_label.astype(str).to_numpy()
    strat = (metadata.primary_label.astype(str) + " × " + metadata.collection.astype(str)).to_numpy()
    groups = metadata.group_id.astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(metadata)), strat, groups))
    if len(set(y)) != len(config.WORKING_LABELS):
        raise ValueError("Training partition must contain all eight labels")
    for tr, va in splits:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Group leakage in training cross-validation")
        if set(y[tr]) != set(y) or set(y[va]) != set(y):
            raise ValueError("A CV fold lacks a training or validation class")
    return splits


def _estimators(seed):
    return {
        "logistic_regression": (Pipeline([("scaler", StandardScaler()),
                                          ("model", LogisticRegression(max_iter=3000, random_state=seed))]),
                                {"model__C": [0.1, 1.0, 10.0], "model__class_weight": ["balanced"]}),
        "random_forest": (Pipeline([("model", RandomForestClassifier(random_state=seed, n_jobs=-1))]),
                          {"model__n_estimators": [200, 300], "model__max_depth": [None, 24],
                           "model__min_samples_leaf": [1, 2], "model__class_weight": ["balanced_subsample"]}),
        "svm": (Pipeline([("scaler", StandardScaler()),
                          ("model", SVC(kernel="rbf", decision_function_shape="ovr"))]),
                {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale"], "model__class_weight": ["balanced"]}),
    }


def _evaluate(model, X, y, labels, predictions=None):
    pred = model.predict(X) if predictions is None else predictions
    result = {"accuracy": float(accuracy_score(y, pred)),
              "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
              "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
              "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
              "precision_weighted": float(precision_score(y, pred, average="weighted", zero_division=0)),
              "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0)),
              "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
              "predictions": pred,
              "classification_report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(y, pred, labels=labels)}
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
            class_order = list(estimator.classes_)
            auc_labels = sorted(labels)
            probs = probs[:, [class_order.index(label) for label in auc_labels]]
            result["roc_auc_ovr_macro"] = float(roc_auc_score(y, probs, labels=auc_labels, multi_class="ovr", average="macro"))
        except (ValueError, AttributeError):
            result["roc_auc_ovr_macro"] = None
    else:
        result["roc_auc_ovr_macro"] = None
    return result


def train_and_evaluate(X: np.ndarray, metadata: pd.DataFrame, model_dir=config.MODEL_ROOT,
                       report_dir=config.REPORT_ROOT, seed=config.SEED):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if len(X) != len(metadata) or not np.isfinite(X).all():
        raise ValueError("Features and metadata are inconsistent or non-finite")
    labels = list(config.WORKING_LABELS)
    train_mask = metadata.partition.eq("train").to_numpy()
    val_mask = metadata.partition.eq("validation").to_numpy()
    X_train, y_train = X[train_mask], metadata.loc[train_mask, "primary_label"].to_numpy()
    X_val, y_val = X[val_mask], metadata.loc[val_mask, "primary_label"].to_numpy()
    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    cv_splits = make_group_cv_splits(train_meta, seed=seed)
    searches, fitted = {}, {}
    for name, (pipeline, grid) in _estimators(seed).items():
        start = time.perf_counter()
        search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=cv_splits, refit=True,
                              n_jobs=1, error_score="raise", return_train_score=False)
        search.fit(X_train, y_train)
        cv_seconds = time.perf_counter() - start
        atomic_write_csv(pd.DataFrame(search.cv_results_), report_dir / f"{name}_cv_results.csv")
        searches[name] = (search, cv_seconds)
        fitted[name] = search.best_estimator_
    # The final holdout is opened for scoring only after every model and parameter
    # choice has been fixed using training-partition group CV.
    results, metric_payloads = [], {}
    for name, (search, cv_seconds) in searches.items():
        infer_start = time.perf_counter()
        predictions = search.best_estimator_.predict(X_val)
        inference_seconds = time.perf_counter() - infer_start
        holdout_start = time.perf_counter()
        metrics = _evaluate(search.best_estimator_, X_val, y_val, labels, predictions=predictions)
        metric_payloads[name] = metrics
        eval_seconds = time.perf_counter() - holdout_start
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(search.best_estimator_, model_path)
        details = {"model": name, "best_cv_macro_f1": float(search.best_score_),
                   "cv_fit_seconds": cv_seconds, "holdout_evaluation_seconds": eval_seconds,
                   "inference_seconds": inference_seconds,
                   "inference_ms_per_record": inference_seconds * 1000 / len(X_val),
                   "best_parameters": search.best_params_, "holdout": {k: v for k, v in metrics.items()
                       if k not in {"predictions", "classification_report", "confusion_matrix"}},
                   "roc_auc_note": "Multiclass OVR macro ROC-AUC when predict_proba is available; otherwise not supported by estimator configuration.",
                   "model_file": model_path.name}
        atomic_write_json(details, report_dir / f"{name}_results.json")
        atomic_write_csv(pd.DataFrame(metrics["classification_report"]).T.rename_axis("class").reset_index(),
                         report_dir / f"{name}_classification_report.csv")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        cm.to_csv(report_dir / f"{name}_confusion_matrix.csv", index=True)
        results.append({"model": name, "best_cv_macro_f1": float(search.best_score_), **details["holdout"],
                        "cv_fit_seconds": cv_seconds, "holdout_evaluation_seconds": eval_seconds,
                        "inference_seconds": inference_seconds,
                        "inference_ms_per_record": inference_seconds * 1000 / len(X_val),
                        "best_parameters": json.dumps(search.best_params_, sort_keys=True)})
    table = pd.DataFrame(results).sort_values(["best_cv_macro_f1", "model"], ascending=[False, True])
    atomic_write_csv(table, report_dir / "model_comparison.csv")
    winner = str(table.iloc[0].model)
    best_path = model_dir / "best_model.joblib"
    joblib.dump(fitted[winner], best_path)
    try:
        package_versions = {name: version(name) for name in ("numpy", "pandas", "librosa", "soundfile", "scikit-learn", "scipy", "matplotlib")}
    except PackageNotFoundError:
        package_versions = {"note": "One or more package versions could not be queried."}
    atomic_write_json({"seed": seed, "primary_selection_metric": "mean group-aware training CV macro-F1",
                       "best_model_by_training_cv": winner, "validation_used_only_for_final_evaluation": True,
                       "cv_splits": 5, "models": list(fitted), "class_labels": labels,
                       "feature_scaling": "StandardScaler is inside Logistic Regression and SVM sklearn pipelines; Random Forest unscaled.",
                       "tuning_scope": "Training partition only; final grouped holdout is not used for model or hyperparameter selection.",
                       "python_version": platform.python_version(), "package_versions": package_versions,
                       "best_model_artifact": str(best_path.relative_to(config.ROOT))},
                      report_dir / "training_config.json")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(table.model, table.f1_macro, color="#267a68")
    ax.set_ylabel("Final holdout macro F1")
    ax.set_title("Final holdout macro F1 (selection based on training CV)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(report_dir / "holdout_macro_f1.png", dpi=160)
    plt.close(fig)
    positions = np.arange(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, table.best_cv_macro_f1, width, label="Training CV mean macro F1", color="#4676a9")
    ax.bar(positions + width / 2, table.f1_macro, width, label="Final holdout macro F1", color="#d88345")
    ax.set_xticks(positions, table.model, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Training-only CV selection versus final holdout evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "cv_vs_holdout_macro_f1.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(labels))
    width = 0.8 / len(metric_payloads)
    for model_index, (name, metrics) in enumerate(metric_payloads.items()):
        f1s = [metrics["classification_report"][label]["f1-score"] for label in labels]
        ax.bar(positions - 0.4 + width / 2 + model_index * width, f1s, width,
               label=name.replace("_", " "))
    ax.set_xticks(positions, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class final holdout F1; model selection used training CV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "per_class_holdout_f1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(metric_payloads), figsize=(15, 5), sharey=True)
    for ax, (name, metrics) in zip(axes, metric_payloads.items()):
        cm = metrics["confusion_matrix"]
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(name.replace("_", " "))
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_xlabel("Predicted")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    axes[0].set_ylabel("True")
    fig.suptitle("Final holdout confusion matrices (model choice based on training CV)")
    fig.tight_layout()
    fig.savefig(report_dir / "holdout_confusion_matrices.png", dpi=160)
    plt.close(fig)
    split = pd.read_csv(config.SPLIT_MANIFEST)
    count_table = metadata.groupby("primary_label").agg(recordings=("filename", "size"),
                                                         author_groups=("author_group_id", "nunique")).reindex(labels).fillna(0).astype(int)
    count_table["train"] = metadata[metadata.partition.eq("train")].groupby("primary_label").size().reindex(labels).fillna(0).astype(int)
    count_table["validation"] = metadata[metadata.partition.eq("validation")].groupby("primary_label").size().reindex(labels).fillna(0).astype(int)
    count_table["percent"] = count_table.recordings / len(metadata) * 100
    count_table["unknown_authors"] = split.groupby("primary_label").unknown_author.sum().reindex(labels).fillna(0).astype(int)
    collections = metadata.groupby(["primary_label", "collection"]).size().unstack(fill_value=0).reindex(labels, fill_value=0)
    count_table["XC"] = collections.get("XC", 0)
    count_table["iNat"] = collections.get("iNat", 0)
    winner_report = metric_payloads[winner]["classification_report"]
    per_class = pd.DataFrame([{"primary_label": label, "precision": winner_report[label]["precision"],
                               "recall": winner_report[label]["recall"], "f1": winner_report[label]["f1-score"],
                               "support": int(winner_report[label]["support"])} for label in labels])
    lines = ["# BioSonic M3 — Eight-class interim experiment", "",
             "## Scope and dataset", "",
             "The intended M3 task has 12 bird species. At execution time, the authoritative manifest had 2,177 verified recordings; four approved labels (`swtman1`, `smbani`, `yecpar`, `spispi1`) had no verified local audio. This is an eight-class interim benchmark, not a 12-class result. The six M2 sample recordings outside the approved manifest were excluded. The other 1,196 approved recordings remain in the intended full acquisition set but were not a prerequisite for this run.", "",
             f"Working set: **{len(metadata):,} recordings** across eight classes with {metadata.sha256.nunique():,} unique SHA-256 hashes. Every source was present, size-matched, decoded OGG/Vorbis, mono at 32 kHz, and its hash still matched after the run.", "",
             "| Class | Recordings | Dataset % | Train | Validation | XC | iNat | Author groups | Unknown authors |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             *[f"| {r.primary_label} | {r.recordings} | {r.percent:.2f}% | {r.train} | {r.validation} | {r.XC} | {r.iNat} | {r.author_groups} | {r.unknown_authors} |" for r in count_table.reset_index().itertuples(index=False)], "",
             "## Split and leakage controls", "",
             "A deterministic `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` generated the fresh holdout. Selected feasible fold 5 contains 1,748 training recordings and 429 validation recordings (19.71%). Stratification used `primary_label × collection`. Groups join normalized authors across species/collections and exact duplicate hashes. There is no recording, author-group, or content-group overlap. All 109 literal Unknown-author recordings remain in training. All eight labels and all 16 observed label×collection cells occur in both partitions. The earlier 676-recording 12-class holdout was not reused.", "",
             "The validation partition holds out known authors; generalization to Unknown authors is unmeasured because all Unknown records remain in training. Author grouping does not establish spatial independence; coordinate grouping remains a future experiment.", "",
             "## Segmentation and features", "",
             "Audio was streamed using SoundFile at its native 32 kHz. Non-overlapping five-second windows yielded 17,273 complete windows and 2,153 short or final partial windows (19,426 total). Partial windows were right-padded with zeros; valid source samples, source offsets, and padding are explicit in the segment manifest. A tail shorter than the 1,024-sample FFT window is represented by one zero-padded frame. All source samples are retained; no segment audio was written.", "",
             "Each recording has a fixed 204-dimensional vector. Per-window frame summaries use mean and standard deviation for 13 power-Mel MFCCs, first- and second-order deltas, spectral centroid, bandwidth, 85% rolloff, zero-crossing rate, RMS, and seven spectral-contrast bands. Per-window statistics are aggregated across the source recording with mean and standard deviation. Mel configuration: 32 kHz, Hann, n_fft/win_length 1,024, hop 256, 128 Slaney-scale (`htk=False`), Slaney-normalized power-Mel bands from 0–16 kHz; log power uses a 1e-10 floor and MFCC DCT-II orthonormal normalization. This is exploratory, not an optimized model configuration.", "",
             "## Models and evaluation", "",
             "Compared Logistic Regression, Random Forest, and RBF SVM in scikit-learn Pipelines. StandardScaler is inside the Logistic Regression and SVM pipelines; Random Forest is unscaled. Class weights are balanced for Logistic Regression/SVM and balanced-subsample for Random Forest. Five-fold StratifiedGroupKFold hyperparameter tuning used only the 1,748-record training partition. Scalers are fitted within each CV fold. Macro F1 is the selection metric. Every model and parameter choice was fixed before the final holdout scoring.", "",
             "| Model | Mean training CV macro F1 | Holdout accuracy | Macro precision | Macro recall | Holdout macro F1 | Weighted F1 | OVR macro ROC-AUC | CV search seconds | Inference ms/record |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             *[f"| {r.model} | {r.best_cv_macro_f1:.4f} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} | {r.f1_macro:.4f} | {r.f1_weighted:.4f} | {'unsupported' if pd.isna(r.roc_auc_ovr_macro) else f'{r.roc_auc_ovr_macro:.4f}'} | {r.cv_fit_seconds:.2f} | {r.inference_ms_per_record:.4f} |" for r in table.itertuples(index=False)], "",
             f"Selected by training-CV macro F1: **{winner}**. Its final holdout macro F1 is {table.loc[table.model.eq(winner), 'f1_macro'].iloc[0]:.4f}. Logistic Regression has a higher score on this one holdout; the holdout was not used to select it, and this difference is reported transparently.", "",
             "### Selected model per-class holdout metrics", "",
             "| Class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |",
             *[f"| {r.primary_label} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.support} |" for r in per_class.itertuples(index=False)], "",
             "Saved artifacts include per-model CV hyperparameter-grid results, final holdout metrics, per-class classification reports, confusion matrices (CSV and plot), per-class F1 plots, model comparison tables, all fitted estimators, and a separate best-model artifact. SVM has no calibrated probabilities, so its ROC-AUC is unsupported; Logistic Regression and Random Forest report multiclass one-vs-rest macro ROC-AUC.", "",
             "## Limitations and next work", "",
             "This is recording-level primary-label classification, not soundscape multi-label detection and not an animal-population estimate. A recording's primary label may be a weak label for every window. Results apply only to this current eight-class author-grouped subset. Per-class performance varies. No claim is made about unseen species or all BirdCLEF recordings.", "",
             "Next, review class errors and confusion patterns, compare feature/segmentation settings using training-only grouped CV, and regenerate a 12-class split after the remaining approved recordings become available.", "",
             "Reproduce with `.venv\\Scripts\\python.exe -m pytest tests\\test_m3_pipeline.py tests\\test_download_approved_audio.py -q`, followed by `scripts\\run_m3_pipeline.py --stage validate`, `--stage features`, and `--stage train-evaluate`. Feature shards are resumable and checked against source SHA-256 and the extraction configuration.", ""]
    report = "\n".join(lines)
    atomic_write_text(config.FINAL_REPORT, report)
    atomic_write_text(report_dir / "m3_eight_class_experiment_report.md", report)
    return table, winner
