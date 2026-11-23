"""One-shot final holdout evaluation after classifier freeze."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .classifiers import _score_matrix, classification_metrics
from .dataset import (extract_partition_embeddings, persist_embeddings,
                      read_holdout_manifest_after_freeze, read_training_manifest)
from .io_utils import atomic_csv, atomic_json, sha256_file
from .model import YamnetEncoder


def _load_or_extract_holdout(progress=print) -> tuple[np.ndarray, pd.DataFrame, dict]:
    frame = read_holdout_manifest_after_freeze()
    matrix_path = config.DATA_ROOT / "validation_embeddings.npz"
    if matrix_path.exists():
        with np.load(matrix_path, allow_pickle=False) as saved:
            x = saved["embeddings"].astype(np.float32)
            filenames = saved["filenames"].astype(str)
            labels = saved["labels"].astype(str)
        metadata = pd.read_csv(config.DATA_ROOT / "validation_embedding_manifest.csv",
                               dtype={"filename": str, "primary_label": str, "partition": str,
                                      "group_id": str, "source_sha256": str})
        if (x.shape != (config.HOLDOUT_RECORDINGS, config.EMBEDDING_DIM) or
                not np.isfinite(x).all() or not np.array_equal(filenames, metadata.filename.astype(str)) or
                not np.array_equal(labels, metadata.primary_label.astype(str)) or
                metadata.filename.astype(str).tolist() != frame.filename.astype(str).tolist()):
            raise ValueError("Existing M5 holdout embeddings fail integrity/alignment checks")
        summary = json.loads((config.DATA_ROOT / "validation_embedding_summary.json").read_text(encoding="utf-8"))
        return x, metadata, summary
    encoder = YamnetEncoder()
    def log(index, total, ok, failed, filename):
        progress(f"YAMNet final holdout extraction {index}/{total}: {ok} verified, {failed} failed; current={filename}")
    start = time.perf_counter()
    x, metadata, summary = extract_partition_embeddings(frame, encoder, "validation", progress=log)
    summary["wall_seconds_including_audio_and_hashing"] = time.perf_counter() - start
    persist_embeddings(x, metadata, "validation", summary)
    return x, metadata, summary


def evaluate_frozen_holdout(progress=print) -> dict:
    freeze_path = config.MODEL_ROOT / "freeze_manifest.json"
    model_path = config.MODEL_ROOT / "final_classifier.joblib"
    if not freeze_path.is_file() or not model_path.is_file():
        raise PermissionError("Final holdout evaluation is blocked until a classifier is frozen")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_hash = sha256_file(model_path)
    if model_hash != freeze.get("classifier_sha256"):
        raise ValueError("Frozen classifier hash changed; refusing holdout evaluation")
    if (config.REPORT_ROOT / "holdout_metrics.json").exists():
        raise FileExistsError("Final holdout evaluation already exists; refusing a second evaluation")

    x, metadata, extraction = _load_or_extract_holdout(progress)
    # The M5 model is loaded only after confirming its frozen digest.
    classifier = joblib.load(model_path)
    train_metadata = read_training_manifest()
    holdout_manifest = read_holdout_manifest_after_freeze()
    if metadata.filename.astype(str).tolist() != holdout_manifest.filename.astype(str).tolist():
        raise ValueError("Holdout embeddings no longer align with the authoritative holdout manifest")
    if (set(train_metadata.group_id.astype(str)) & set(holdout_manifest.group_id.astype(str)) or
            set(train_metadata.filename.astype(str)) & set(holdout_manifest.filename.astype(str)) or
            set(train_metadata.sha256.astype(str)) & set(holdout_manifest.sha256.astype(str))):
        raise ValueError("Training/holdout leakage detected after freeze")

    started = time.perf_counter()
    predictions = classifier.predict(x).astype(str)
    scores, score_kind = _score_matrix(classifier, x)
    inference_seconds = time.perf_counter() - started
    y_true = metadata.primary_label.astype(str).to_numpy()
    if predictions.shape != y_true.shape or len(predictions) != config.HOLDOUT_RECORDINGS:
        raise ValueError("Final holdout prediction count or label alignment is invalid")
    metrics = classification_metrics(classifier, x, y_true, predictions=predictions)
    report = classification_report(y_true, predictions, labels=list(config.WORKING_LABELS),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=list(config.WORKING_LABELS))
    prediction_rows = []
    for row_index, row in enumerate(metadata.itertuples(index=False)):
        payload = {"filename": row.filename, "true_label": row.primary_label,
                   "predicted_label": predictions[row_index], "collection": row.collection,
                   "patch_count": int(row.patch_count), "score_kind": score_kind}
        payload.update({f"score_{label}": float(scores[row_index, i])
                        for i, label in enumerate(config.WORKING_LABELS)})
        if score_kind == "probability":
            payload.update({f"probability_{label}": float(scores[row_index, i])
                            for i, label in enumerate(config.WORKING_LABELS)})
        prediction_rows.append(payload)

    config.REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(prediction_rows[0])
    atomic_csv(prediction_rows, fields, config.REPORT_ROOT / "holdout_recording_predictions.csv")
    atomic_csv([{"primary_label": label,
                 "precision": report[label]["precision"], "recall": report[label]["recall"],
                 "f1": report[label]["f1-score"], "support": int(report[label]["support"])}
                for label in config.WORKING_LABELS],
               ["primary_label", "precision", "recall", "f1", "support"],
               config.REPORT_ROOT / "holdout_per_class_metrics.csv")
    atomic_csv([{"true_label": config.WORKING_LABELS[i], **{
        f"predicted_{config.WORKING_LABELS[j]}": int(cm[i, j])
        for j in range(len(config.WORKING_LABELS))}}
        for i in range(len(config.WORKING_LABELS))],
        ["true_label", *[f"predicted_{label}" for label in config.WORKING_LABELS]],
        config.REPORT_ROOT / "holdout_confusion_matrix.csv")
    result = {
        "status": "final_evaluated_once", "holdout_recordings": len(metadata),
        "selected_classifier": freeze["selected_model"], "frozen_classifier_sha256": model_hash,
        "freeze_manifest_sha256": sha256_file(freeze_path), "metrics": metrics,
        "class_order": list(config.WORKING_LABELS), "per_class": report,
        "confusion_matrix": cm.tolist(), "prediction_score_kind": score_kind,
        "classifier_inference_seconds": inference_seconds,
        "classifier_inference_ms_per_record": inference_seconds * 1000 / len(metadata),
        "embedding_extraction": extraction,
        "holdout_used_for_selection": False,
        "holdout_evaluation_count": 1,
    }
    atomic_json(result, config.REPORT_ROOT / "holdout_metrics.json")
    return result
