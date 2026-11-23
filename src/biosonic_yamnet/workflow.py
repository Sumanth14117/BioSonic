"""Training-only embedding extraction and frozen classifier workflow."""

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .classifiers import tune_with_training_cv
from .dataset import extract_partition_embeddings, persist_embeddings, read_training_manifest
from .io_utils import atomic_csv, atomic_joblib_dump, atomic_json, sha256_file
from .model import YamnetEncoder
from .smoke import run_smoke_test


def _load_aggregate(partition: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    matrix_path = config.DATA_ROOT / f"{partition}_embeddings.npz"
    metadata_path = config.DATA_ROOT / f"{partition}_embedding_manifest.csv"
    summary_path = config.DATA_ROOT / f"{partition}_embedding_summary.json"
    if not (matrix_path.is_file() and metadata_path.is_file() and summary_path.is_file()):
        raise FileNotFoundError(f"Complete {partition} embedding artifacts are required")
    with np.load(matrix_path, allow_pickle=False) as data:
        matrix = data["embeddings"].astype(np.float32)
        labels = data["labels"].astype(str)
        filenames = data["filenames"].astype(str)
    metadata = pd.read_csv(metadata_path, dtype={"filename": str, "primary_label": str,
                                                 "collection": str, "partition": str,
                                                 "group_id": str, "source_sha256": str})
    if matrix.shape != (len(metadata), config.EMBEDDING_DIM):
        raise ValueError(f"{partition} embedding matrix has unexpected shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{partition} embedding matrix contains non-finite values")
    if not (np.array_equal(labels, metadata.primary_label.astype(str).to_numpy()) and
            np.array_equal(filenames, metadata.filename.astype(str).to_numpy())):
        raise ValueError(f"{partition} embedding rows do not match their metadata")
    if metadata.filename.duplicated().any() or not metadata.partition.eq(partition).all():
        raise ValueError(f"{partition} embedding metadata has duplicate IDs or wrong partition")
    return matrix, metadata, json.loads(summary_path.read_text(encoding="utf-8"))


def extract_training_stage(progress=print) -> dict:
    frame = read_training_manifest()
    if (config.DATA_ROOT / "train_embeddings.npz").exists():
        matrix, metadata, summary = _load_aggregate("train")
        if len(frame) != len(metadata) or frame.filename.astype(str).tolist() != metadata.filename.astype(str).tolist():
            raise ValueError("Existing training embedding aggregate does not align with the authoritative train manifest")
        return {"status": "already_complete", **summary}
    config.REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    encoder = YamnetEncoder()
    def log(index, total, ok, failed, filename):
        progress(f"YAMNet train extraction {index}/{total}: {ok} verified, {failed} failed; current={filename}")
    start = time.perf_counter()
    matrix, metadata, summary = extract_partition_embeddings(frame, encoder, "train", progress=log)
    summary["wall_seconds_including_audio_and_hashing"] = time.perf_counter() - start
    paths = persist_embeddings(matrix, metadata, "train", summary)
    atomic_csv([{"filename": row.filename, "primary_label": row.primary_label,
                 "status": "verified", "patch_count": int(row.patch_count),
                 "embedding_dim": config.EMBEDDING_DIM, "source_sha256": row.source_sha256}
                for row in metadata.itertuples(index=False)],
               ["filename", "primary_label", "status", "patch_count", "embedding_dim", "source_sha256"],
               config.REPORT_ROOT / "training_extraction_manifest.csv")
    summary["aggregate_paths"] = [str(path.relative_to(config.ROOT)) for path in paths]
    atomic_json(summary, config.REPORT_ROOT / "training_extraction_summary.json")
    return summary


def _freeze_exists_valid() -> bool:
    freeze_path = config.MODEL_ROOT / "freeze_manifest.json"
    model_path = config.MODEL_ROOT / "final_classifier.joblib"
    if not freeze_path.exists():
        return False
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    return model_path.is_file() and sha256_file(model_path) == freeze.get("classifier_sha256")


def train_and_freeze_stage(progress=print) -> dict:
    if _freeze_exists_valid():
        return {"status": "already_frozen", **json.loads((config.MODEL_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))}
    protected = [config.MODEL_ROOT / "final_classifier.joblib",
                 config.REPORT_ROOT / "classifier_cv_candidates.csv",
                 config.REPORT_ROOT / "classifier_cv_folds.csv"]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"Partial M5 training outputs exist; refusing to overwrite: {existing}")

    smoke_path = config.REPORT_ROOT / "model_smoke.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("Complete the M5 YAMNet smoke stage before classifier training")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    x, metadata, extraction = _load_aggregate("train")
    expected = read_training_manifest()
    if metadata.filename.astype(str).tolist() != expected.filename.astype(str).tolist():
        raise ValueError("Training embedding manifest order differs from the M3 training partition")
    if not metadata.primary_label.astype(str).equals(expected.primary_label.astype(str)):
        raise ValueError("Training embedding labels differ from the M3 training partition")
    if not metadata.group_id.astype(str).equals(expected.group_id.astype(str)):
        raise ValueError("Training embedding groups differ from the M3 author/content groups")

    config.REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    def log(done, total, model_name, params, score):
        progress(f"CV candidate {done}/{total}: {model_name} {params}; mean macro-F1={score:.5f}")
    result = tune_with_training_cv(x, metadata, progress=log)
    atomic_csv(result["candidate_rows"], list(result["candidate_rows"][0]),
               config.REPORT_ROOT / "classifier_cv_candidates.csv")
    atomic_csv(result["fold_rows"], list(result["fold_rows"][0]),
               config.REPORT_ROOT / "classifier_cv_folds.csv")
    model_path = config.MODEL_ROOT / "final_classifier.joblib"
    atomic_joblib_dump(result["estimator"], model_path)
    training_map = [{"index": index, "primary_label": label}
                    for index, label in enumerate(config.WORKING_LABELS)]
    atomic_json({"classes": training_map, "order": list(config.WORKING_LABELS)},
                config.MODEL_ROOT / "class_mapping.json")
    versions = {name: version(name) for name in (
        "tensorflow", "tf-keras", "numpy", "scipy", "soundfile", "resampy",
        "h5py", "scikit-learn", "pandas", "matplotlib", "joblib")}
    info = {
        "selected_model": result["selected_model"],
        "yamnet_parameter_count": int(smoke["parameter_count"]),
        "selected_parameters": result["selected_params"],
        "selection_metric": "mean five-fold author/content-grouped CV macro-F1",
        "selected_cv_metrics": result["selected_cv"],
        "cv_candidate_count": len(result["candidate_rows"]),
        "cv_fold_count": result["cv_splits"],
        "cv_total_seconds": result["cv_total_seconds"],
        "final_refit_seconds": result["final_fit_seconds"],
        "training_recordings": len(metadata),
        "training_embedding_dimension": config.EMBEDDING_DIM,
        "training_embeddings_sha256": sha256_file(config.DATA_ROOT / "train_embeddings.npz"),
        "training_metadata_sha256": sha256_file(config.DATA_ROOT / "train_embedding_manifest.csv"),
        "training_extraction_summary": extraction,
        "classifier_sha256": sha256_file(model_path),
        "classifier_bytes": model_path.stat().st_size,
        "fixed_class_order": list(config.WORKING_LABELS),
        "class_mapping_sha256": sha256_file(config.MODEL_ROOT / "class_mapping.json"),
        "seed": config.SEED,
        "python": platform.python_version(),
        "package_versions": versions,
        "model_source_commit": config.OFFICIAL_SOURCE_COMMIT,
        "pretrained_weights_sha256": config.WEIGHTS_SHA256,
        "training_only": True,
        "holdout_accessed": False,
    }
    atomic_json(info, config.MODEL_ROOT / "final_model_info.json")
    freeze = {
        "frozen": True, "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classifier_sha256": info["classifier_sha256"],
        "training_recordings": len(metadata), "selection_metric": info["selection_metric"],
        "selected_model": info["selected_model"], "selected_parameters": info["selected_parameters"],
        "cv_mean_macro_f1": info["selected_cv_metrics"]["mean_macro_f1"],
        "cv_std_macro_f1": info["selected_cv_metrics"]["std_macro_f1"],
        "training_embeddings_sha256": info["training_embeddings_sha256"],
        "training_metadata_sha256": info["training_metadata_sha256"],
        "model_info_sha256": sha256_file(config.MODEL_ROOT / "final_model_info.json"),
        "class_mapping_sha256": info["class_mapping_sha256"],
        "holdout_used_for_selection": False,
    }
    atomic_json(freeze, config.MODEL_ROOT / "freeze_manifest.json")
    return info
