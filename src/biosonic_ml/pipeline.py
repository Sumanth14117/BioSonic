import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .dataset import load_working_dataset
from .features import extract_feature_table, feature_configuration
from .io_utils import atomic_write_json, atomic_write_text
from .splitting import make_holdout
from .training import train_and_evaluate


def _write_dataset_summary(frame: pd.DataFrame, path: Path):
    species = frame.groupby("primary_label").agg(recordings=("filename", "size"), authors=("author_group_id", "nunique"),
                                                   mean_duration=("duration_seconds", "mean"),
                                                   min_duration=("duration_seconds", "min"),
                                                   max_duration=("duration_seconds", "max")).reset_index()
    species["percent"] = species.recordings / len(frame) * 100
    collections = frame.groupby(["primary_label", "collection"]).size().rename("recordings").reset_index()
    payload = {"dataset_count": len(frame), "classes": list(config.WORKING_LABELS),
               "verified_count": int(len(frame)), "unique_sha256": int(frame.sha256.nunique()),
               "duplicate_file_count": int(len(frame) - frame.sha256.nunique()),
               "sample_rate_counts": frame.sample_rate.value_counts().to_dict(),
               "channel_counts": frame.channels.value_counts().to_dict(),
               "species_counts": species.to_dict("records"), "species_collection_counts": collections.to_dict("records"),
               "durations_seconds": {"min": float(frame.duration_seconds.min()), "median": float(frame.duration_seconds.median()),
                                     "max": float(frame.duration_seconds.max())},
               "source_audio_changes": False}
    atomic_write_json(payload, path)


def prepare_split() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = load_working_dataset()
    split, summary = make_holdout(frame)
    _write_dataset_summary(frame, config.REPORT_ROOT / "dataset_audit_summary.json")
    joined = frame.drop(columns=["partition"], errors="ignore").merge(
        split[["filename", "partition", "unknown_author"]], on="filename", validate="one_to_one")
    if len(joined) != len(frame):
        raise RuntimeError("Split manifest does not map one-to-one to working audio")
    return frame, joined, summary


def load_saved_features():
    if not config.FEATURES_NPZ.is_file() or not config.FEATURES_CSV.is_file():
        raise FileNotFoundError("Feature outputs not found; run --stage features first")
    with np.load(config.FEATURES_NPZ, allow_pickle=False) as arrays:
        X = arrays["X"].astype(np.float32)
        names = arrays["filenames"].astype(str)
    metadata = pd.read_csv(config.FEATURES_CSV, dtype={"filename": str})
    if names.tolist() != metadata.filename.astype(str).tolist():
        raise RuntimeError("Feature matrix and metadata row order differ")
    if set(metadata.partition) != {"train", "validation"}:
        raise RuntimeError("Feature metadata lacks both split partitions")
    if not np.isfinite(X).all():
        raise RuntimeError("Feature matrix contains non-finite values")
    return X, metadata


def write_segment_manifest(metadata: pd.DataFrame, path=config.ROOT / "reports/ml/eight_class_segment_manifest.csv"):
    rows = []
    for item in metadata.itertuples(index=False):
        n = int(item.sample_count)
        count = (n + config.SEGMENT_SAMPLES - 1) // config.SEGMENT_SAMPLES
        for index in range(count):
            start = index * config.SEGMENT_SAMPLES
            end = min(start + config.SEGMENT_SAMPLES, n)
            rows.append({"filename": item.filename, "primary_label": item.primary_label,
                         "collection": item.collection, "partition": item.partition,
                         "group_id": item.group_id, "sha256": item.sha256, "segment_index": index,
                         "start_sample": start, "end_sample_exclusive": end,
                         "source_sample_count": end - start,
                         "padded_sample_count": config.SEGMENT_SAMPLES - (end - start),
                         "sample_rate": config.SR, "segment_length_samples": config.SEGMENT_SAMPLES,
                         "window_duration_seconds": config.SEGMENT_SECONDS,
                         "source_start_seconds": start / config.SR, "source_end_seconds": end / config.SR,
                         "valid_source_duration_seconds": (end - start) / config.SR,
                         "is_incomplete_tail_or_short_recording": (end - start) < config.SEGMENT_SAMPLES})
    result = pd.DataFrame(rows)
    if len(result) != int(metadata.segment_count.sum()):
        raise RuntimeError("Segment manifest does not reconcile with extracted windows")
    from .io_utils import atomic_write_csv
    atomic_write_csv(result, path)
    return result


def run(stage="all"):
    if stage in {"validate", "all"}:
        frame, joined, summary = prepare_split()
        print(f"Dataset validated: {len(frame):,} recordings, {len(config.WORKING_LABELS)} classes; split fold {summary['selected_fold']}: {summary['training_recordings']} train / {summary['validation_recordings']} validation")
        if stage == "validate":
            return
    if stage in {"features", "all"}:
        if stage == "features":
            if not config.SPLIT_MANIFEST.exists():
                raise FileNotFoundError("Split manifest missing; run --stage validate first")
            saved = pd.read_csv(config.SPLIT_MANIFEST)
            if len(saved) != 2177 or saved.filename.duplicated().any():
                raise RuntimeError("Split manifest is stale, incomplete, or contains duplicate recordings")
            if set(saved.primary_label) != set(config.WORKING_LABELS) or set(saved.partition) != {"train", "validation"}:
                raise RuntimeError("Split manifest does not match the approved eight-class split")
            if set(saved.filename) & config.SAMPLE_FILES:
                raise RuntimeError("Split manifest unexpectedly contains an M2 sample recording")
            saved["audio_path"] = saved.filename.map(lambda name: str(config.AUDIO_ROOT.joinpath(*str(name).split("/"))))
            joined = saved
        def progress(done, total):
            print(f"Feature extraction: {done:,}/{total:,} recordings")
        X, feat_meta = extract_feature_table(joined, progress=progress)
        segment_rows = write_segment_manifest(feat_meta)
        print(f"Features ready: {X.shape[0]:,} recordings × {X.shape[1]} features; {int(feat_meta.segment_count.sum()):,} padded-or-complete windows")
        print(f"Segment manifest: {len(segment_rows):,} windows; partial tails and padding are explicitly recorded")
        if stage == "features":
            return
    if stage in {"train-evaluate", "all"}:
        X, feat_meta = load_saved_features()
        table, winner = train_and_evaluate(X, feat_meta)
        print(f"Training/evaluation complete. Best by training CV macro-F1: {winner}")
        print(table.to_string(index=False))


def write_feature_readme():
    if config.FEATURE_CONFIG.exists():
        config_json = json.loads(config.FEATURE_CONFIG.read_text(encoding="utf-8"))
        return config_json
    return feature_configuration()
