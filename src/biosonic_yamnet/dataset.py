"""Partition-gated readers for M5; training APIs never parse holdout rows."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config
from .audio import prepare_waveform
from .io_utils import atomic_csv, atomic_json, atomic_npz, sha256_file


class DatasetIntegrityError(ValueError):
    pass


def read_training_manifest(path: Path = config.SPLIT_MANIFEST) -> pd.DataFrame:
    """Read the sorted training prefix only; stop before parsing holdout rows."""
    rows = []
    crossed_boundary = False
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "primary_label", "collection", "group_id", "author_group_id",
                    "sha256", "expected_size_bytes", "actual_size_bytes", "sample_rate",
                    "channels", "sample_count", "duration_seconds", "partition", "unknown_author"}
        if not required.issubset(reader.fieldnames or []):
            raise DatasetIntegrityError(f"Split manifest missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["partition"] != "train":
                crossed_boundary = True
                break
            rows.append(row)
    if not crossed_boundary:
        raise DatasetIntegrityError("Training partition boundary was not found; refusing incomplete split")
    frame = pd.DataFrame(rows)
    if len(frame) != config.TRAIN_RECORDINGS:
        raise DatasetIntegrityError(f"Expected {config.TRAIN_RECORDINGS} training recordings; found {len(frame)}")
    return _validate_frame(frame, expected_partition="train")


def _validate_frame(frame: pd.DataFrame, expected_partition: str) -> pd.DataFrame:
    if frame.filename.duplicated().any():
        raise DatasetIntegrityError(f"Duplicate recording filename in {expected_partition} partition")
    if set(frame.primary_label.astype(str)) != set(config.WORKING_LABELS):
        raise DatasetIntegrityError(f"{expected_partition} partition does not contain the fixed eight classes")
    if frame.group_id.isna().any() or frame.group_id.astype(str).str.len().eq(0).any():
        raise DatasetIntegrityError("Missing group identifier")
    if frame.sha256.isna().any() or frame.sha256.astype(str).str.len().ne(64).any():
        raise DatasetIntegrityError("Invalid source SHA-256 metadata")
    if frame.sha256.duplicated().any():
        raise DatasetIntegrityError("Exact duplicate source audio found in an M5 partition")
    if not frame.partition.eq(expected_partition).all():
        raise DatasetIntegrityError(f"Unexpected partition value while reading {expected_partition}")
    if set(frame.filename.astype(str)) & config.M2_SAMPLE_FILES:
        raise DatasetIntegrityError("An out-of-scope M2 sample entered the eight-class experiment")
    numeric = ["expected_size_bytes", "actual_size_bytes", "sample_rate", "channels", "sample_count"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(np.int64)
    frame["duration_seconds"] = pd.to_numeric(frame.duration_seconds, errors="raise")
    frame["unknown_author"] = frame.unknown_author.astype(str).str.casefold().isin({"true", "1"})
    if not frame.collection.astype(str).isin({"XC", "iNat"}).all():
        raise DatasetIntegrityError("Unexpected source collection value")
    return frame.reset_index(drop=True)


def read_holdout_manifest_after_freeze(path: Path = config.SPLIT_MANIFEST,
                                       frozen_path: Path | None = None) -> pd.DataFrame:
    """Read holdout rows only after checking the immutable M5 freeze marker."""
    frozen_path = frozen_path or (config.MODEL_ROOT / "freeze_manifest.json")
    if not frozen_path.is_file():
        raise PermissionError("Holdout access is blocked until the M5 classifier is frozen")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    classifier_path = config.MODEL_ROOT / "final_classifier.joblib"
    if not classifier_path.is_file() or sha256_file(classifier_path) != frozen.get("classifier_sha256"):
        raise DatasetIntegrityError("Frozen classifier changed; holdout access is blocked")
    if frozen.get("training_recordings") != config.TRAIN_RECORDINGS:
        raise DatasetIntegrityError("Freeze marker does not describe the approved training partition")

    rows = []
    crossed_boundary = False
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["partition"] == "validation":
                crossed_boundary = True
                rows.append(row)
    if not crossed_boundary:
        raise DatasetIntegrityError("No holdout partition was found after classifier freeze")
    holdout = _validate_frame(pd.DataFrame(rows), expected_partition="validation")
    if len(holdout) != config.HOLDOUT_RECORDINGS:
        raise DatasetIntegrityError(f"Expected {config.HOLDOUT_RECORDINGS} final holdout recordings; found {len(holdout)}")
    training = read_training_manifest(path)
    for col in ("filename", "group_id", "sha256"):
        overlap = set(training[col].astype(str)) & set(holdout[col].astype(str))
        if overlap:
            raise DatasetIntegrityError(f"Train/holdout {col} overlap has {len(overlap)} values")
    return holdout


def recording_shard_key(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:32]


def aggregate_patch_embeddings(patch_embeddings: np.ndarray) -> np.ndarray:
    patches = np.asarray(patch_embeddings)
    if patches.ndim != 2 or patches.shape[0] < 1 or patches.shape[1] != config.EMBEDDING_DIM:
        raise ValueError(f"Expected non-empty [patches, {config.EMBEDDING_DIM}] array, got {patches.shape}")
    if not np.isfinite(patches).all():
        raise ValueError("Patch embeddings contain NaN or infinity")
    pooled = patches.astype(np.float64).mean(axis=0).astype(np.float32)
    if pooled.shape != (config.EMBEDDING_DIM,) or not np.isfinite(pooled).all():
        raise ValueError("Recording-level embedding failed shape or finite-value validation")
    return pooled


def _load_shard(path: Path, record: dict, config_hash: str) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    checks = {
        "filename": record["filename"], "primary_label": record["primary_label"],
        "group_id": record["group_id"], "partition": record["partition"],
        "source_sha256": record["sha256"], "feature_config_hash": config_hash,
    }
    for key, expected in checks.items():
        if str(result[key].item()) != str(expected):
            raise DatasetIntegrityError(f"Stale M5 embedding shard metadata for {record['filename']}: {key}")
    embedding = np.asarray(result["embedding"], dtype=np.float32)
    if embedding.shape != (config.EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise DatasetIntegrityError(f"Invalid cached embedding for {record['filename']}")
    if int(result["patch_count"].item()) < 1:
        raise DatasetIntegrityError(f"Invalid patch count for {record['filename']}")
    return result


def _feature_config_hash() -> str:
    import importlib.metadata as metadata

    payload = {
        "experiment": config.EXPERIMENT_VERSION,
        "model_commit": config.OFFICIAL_SOURCE_COMMIT,
        "weights_sha256": config.WEIGHTS_SHA256,
        "source_sample_rate": config.SAMPLE_RATE,
        "model_sample_rate": config.MODEL_SAMPLE_RATE,
        "resampling": f"resampy {metadata.version('resampy')} default kaiser_best",
        "source_decode": "SoundFile int16 then float32 / 32768, matching official inference.py",
        "resampled_length_convention": "floor(source_samples * target_rate / source_rate)",
        "post_resample_clipping": False,
        "aggregation": config.AGGREGATION,
        "official_frontend": {
            "sample_rate": 16_000, "stft_window_seconds": 0.025,
            "stft_hop_seconds": 0.010, "window": "periodic Hann",
            "mel_bands": 64, "mel_min_hz": 125.0, "mel_max_hz": 7500.0,
            "log_offset": 0.001, "patch_window_seconds": 0.96,
            "patch_hop_seconds": 0.48,
        },
        "embedding_dim": config.EMBEDDING_DIM,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def extract_partition_embeddings(frame: pd.DataFrame, encoder, partition: str,
                                 shard_root: Path | None = None, progress=None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    if partition not in {"train", "validation"}:
        raise ValueError("partition must be train or validation")
    if frame.partition.nunique() != 1 or not frame.partition.eq(partition).all():
        raise DatasetIntegrityError("Mixed or incorrect partition passed to embedding extraction")
    if frame.filename.duplicated().any():
        raise DatasetIntegrityError("Duplicate recording IDs passed to embedding extraction")
    shard_root = Path(shard_root or (config.TRAIN_SHARDS if partition == "train" else config.HOLDOUT_SHARDS))
    config_hash = _feature_config_hash()
    errors = []
    records = []
    vectors = []
    for index, row in enumerate(frame.to_dict("records"), start=1):
        filename = str(row["filename"])
        audio_path = config.AUDIO_ROOT.joinpath(*filename.replace("\\", "/").split("/"))
        shard_path = shard_root / f"{recording_shard_key(filename)}.npz"
        try:
            if shard_path.is_file():
                saved = _load_shard(shard_path, row, config_hash)
                vector = np.asarray(saved["embedding"], dtype=np.float32)
                record_info = json.loads(str(saved["audio_metadata"].item()))
                patches = int(saved["patch_count"].item())
                elapsed = float(saved["inference_seconds"].item())
            else:
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Approved source recording is missing: {audio_path}")
                if audio_path.stat().st_size != int(row["expected_size_bytes"]):
                    raise DatasetIntegrityError(f"Source byte count differs from manifest: {filename}")
                before_hash = sha256_file(audio_path)
                if before_hash != str(row["sha256"]):
                    raise DatasetIntegrityError(f"Source SHA-256 differs from split manifest: {filename}")
                info = sf.info(str(audio_path))
                if (info.samplerate != config.SAMPLE_RATE or info.channels != 1 or
                        info.frames != int(row["sample_count"]) or info.format != "OGG"):
                    raise DatasetIntegrityError(f"Source audio metadata differs from split manifest: {filename}")
                waveform, record_info = prepare_waveform(audio_path)
                import time

                start = time.perf_counter()
                _, patch_embeddings, _ = encoder.infer(waveform)
                elapsed = time.perf_counter() - start
                patches = int(patch_embeddings.shape[0])
                vector = aggregate_patch_embeddings(patch_embeddings)
                if sha256_file(audio_path) != before_hash:
                    raise DatasetIntegrityError(f"Source audio changed during extraction: {filename}")
                shard_root.mkdir(parents=True, exist_ok=True)
                atomic_npz(
                    shard_path,
                    embedding=vector,
                    patch_count=np.array(patches, dtype=np.int32),
                    inference_seconds=np.array(elapsed, dtype=np.float64),
                    audio_metadata=np.array(json.dumps(record_info, sort_keys=True)),
                    filename=np.array(filename),
                    primary_label=np.array(str(row["primary_label"])),
                    group_id=np.array(str(row["group_id"])),
                    partition=np.array(partition),
                    source_sha256=np.array(str(row["sha256"])),
                    feature_config_hash=np.array(config_hash),
                )
            vectors.append(vector)
            records.append({
                "filename": filename,
                "primary_label": str(row["primary_label"]),
                "collection": str(row["collection"]),
                "partition": partition,
                "group_id": str(row["group_id"]),
                "author_group_id": str(row["author_group_id"]),
                "unknown_author": bool(row["unknown_author"]),
                "source_sha256": str(row["sha256"]),
                "patch_count": patches,
                "embedding_dim": config.EMBEDDING_DIM,
                "source_sample_count": int(row["sample_count"]),
                "source_duration_seconds": float(row["duration_seconds"]),
                "resampled_sample_count": int(record_info["model_sample_count"]),
                "resampled_sample_rate": config.MODEL_SAMPLE_RATE,
                "source_full_scale_sample_count": int(record_info["source_full_scale_sample_count"]),
                "resampled_samples_outside_unit_interval": int(
                    record_info["resampled_samples_outside_unit_interval"]
                ),
                "resampled_peak_abs": float(record_info["resampled_peak_abs"]),
                "inference_seconds": elapsed,
                "status": "verified",
            })
        except Exception as exc:
            errors.append({
                "filename": filename, "primary_label": str(row.get("primary_label", "")),
                "partition": partition, "status": "failed",
                "error_type": type(exc).__name__, "error_message": str(exc),
            })
        if progress and (index % 50 == 0 or index == len(frame)):
            progress(index, len(frame), len(records), len(errors), filename)

    if errors:
        failure_path = config.REPORT_ROOT / f"{partition}_embedding_failures.csv"
        atomic_csv(errors, ["filename", "primary_label", "partition", "status", "error_type", "error_message"], failure_path)
        raise RuntimeError(f"{len(errors)} {partition} embedding failures; details saved to {failure_path}")
    metadata = pd.DataFrame(records)
    matrix = np.vstack(vectors).astype(np.float32)
    expected = config.TRAIN_RECORDINGS if partition == "train" else config.HOLDOUT_RECORDINGS
    if matrix.shape != (expected if len(frame) == expected else len(frame), config.EMBEDDING_DIM):
        raise DatasetIntegrityError(f"Unexpected aggregated embedding matrix shape: {matrix.shape}")
    if len(metadata) != len(frame) or metadata.filename.tolist() != frame.filename.astype(str).tolist():
        raise DatasetIntegrityError("Embedding row order or recording identity differs from its manifest")
    if not np.isfinite(matrix).all():
        raise DatasetIntegrityError("Aggregated embedding matrix contains non-finite values")
    return matrix, metadata, {"feature_config_hash": config_hash, "recordings": len(metadata),
                              "embedding_dim": config.EMBEDDING_DIM,
                              "patches_total": int(metadata.patch_count.sum()),
                              "patches_min": int(metadata.patch_count.min()),
                              "patches_max": int(metadata.patch_count.max()),
                              "source_full_scale_sample_count": int(metadata.source_full_scale_sample_count.sum()),
                              "resampled_samples_outside_unit_interval": int(metadata.resampled_samples_outside_unit_interval.sum()),
                              "resampled_peak_abs_max": float(metadata.resampled_peak_abs.max()),
                              "mean_inference_seconds_per_recording": float(metadata.inference_seconds.mean()),
                              "total_model_inference_seconds": float(metadata.inference_seconds.sum())}


def persist_embeddings(matrix: np.ndarray, metadata: pd.DataFrame, partition: str,
                        summary: dict) -> tuple[Path, Path]:
    if matrix.shape != (len(metadata), config.EMBEDDING_DIM):
        raise DatasetIntegrityError("Cannot persist misaligned M5 embeddings")
    root = config.DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    matrix_path = root / f"{partition}_embeddings.npz"
    metadata_path = root / f"{partition}_embedding_manifest.csv"
    if matrix_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {partition} embedding aggregate")
    # Explicit Unicode arrays remain loadable with NumPy's safe allow_pickle=False default.
    atomic_npz(matrix_path, embeddings=matrix,
               labels=metadata.primary_label.astype(str).to_numpy(dtype=str),
               filenames=metadata.filename.astype(str).to_numpy(dtype=str))
    atomic_csv(metadata.to_dict("records"), list(metadata.columns), metadata_path)
    atomic_json(summary, root / f"{partition}_embedding_summary.json")
    return matrix_path, metadata_path
