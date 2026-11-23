import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

from biosonic_ml.dataset import sha256_file
from biosonic_ml.io_utils import atomic_write_csv, atomic_write_json
from . import config


class M4DataError(ValueError):
    pass


def source_table() -> pd.DataFrame:
    split = pd.read_csv(config.M3_SPLIT, dtype={"filename": str, "primary_label": str,
                                                "group_id": str, "sha256": str})
    seg = pd.read_csv(config.M3_SEGMENTS, dtype={"filename": str, "primary_label": str,
                                                 "group_id": str, "sha256": str})
    features = pd.read_csv(config.M3_FEATURE_ROOT / "feature_metadata.csv", dtype={"filename": str,
                                                                                    "primary_label": str})
    feature_config = json.loads((config.M3_FEATURE_ROOT / "feature_config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(config.DOWNLOAD_MANIFEST, dtype={"filename": str, "primary_label": str})
    required = {"filename", "primary_label", "collection", "group_id", "author_group_id", "sha256",
                "expected_size_bytes", "actual_size_bytes", "sample_rate", "channels", "sample_count",
                "duration_seconds", "partition", "unknown_author"}
    if not required.issubset(split.columns):
        raise M4DataError(f"M3 split is missing columns: {sorted(required - set(split.columns))}")
    if len(split) != config.EXPECTED_RECORDINGS or split.filename.duplicated().any():
        raise M4DataError("M3 split row count or recording identity is inconsistent")
    if set(split.primary_label) != set(config.WORKING_LABELS):
        raise M4DataError("M3 split labels do not match the approved eight-class scope")
    for key, expected in (("sample_rate", config.SR), ("segment_seconds", config.SEGMENT_SECONDS),
                          ("n_fft", 1024), ("hop_length", 256), ("mel_bands", 128),
                          ("mel_scale", "Slaney (htk=False)"), ("mel_norm", "Slaney"),
                          ("spectrogram", "power")):
        if feature_config.get(key) != expected:
            raise M4DataError(f"Existing M3 feature configuration differs for {key}")
    if split.groupby("primary_label").size().to_dict() != config.EXPECTED_CLASS_COUNTS:
        raise M4DataError("M3 split class counts differ from the audited 2,177-recording set")
    if split.partition.value_counts().to_dict() != {"train": 1748, "validation": 429}:
        raise M4DataError("M3 partition sizes changed; refusing to reconstruct or replace the split")
    if set(split.collection) - {"XC", "iNat"}:
        raise M4DataError("Unexpected collection labels in M3 split")
    if features.filename.tolist() != split.filename.tolist():
        raise M4DataError("M3 feature metadata row order differs from split manifest")
    if not np.array_equal(np.load(config.M3_FEATURE_ROOT / "features.npz", allow_pickle=False)["y"],
                          features.primary_label.to_numpy(dtype=str)):
        raise M4DataError("M3 feature labels differ from aligned feature metadata")
    if not seg.filename.isin(split.filename).all() or len(seg) != 19426:
        raise M4DataError("M3 segment manifest is inconsistent with the split")
    if not seg.groupby("filename").partition.nunique().eq(1).all():
        raise M4DataError("M3 segment partition mapping is inconsistent")
    if not set(manifest.loc[manifest.verification_status.eq("verified"), "filename"]) >= set(split.filename):
        raise M4DataError("M3 split includes a file not marked verified by the acquisition manifest")
    if split.filename.duplicated().any() or set(split.filename).intersection({
        "rubthr1/XC1006659.ogg", "banana/XC1034681.ogg", "fepowl/XC1034537.ogg",
        "bafcur1/XC1018568.ogg", "22973/XC892927.ogg", "244024/iNat103300.ogg"}):
        raise M4DataError("Duplicate identity or M2 sample found in M3 split")
    # Current source-file verification: file presence, exact expected size, decode metadata.
    for row in split.itertuples(index=False):
        path = config.AUDIO_ROOT.joinpath(*str(row.filename).split("/"))
        if not path.is_file() or path.stat().st_size != int(row.expected_size_bytes):
            raise M4DataError(f"Source file missing or size changed: {row.filename}")
        info = sf.info(str(path))
        if info.samplerate != config.SR or info.channels != 1 or info.frames != int(row.sample_count):
            raise M4DataError(f"Source audio metadata changed: {row.filename}")
    return split


def make_internal_partition(split: pd.DataFrame, output_path=config.M4_SPLIT,
                            summary_path=config.M4_SPLIT_SUMMARY, expected_training_count=1748) -> tuple[pd.DataFrame, dict]:
    train = split[split.partition.eq("train")].copy().reset_index(drop=True)
    if len(train) != expected_training_count:
        raise M4DataError(f"Expected {expected_training_count} M3 training recordings")
    stratify = train.primary_label.astype(str) + " × " + train.collection.astype(str)
    groups = train.group_id.astype(str)
    candidates = []
    splitter = StratifiedGroupKFold(n_splits=config.INTERNAL_SPLITS, shuffle=True, random_state=config.SEED)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, stratify, groups), start=1):
        inner_train, inner_val = train.iloc[tr_idx], train.iloc[va_idx]
        if inner_val.unknown_author.any():
            continue
        if set(inner_train.primary_label) != set(config.WORKING_LABELS) or set(inner_val.primary_label) != set(config.WORKING_LABELS):
            continue
        if set(inner_train.group_id) & set(inner_val.group_id):
            continue
        if set(inner_train.collection) != {"XC", "iNat"} or set(inner_val.collection) != {"XC", "iNat"}:
            continue
        if set(inner_train.primary_label.astype(str) + " × " + inner_train.collection.astype(str)) != set(stratify):
            continue
        if set(inner_val.primary_label.astype(str) + " × " + inner_val.collection.astype(str)) != set(stratify):
            continue
        candidates.append((abs(len(va_idx) / len(train) - config.INTERNAL_TARGET), fold, tr_idx, va_idx))
    if not candidates:
        raise M4DataError("No deterministic internal fold preserves class/collection coverage and author separation")
    _, chosen, tr_idx, va_idx = min(candidates, key=lambda x: (x[0], x[1]))
    internal = train.copy()
    internal["m4_partition"] = "internal_train"
    internal.loc[va_idx, "m4_partition"] = "internal_validation"
    internal["internal_fold"] = chosen
    holdout = split[split.partition.eq("validation")].copy()
    holdout["m4_partition"] = "holdout"
    holdout["internal_fold"] = chosen
    result = pd.concat([internal, holdout], ignore_index=True)
    if len(result) != len(split) or result.filename.duplicated().any():
        raise M4DataError("Internal split does not preserve original recording identities")
    check = result.groupby("m4_partition")
    summary = {"seed": config.SEED, "algorithm": "StratifiedGroupKFold", "n_splits": config.INTERNAL_SPLITS,
               "selected_fold": chosen, "stratification": "primary_label × collection",
               "group_key": "M3 connected component: normalized author across species/collections + SHA-256 duplicates",
               "dataset_recordings": len(result), "internal_train_recordings": int((result.m4_partition == "internal_train").sum()),
               "internal_validation_recordings": int((result.m4_partition == "internal_validation").sum()),
               "final_holdout_recordings": int((result.m4_partition == "holdout").sum()),
               "internal_validation_fraction_of_m3_train": len(va_idx) / len(train),
               "unknown_author_internal_train": int(result.loc[result.m4_partition.eq("internal_train"), "unknown_author"].sum()),
               "unknown_author_internal_validation": int(result.loc[result.m4_partition.eq("internal_validation"), "unknown_author"].sum()),
               "author_content_group_overlap": 0,
               "all_classes_in_both_internal_partitions": True,
               "all_observed_class_collection_strata_in_both_internal_partitions": True,
               "class_counts_by_partition": result.groupby(["m4_partition", "primary_label"]).size().rename("count").reset_index().to_dict("records"),
               "collection_counts_by_partition": result.groupby(["m4_partition", "collection"]).size().rename("count").reset_index().to_dict("records"),
               "holdout_reused_from_m3": True,
               "holdout_usage": "final scoring only; no fit, early stopping, architecture selection, or tuning"}
    atomic_write_csv(result, output_path)
    atomic_write_json(summary, summary_path)
    return result, summary


def make_segment_manifest(recording_split: pd.DataFrame, source_segments: pd.DataFrame,
                          output_path=config.M4_SEGMENTS) -> pd.DataFrame:
    partitions = recording_split[["filename", "m4_partition", "group_id", "author_group_id", "collection",
                                  "primary_label", "sha256", "unknown_author", "expected_size_bytes",
                                  "actual_size_bytes", "sample_count", "sample_rate", "channels"]]
    merged = source_segments.merge(partitions, on=["filename", "group_id", "collection", "primary_label", "sha256"],
                                   how="left", validate="many_to_one", suffixes=("_m3", ""))
    if merged.m4_partition.isna().any() or len(merged) != len(source_segments):
        raise M4DataError("Could not map every M3 segment to one M4 recording partition")
    merged["source_author_group_id"] = merged.author_group_id
    # Author identifiers remain hashed, as in the original M3 split.
    atomic_write_csv(merged, output_path)
    return merged


def sha_key(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24]
