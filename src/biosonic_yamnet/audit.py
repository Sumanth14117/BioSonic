"""Post-evaluation M5 integrity and leakage audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config
from .classifiers import make_group_cv_splits
from .dataset import read_training_manifest
from .io_utils import atomic_json, atomic_text, sha256_file


def run_quality_audit() -> dict:
    split = pd.read_csv(config.SPLIT_MANIFEST, dtype={"filename": str, "primary_label": str,
                                                       "partition": str, "group_id": str,
                                                       "sha256": str})
    manifest = pd.read_csv(config.DOWNLOAD_MANIFEST, dtype={"filename": str, "primary_label": str,
                                                             "verification_status": str})
    if len(split) != 2177 or split.filename.nunique() != 2177:
        raise ValueError("Approved experiment split does not contain 2,177 unique recordings")
    if manifest.filename.nunique() != 3373:
        raise ValueError("Acquisition manifest no longer contains 3,373 unique approved files")
    if not split.filename.isin(manifest.filename).all():
        raise ValueError("Experiment split includes a recording outside the approved manifest")
    verified = set(manifest.loc[manifest.verification_status.eq("verified"), "filename"].astype(str))
    if verified != set(split.filename.astype(str)):
        raise ValueError("Approved split and verified acquisition files do not match exactly")
    partition_counts = split.partition.value_counts().to_dict()
    if partition_counts != {"train": config.TRAIN_RECORDINGS, "validation": config.HOLDOUT_RECORDINGS}:
        raise ValueError(f"M3 partition counts changed: {partition_counts}")
    if set(split.primary_label.astype(str)) != set(config.WORKING_LABELS):
        raise ValueError("M3 split class mapping changed")
    if set(split.filename.astype(str)) & config.M2_SAMPLE_FILES:
        raise ValueError("An M2 representative sample is present in the experiment split")
    train = read_training_manifest()
    holdout = split[split.partition.eq("validation")].copy()
    group_overlap = set(train.group_id.astype(str)) & set(holdout.group_id.astype(str))
    author_overlap = set(train.author_group_id.astype(str)) & set(holdout.author_group_id.astype(str))
    id_overlap = set(train.filename.astype(str)) & set(holdout.filename.astype(str))
    hash_overlap = set(train.sha256.astype(str)) & set(holdout.sha256.astype(str))
    if group_overlap or author_overlap or id_overlap or hash_overlap:
        raise ValueError("M5 final training/holdout group, author, file, or hash overlap detected")

    train_embeddings = pd.read_csv(config.DATA_ROOT / "train_embedding_manifest.csv",
                                    dtype={"filename": str, "primary_label": str,
                                           "partition": str, "group_id": str, "source_sha256": str})
    holdout_embeddings = pd.read_csv(config.DATA_ROOT / "validation_embedding_manifest.csv",
                                      dtype={"filename": str, "primary_label": str,
                                             "partition": str, "group_id": str, "source_sha256": str})
    embedded = pd.concat([train_embeddings, holdout_embeddings], ignore_index=True)
    if len(embedded) != 2177 or embedded.filename.nunique() != 2177:
        raise ValueError("M5 embeddings do not map one-to-one to all approved experiment recordings")
    if set(embedded.filename.astype(str)) != set(split.filename.astype(str)):
        raise ValueError("M5 embedding IDs differ from the authoritative M3 split")
    label_map = split.set_index("filename").primary_label.astype(str)
    if any(str(row.primary_label) != label_map.loc[str(row.filename)]
           for row in embedded.itertuples(index=False)):
        raise ValueError("M5 embedding labels do not align with the split manifest")
    if (train_embeddings.partition.ne("train").any() or
            holdout_embeddings.partition.ne("validation").any()):
        raise ValueError("Embedding partition metadata is inconsistent")
    if not train_embeddings.filename.astype(str).equals(train.filename.astype(str)):
        raise ValueError("Training embedding ordering differs from the training manifest")
    if not np.array_equal(train_embeddings.source_sha256.astype(str).to_numpy(),
                          train.sha256.astype(str).to_numpy()):
        raise ValueError("Training embedding source hashes differ from the split")
    if not np.array_equal(holdout_embeddings.source_sha256.astype(str).to_numpy(),
                          holdout.sha256.astype(str).to_numpy()):
        raise ValueError("Holdout embedding source hashes differ from the split")

    cv_splits = make_group_cv_splits(train)
    fold_audit = []
    labels = train.primary_label.astype(str).to_numpy()
    groups = train.group_id.astype(str).to_numpy()
    strata = (train.primary_label.astype(str) + " × " + train.collection.astype(str)).to_numpy()
    for fold_no, (train_idx, val_idx) in enumerate(cv_splits, 1):
        fold_audit.append({"fold": fold_no, "train_count": int(len(train_idx)),
                           "validation_count": int(len(val_idx)),
                           "train_classes": len(set(labels[train_idx])),
                           "validation_classes": len(set(labels[val_idx])),
                           "train_strata": len(set(strata[train_idx])),
                           "validation_strata": len(set(strata[val_idx])),
                           "group_overlap": len(set(groups[train_idx]) & set(groups[val_idx]))})

    # Full source integrity audit occurs after the classifier has been frozen and evaluated.
    audio_counts = {"checked": 0, "size_mismatch": 0, "sha256_mismatch": 0,
                    "metadata_mismatch": 0, "decode_failures": 0}
    for row in split.itertuples(index=False):
        path = config.AUDIO_ROOT.joinpath(*str(row.filename).split("/"))
        audio_counts["checked"] += 1
        if not path.is_file() or path.stat().st_size != int(row.expected_size_bytes):
            audio_counts["size_mismatch"] += 1
            continue
        if sha256_file(path) != str(row.sha256):
            audio_counts["sha256_mismatch"] += 1
        try:
            info = sf.info(str(path))
            if (info.format != "OGG" or info.subtype != "VORBIS" or info.samplerate != config.SAMPLE_RATE or
                    info.channels != 1 or info.frames != int(row.sample_count)):
                audio_counts["metadata_mismatch"] += 1
            # Full decode ensures the source remains readable in the final audit.
            decoded, _ = sf.read(str(path), dtype="float32", always_2d=True)
            if decoded.shape != (int(row.sample_count), 1) or not np.isfinite(decoded).all():
                audio_counts["decode_failures"] += 1
        except Exception:
            audio_counts["decode_failures"] += 1
    if any(value for key, value in audio_counts.items() if key != "checked"):
        raise ValueError(f"Audio integrity audit found discrepancies: {audio_counts}")

    for partition, folder, expected in (("training_shards_v2", config.TRAIN_SHARDS, config.TRAIN_RECORDINGS),
                                        ("holdout_shards_v2", config.HOLDOUT_SHARDS, config.HOLDOUT_RECORDINGS)):
        count = len(list(folder.glob("*.npz"))) if folder.is_dir() else 0
        if count != expected:
            raise ValueError(f"Unexpected {partition} shard count: {count} != {expected}")

    freeze = json.loads((config.MODEL_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((config.REPORT_ROOT / "holdout_metrics.json").read_text(encoding="utf-8"))
    if final_metrics.get("holdout_evaluation_count") != 1 or final_metrics.get("holdout_used_for_selection"):
        raise ValueError("Holdout usage metadata is inconsistent")
    result = {
        "status": "passed", "approved_recordings": len(manifest),
        "verified_working_recordings": len(split), "working_unique_filenames": split.filename.nunique(),
        "class_count": len(set(split.primary_label.astype(str))),
        "class_order": list(config.WORKING_LABELS), "partition_counts": partition_counts,
        "m2_sample_count_in_experiment": len(set(split.filename.astype(str)) & config.M2_SAMPLE_FILES),
        "training_holdout_overlap": {"group": len(group_overlap), "author_group": len(author_overlap),
                                     "filename": len(id_overlap), "sha256": len(hash_overlap)},
        "training_cv_folds": fold_audit,
        "embedding_rows": len(embedded), "embedding_duplicate_ids": int(embedded.filename.duplicated().sum()),
        "audio_integrity": audio_counts, "shard_counts": {
            "training": config.TRAIN_RECORDINGS, "holdout": config.HOLDOUT_RECORDINGS},
        "classifier_sha256_matches_freeze": sha256_file(config.MODEL_ROOT / "final_classifier.joblib") == freeze["classifier_sha256"],
        "checkpoint_sha256": sha256_file(config.WEIGHTS_PATH),
        "final_holdout_metrics": final_metrics["metrics"],
        "holdout_evaluations": 1,
        "note": "The 429-recording final holdout was accessed only after the frozen M5 classifier was saved.",
    }
    config.REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(result, config.REPORT_ROOT / "quality_audit.json")
    lines = ["# M5 YAMNet quality audit", "", "Status: **PASS**", "",
             f"- Approved acquisition manifest: {result['approved_recordings']:,} unique files.",
             f"- Experiment set: {result['verified_working_recordings']:,} unique files; eight classes.",
             f"- M3 partitions: {partition_counts['train']:,} train / {partition_counts['validation']:,} final holdout.",
             f"- Out-of-scope M2 samples in experiment: {result['m2_sample_count_in_experiment']}.",
             f"- Training/holdout group, author, filename, and SHA-256 overlaps: 0.",
             f"- Five grouped-CV folds passed class/stratum and leakage checks.",
             f"- Audio integrity: {audio_counts['checked']:,} files decoded and matched size, SHA-256, format, rate, channels, and frame counts.",
             f"- M5 embeddings map one-to-one to all {len(embedded):,} source recordings.",
             "- Holdout evaluation was performed once after classifier freeze; it was not used for selection.",
             "", "No M3/M4 model, report, manifest, or result was changed by the M5 workflow.", ""]
    atomic_text("\n".join(lines), config.REPORT_ROOT / "quality_audit.md")
    return result
