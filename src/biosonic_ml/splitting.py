import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def make_holdout(frame: pd.DataFrame, manifest_path=config.SPLIT_MANIFEST,
                 summary_path=config.SPLIT_SUMMARY, strategy_path=config.STRATEGY_REPORT) -> tuple[pd.DataFrame, dict]:
    if set(frame.primary_label) != set(config.WORKING_LABELS):
        raise ValueError("Expected all eight present classes")
    y = frame.primary_label.astype(str) + " × " + frame.collection.astype(str)
    groups = frame.group_id.astype(str)
    unknown = frame.author_group_key.eq("__unknown__")
    candidates = []
    splitter = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    all_strata = set(y)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y, groups), start=1):
        train_rows, val_rows = frame.iloc[train_idx], frame.iloc[val_idx]
        if unknown.iloc[val_idx].any():
            continue
        if set(train_rows.primary_label) != set(config.WORKING_LABELS) or set(val_rows.primary_label) != set(config.WORKING_LABELS):
            continue
        if set(y.iloc[train_idx]) != all_strata or set(y.iloc[val_idx]) != all_strata:
            continue
        if set(train_rows.group_id) & set(val_rows.group_id):
            continue
        ratio = len(val_idx) / len(frame)
        candidates.append((abs(ratio - config.SPLIT_TARGET), fold, train_idx, val_idx, ratio))
    if not candidates:
        raise RuntimeError("No grouped fold satisfies unknown-author and label×collection coverage constraints")
    _, selected_fold, train_idx, val_idx, ratio = min(candidates, key=lambda x: (x[0], x[1]))
    output = frame[["filename", "primary_label", "collection", "group_id", "author_group_id", "author_group_key",
                    "sha256", "expected_size_bytes", "actual_size_bytes", "sample_rate", "channels",
                    "sample_count", "duration_seconds"]].copy()
    output["partition"] = "train"
    output.loc[val_idx, "partition"] = "validation"
    # Do not publish source author names. The literal Unknown status is retained for audit.
    output["unknown_author"] = output.author_group_key.eq("__unknown__")
    output = output.drop(columns="author_group_key").sort_values(["partition", "primary_label", "filename"], kind="stable")
    train_out, val_out = output[output.partition.eq("train")], output[output.partition.eq("validation")]
    assert not set(train_out.group_id) & set(val_out.group_id)
    assert not val_out.unknown_author.any()
    counts = output.groupby(["partition", "primary_label", "collection"], observed=True).size().rename("recordings").reset_index()
    species_counts = output.groupby(["partition", "primary_label"], observed=True).size().rename("recordings").reset_index()
    summary = {"dataset_recordings": len(output), "classes": list(config.WORKING_LABELS), "class_count": len(config.WORKING_LABELS),
              "seed": config.SEED, "splitter": "StratifiedGroupKFold", "candidate_folds": config.N_SPLITS,
              "selected_fold": selected_fold, "stratification_target": "primary_label × collection",
              "validation_recordings": len(val_out), "training_recordings": len(train_out),
              "validation_fraction": ratio, "validation_percent": ratio * 100,
              "unknown_author_training": int(train_out.unknown_author.sum()), "unknown_author_validation": 0,
              "known_group_overlap": 0, "all_label_collection_strata_present_in_both": True,
              "species_counts": species_counts.to_dict("records"), "species_collection_counts": counts.to_dict("records"),
              "note": "Fresh split for the verified 2,177-recording eight-class interim set; not the earlier 12-class fold-2 split."}
    atomic_write_csv(output, Path(manifest_path))
    atomic_write_json(summary, Path(summary_path))
    text = f"""# Eight-class interim validation strategy

This split applies only to the **{len(output):,} verified, locally available approved recordings**. It includes the eight classes present in that working set: {', '.join(config.WORKING_LABELS)}. It is not a 12-class result; `swtman1`, `smbani`, `yecpar`, and `spispi1` are unavailable and excluded from this experiment. The six M2 sample recordings outside the approved manifest are excluded.

## Split

- Fresh deterministic grouped holdout from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; selected candidate fold {selected_fold} as the feasible fold nearest 20% validation.
- Training: {len(train_out):,}; validation: {len(val_out):,} ({ratio:.2%}).
- Stratification: `primary_label × collection` (XC/iNat).
- Group key: connected components of normalized author identity and exact SHA-256 duplicate identity. All literal `Unknown` authors form one group and remain in training ({int(train_out.unknown_author.sum())} recordings).
- All eight labels and all observed label×collection strata occur in both partitions. No author/content group crosses partitions.

## Leakage controls and limits

Split source recordings before segmentation. Any derived segment, feature vector, or transformation inherits its source partition. Learned preprocessing and feature selection are fit only on training folds. Group-aware cross-validation and model selection use training data only; this validation holdout remains untouched until final evaluation. Author grouping reduces same-contributor leakage but does not guarantee geographic independence or prevent near-duplicate recordings by different authors. The 109 Unknown-author recordings are training-only, limiting evaluation of unseen/unknown contributors. Spatial grouping is a separate future analysis.

The split is valid for the current eight-class subset only. It must be regenerated if the working dataset changes; the historic 676-recording holdout from the full intended dataset is not reused.
"""
    atomic_write_text(Path(strategy_path), text)
    return output.reset_index(drop=True), summary
