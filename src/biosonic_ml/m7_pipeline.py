"""Reproducible, site-aware M7 YAMNet soundscape experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .m7_soundscape import load_soundscape_annotations, build_multihot_targets


SITE_RE = re.compile(r"_S(\d+)_")
SOURCE_RATE = 32_000
MODEL_RATE = 16_000
INTERVAL_SECONDS = 5
EMBEDDING_DIM = 1_024
MIN_POSITIVE_INTERVALS = 20
MIN_POSITIVE_RECORDINGS = 5
MIN_POSITIVE_SITES = 3
THRESHOLD_GRID = np.arange(0.05, 0.951, 0.05)
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def site_from_filename(filename: str) -> str:
    match = SITE_RE.search(Path(filename).name)
    if not match:
        raise ValueError(f"Cannot derive site group from soundscape filename: {filename}")
    return match.group(1)


def species_support(
    windows: pd.DataFrame, targets: np.ndarray, labels: Sequence[str]
) -> pd.DataFrame:
    if targets.shape != (len(windows), len(labels)):
        raise ValueError("Target shape does not match windows and class labels")
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        positive = targets[:, index].astype(bool)
        rows.append({
            "primary_label": label,
            "positive_intervals": int(positive.sum()),
            "positive_recordings": int(windows.loc[positive, "filename"].nunique()),
            "positive_sites": int(windows.loc[positive, "site_group"].nunique()),
            "positive_percent": float(100 * positive.mean()) if len(positive) else 0.0,
            "negative_labeled_intervals": int((~positive).sum()),
            "eligible": bool(
                positive.sum() >= MIN_POSITIVE_INTERVALS
                and windows.loc[positive, "filename"].nunique() >= MIN_POSITIVE_RECORDINGS
                and windows.loc[positive, "site_group"].nunique() >= MIN_POSITIVE_SITES
            ),
        })
    return pd.DataFrame(rows)


def build_interval_dataset(
    annotation_path: Path,
    taxonomy_path: Path,
    acquisition_manifest_path: Path,
    audio_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...], pd.DataFrame, dict[str, str]]:
    """Validate local source data and return all explicit 5-second labels."""
    taxonomy = pd.read_csv(taxonomy_path, dtype=str)
    annotations = load_soundscape_annotations(annotation_path, taxonomy)
    acquisition = pd.read_csv(acquisition_manifest_path, dtype={"filename": str})
    if len(acquisition) != 66 or acquisition.filename.nunique() != 66:
        raise ValueError("M7 acquisition manifest must contain exactly 66 unique recordings")
    required = {"filename", "expected_size_bytes", "actual_size_bytes", "verification_status", "sample_rate", "channels", "duration_seconds", "sha256"}
    if not required.issubset(acquisition.columns):
        raise ValueError(f"Acquisition manifest lacks columns: {sorted(required - set(acquisition.columns))}")
    if not acquisition.verification_status.eq("verified").all():
        raise ValueError("Not all approved soundscapes are marked verified")
    annotated_files = set(annotations.filename)
    if annotated_files != set(acquisition.filename):
        raise ValueError("Annotation filenames do not exactly match the verified 66-file manifest")

    audio_hashes: dict[str, str] = {}
    audio_status: dict[str, dict[str, object]] = {}
    for row in acquisition.itertuples(index=False):
        path = audio_root / row.filename
        if not path.is_file():
            raise FileNotFoundError(f"Verified soundscape is missing: {row.filename}")
        if path.stat().st_size != int(row.expected_size_bytes) or int(row.actual_size_bytes) != int(row.expected_size_bytes):
            raise ValueError(f"Soundscape size mismatch: {row.filename}")
        actual_hash = sha256_file(path)
        if actual_hash.lower() != str(row.sha256).lower():
            raise ValueError(f"Soundscape SHA-256 mismatch: {row.filename}")
        info = sf.info(str(path))
        if (info.samplerate, info.channels, info.format) != (SOURCE_RATE, 1, "OGG") or abs(info.duration - 60.0) > 1e-6:
            raise ValueError(f"Unexpected soundscape audio metadata: {row.filename}")
        audio, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
        if sample_rate != SOURCE_RATE or audio.shape != (SOURCE_RATE * 60, 1):
            raise ValueError(f"Decoded audio dimensions mismatch: {row.filename}")
        audio_hashes[row.filename] = actual_hash
        audio_status[row.filename] = {
            "audio_verification_status": "verified",
            "audio_sha256": actual_hash,
            "sample_rate": SOURCE_RATE,
            "channels": 1,
            "sample_count": int(audio.shape[0]),
            "duration_seconds": 60.0,
        }

    taxonomy_class = taxonomy.set_index("primary_label").class_name
    all_birds = tuple(sorted({label for label_values in annotations.labels for label in label_values if taxonomy_class[label] == "Aves"}))
    windows, targets, labels = build_multihot_targets(annotations, taxonomy, class_labels=all_birds)
    windows["site_group"] = windows.filename.map(site_from_filename)
    windows["label_status"] = "annotated"
    windows["original_annotation_codes"] = annotations.labels.map(lambda values: ";".join(values)).to_numpy()
    windows["start"] = annotations.start.to_numpy()
    windows["end"] = annotations.end.to_numpy()
    windows["audio_verification_status"] = windows.filename.map(lambda name: audio_status[name]["audio_verification_status"])
    # Explicitly annotated rows with no bird label are negative for bird targets;
    # absent timestamp rows are not created anywhere in this table.
    for index, label in enumerate(labels):
        windows[f"target_{label}"] = targets[:, index].astype(np.uint8)
    support = species_support(windows, targets, labels)
    if windows.duplicated(["filename", "start_seconds", "end_seconds"]).any():
        raise ValueError("Duplicate annotation intervals remain after exact deduplication")
    if not np.isin(targets, [0, 1]).all() or targets.shape != (len(windows), len(labels)):
        raise ValueError("Invalid multi-label target matrix")
    return windows, targets, labels, support, audio_hashes


def _pcm16_segment(path: Path, start_seconds: int, end_seconds: int) -> tuple[np.ndarray, dict[str, object]]:
    """Read a native-rate segment, then apply the unchanged M5 PCM16/resampy path."""
    frame_start = start_seconds * SOURCE_RATE
    frame_end = end_seconds * SOURCE_RATE
    info = sf.info(str(path))
    if info.samplerate != SOURCE_RATE or info.channels != 1 or frame_end > info.frames:
        raise ValueError(f"Invalid source interval bounds for {path.name} {start_seconds}:{end_seconds}")
    pcm, rate = sf.read(str(path), start=frame_start, frames=frame_end - frame_start, dtype="int16", always_2d=True)
    if rate != SOURCE_RATE or pcm.shape != (INTERVAL_SECONDS * SOURCE_RATE, 1):
        raise ValueError(f"Could not read exact interval samples for {path.name} {start_seconds}")
    import resampy

    waveform = pcm[:, 0].astype(np.float32) / np.float32(32768.0)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite decoded waveform: {path.name}")
    waveform = resampy.resample(waveform, SOURCE_RATE, MODEL_RATE).astype(np.float32)
    if waveform.shape != (INTERVAL_SECONDS * MODEL_RATE,) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid resampled interval for {path.name} at {start_seconds}")
    diagnostics = {
        "source_min": float(pcm.min() / 32768.0),
        "source_max": float(pcm.max() / 32768.0),
        "resampled_peak_abs": float(np.max(np.abs(waveform), initial=0.0)),
        "resampled_samples_outside_unit_interval": int(np.count_nonzero(np.abs(waveform) > 1.0)),
    }
    return waveform, diagnostics


def extract_interval_features(
    windows: pd.DataFrame,
    audio_root: Path,
    encoder: object,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract mean-pooled 1,024-D YAMNet embeddings per labeled interval."""
    vectors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for index, row in enumerate(windows.itertuples(index=False), start=1):
        waveform, diagnostics = _pcm16_segment(
            audio_root / row.filename, int(row.start_seconds), int(row.end_seconds)
        )
        _, patch_embeddings, _ = encoder.infer(waveform)
        if patch_embeddings.ndim != 2 or patch_embeddings.shape[1] != EMBEDDING_DIM or patch_embeddings.shape[0] == 0:
            raise ValueError(f"Unexpected YAMNet embedding shape for {row.filename} at {row.start_seconds}: {patch_embeddings.shape}")
        if not np.isfinite(patch_embeddings).all():
            raise ValueError(f"Non-finite YAMNet embedding for {row.filename} at {row.start_seconds}")
        pooled = np.mean(patch_embeddings.astype(np.float64), axis=0).astype(np.float32)
        if pooled.shape != (EMBEDDING_DIM,) or not np.isfinite(pooled).all():
            raise ValueError(f"Invalid pooled embedding for {row.filename} at {row.start_seconds}")
        vectors.append(pooled)
        records.append({
            "filename": row.filename,
            "start_seconds": int(row.start_seconds),
            "end_seconds": int(row.end_seconds),
            "site_group": row.site_group,
            "yamnet_patch_count": int(patch_embeddings.shape[0]),
            **diagnostics,
        })
        if progress:
            progress(index, len(windows), row.filename)
    matrix = np.stack(vectors) if vectors else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    if matrix.shape != (len(windows), EMBEDDING_DIM) or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix failed dimension/finiteness checks")
    return matrix, pd.DataFrame(records)


def make_group_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED),
    )


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.uint8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or np.unique(y).size != 2:
        raise ValueError("Threshold selection requires finite OOF probabilities and both target classes")
    scores = np.asarray([f1_score(y, p >= threshold, zero_division=0) for threshold in THRESHOLD_GRID])
    best = scores.max()
    candidates = THRESHOLD_GRID[np.isclose(scores, best)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def nested_site_loso(
    features: np.ndarray,
    targets: np.ndarray,
    windows: pd.DataFrame,
    labels: Sequence[str],
) -> dict[str, object]:
    """Nested site-LOSO predictions; outer sites are never threshold-tuned."""
    X = np.asarray(features, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.uint8)
    if X.ndim != 2 or X.shape[0] != len(windows) or Y.shape != (len(windows), len(labels)):
        raise ValueError("Features, windows, targets, and labels have inconsistent dimensions")
    if not np.isfinite(X).all() or not np.isin(Y, [0, 1]).all():
        raise ValueError("Features/targets contain invalid values")
    groups = windows.site_group.astype(str).to_numpy()
    unique_groups = tuple(sorted(np.unique(groups)))
    if len(unique_groups) < 3:
        raise ValueError("At least three sites are required for site-aware evaluation")
    support = species_support(windows, Y, labels).set_index("primary_label")
    eligible = tuple(label for label in labels if bool(support.loc[label, "eligible"]))
    if not eligible:
        raise ValueError("No bird labels meet the predeclared support rule")
    col_map = {label: i for i, label in enumerate(labels)}
    eligible_cols = [col_map[label] for label in eligible]
    y = Y[:, eligible_cols]
    probabilities = np.full(y.shape, np.nan, dtype=np.float64)
    fold_thresholds: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    fold_diagnostics: list[dict[str, object]] = []

    for outer_fold, held_site in enumerate(unique_groups, start=1):
        test = groups == held_site
        train = ~test
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Site leakage in outer split")
        inner_probs = np.full((int(train.sum()), len(eligible)), np.nan, dtype=np.float64)
        train_indices = np.flatnonzero(train)
        for inner_site in sorted(set(groups[train])):
            inner_valid_local = groups[train][np.arange(len(train_indices))] == inner_site
            inner_train_local = ~inner_valid_local
            for class_index in range(len(eligible)):
                y_train = y[train_indices[inner_train_local], class_index]
                if np.unique(y_train).size != 2:
                    raise ValueError(f"Inner training partition lacks both classes for {eligible[class_index]} (outer={held_site}, inner={inner_site})")
                model = make_group_model()
                model.fit(X[train_indices[inner_train_local]], y_train)
                inner_probs[inner_valid_local, class_index] = model.predict_proba(X[train_indices[inner_valid_local]])[:, 1]
        if not np.isfinite(inner_probs).all():
            raise RuntimeError("Inner site-OOF threshold predictions are incomplete")
        thresholds = [choose_f1_threshold(y[train, class_index], inner_probs[:, class_index]) for class_index in range(len(eligible))]
        fold_thresholds.append({"outer_fold": outer_fold, "held_out_site": held_site, **dict(zip(eligible, thresholds))})
        for class_index, label in enumerate(eligible):
            y_train = y[train, class_index]
            if np.unique(y_train).size != 2:
                raise ValueError(f"Outer training partition lacks both classes for {label}, held site {held_site}")
            model = make_group_model()
            model.fit(X[train], y_train)
            probabilities[test, class_index] = model.predict_proba(X[test])[:, 1]
        if not np.isfinite(probabilities[test]).all():
            raise RuntimeError(f"Outer predictions are incomplete for site {held_site}")
        fold_diagnostics.append({
            "outer_fold": outer_fold,
            "held_out_site": held_site,
            "train_intervals": int(train.sum()),
            "validation_intervals": int(test.sum()),
            "train_sites": len(set(groups[train])),
            "validation_sites": len(set(groups[test])),
        })
        for idx, row in enumerate(windows.itertuples(index=False)):
            split_rows.append({
                "outer_fold": outer_fold,
                "held_out_site": held_site,
                "filename": row.filename,
                "start_seconds": int(row.start_seconds),
                "end_seconds": int(row.end_seconds),
                "site_group": row.site_group,
                "partition": "validation" if test[idx] else "train",
            })

    y_eval = y
    y_pred = np.zeros(y.shape, dtype=np.uint8)
    row_fold = np.asarray([unique_groups.index(group) for group in groups])
    threshold_array = np.zeros(len(eligible), dtype=np.float64)
    for class_index, label in enumerate(eligible):
        threshold_array[class_index] = float(np.median([row[label] for row in fold_thresholds]))
        for row_index, group in enumerate(groups):
            fold = fold_thresholds[row_fold[row_index]]
            y_pred[row_index, class_index] = probabilities[row_index, class_index] >= float(fold[label])

    per_species = []
    for index, label in enumerate(eligible):
        per_species.append({
            "primary_label": label,
            "positive_support": int(y_eval[:, index].sum()),
            "negative_support": int((1 - y_eval[:, index]).sum()),
            "threshold_range_min": float(min(row[label] for row in fold_thresholds)),
            "threshold_range_max": float(max(row[label] for row in fold_thresholds)),
            "threshold_median": float(threshold_array[index]),
            "precision": float(precision_score(y_eval[:, index], y_pred[:, index], zero_division=0)),
            "recall": float(recall_score(y_eval[:, index], y_pred[:, index], zero_division=0)),
            "f1": float(f1_score(y_eval[:, index], y_pred[:, index], zero_division=0)),
            "average_precision": float(average_precision_score(y_eval[:, index], probabilities[:, index])),
            "predicted_positive": int(y_pred[:, index].sum()),
        })
    site_metrics = []
    for site in unique_groups:
        rows = groups == site
        for index, label in enumerate(eligible):
            positives = int(y_eval[rows, index].sum())
            base: dict[str, object] = {"site_group": site, "primary_label": label, "intervals": int(rows.sum()), "positive_support": positives}
            if positives == 0:
                base.update({"precision": None, "recall": None, "f1": None, "average_precision": None, "undefined_reason": "no positive support at this site"})
            else:
                base.update({
                    "precision": float(precision_score(y_eval[rows, index], y_pred[rows, index], zero_division=0)),
                    "recall": float(recall_score(y_eval[rows, index], y_pred[rows, index], zero_division=0)),
                    "f1": float(f1_score(y_eval[rows, index], y_pred[rows, index], zero_division=0)),
                    "average_precision": float(average_precision_score(y_eval[rows, index], probabilities[rows, index])) if np.unique(y_eval[rows, index]).size == 2 else None,
                    "undefined_reason": "AP unavailable when site lacks negatives" if np.unique(y_eval[rows, index]).size != 2 else "",
                })
            site_metrics.append(base)

    errors: list[dict[str, object]] = []
    for row_index, row in enumerate(windows.itertuples(index=False)):
        for index, label in enumerate(eligible):
            if y_pred[row_index, index] != y_eval[row_index, index]:
                errors.append({
                    "filename": row.filename,
                    "site_group": row.site_group,
                    "start_seconds": int(row.start_seconds),
                    "end_seconds": int(row.end_seconds),
                    "primary_label": label,
                    "error_type": "false_positive" if y_pred[row_index, index] else "false_negative",
                    "probability": float(probabilities[row_index, index]),
                    "fold_threshold": float(fold_thresholds[row_fold[row_index]][label]),
                })
    macro_f1 = float(np.mean([item["f1"] for item in per_species]))
    macro_precision = float(np.mean([item["precision"] for item in per_species]))
    macro_recall = float(np.mean([item["recall"] for item in per_species]))
    macro_ap = float(np.mean([item["average_precision"] for item in per_species]))
    return {
        "eligible_labels": list(eligible),
        "probabilities": probabilities,
        "targets": y_eval,
        "predictions": y_pred,
        "thresholds": threshold_array,
        "fold_thresholds": fold_thresholds,
        "split_manifest": pd.DataFrame(split_rows),
        "folds": fold_diagnostics,
        "per_species": pd.DataFrame(per_species),
        "per_site": pd.DataFrame(site_metrics),
        "errors": pd.DataFrame(errors),
        "metrics": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": float(precision_score(y_eval.ravel(), y_pred.ravel(), zero_division=0)),
            "micro_recall": float(recall_score(y_eval.ravel(), y_pred.ravel(), zero_division=0)),
            "micro_f1": float(f1_score(y_eval.ravel(), y_pred.ravel(), zero_division=0)),
            "macro_average_precision": macro_ap,
            "micro_average_precision": float(average_precision_score(y_eval.ravel(), probabilities.ravel())),
            "interval_count": int(len(y_eval)),
            "class_label_count": int(len(eligible)),
            "site_count": int(len(unique_groups)),
        },
    }


def fit_final_models(features: np.ndarray, targets: np.ndarray, windows: pd.DataFrame, labels: Sequence[str]) -> tuple[dict[str, object], dict[str, float]]:
    """Fit final development models; thresholds come from site-OOF predictions."""
    groups = windows.site_group.astype(str).to_numpy()
    models: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    for index, label in enumerate(labels):
        y = targets[:, index]
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for held_site in sorted(set(groups)):
            train, valid = groups != held_site, groups == held_site
            model = make_group_model()
            model.fit(features[train], y[train])
            oof[valid] = model.predict_proba(features[valid])[:, 1]
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Final threshold OOF predictions incomplete for {label}")
        thresholds[label] = choose_f1_threshold(y, oof)
        final_model = make_group_model()
        final_model.fit(features, y)
        models[label] = final_model
    return models, thresholds


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_feature_cache(
    cache_path: Path,
    metadata_path: Path,
    features: np.ndarray,
    feature_rows: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or not np.isfinite(matrix).all():
        raise ValueError("Refusing to cache invalid feature matrix")
    if len(feature_rows) != len(matrix):
        raise ValueError("Feature row manifest does not align with feature matrix")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, embeddings=matrix)
    temporary.replace(cache_path)
    feature_rows.to_csv(cache_path.with_name("interval_feature_manifest.csv"), index=False)
    write_json(metadata_path, metadata)


def load_feature_cache(
    cache_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, object],
    expected_rows: int,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Reuse a cache only when model, source hashes, config, shape, and rows match."""
    rows_path = cache_path.with_name("interval_feature_manifest.csv")
    present = [path.exists() for path in (cache_path, metadata_path, rows_path)]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("M7 feature cache is incomplete; refusing silent reuse or overwrite")
    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    if stored != expected_metadata:
        raise ValueError("M7 feature cache metadata differs from current source/model/config")
    with np.load(cache_path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings"}:
            raise ValueError("Unexpected arrays in M7 feature cache")
        features = archive["embeddings"]
    feature_rows = pd.read_csv(rows_path)
    if features.shape != (expected_rows, EMBEDDING_DIM) or len(feature_rows) != expected_rows or not np.isfinite(features).all():
        raise ValueError("M7 feature cache shape/finiteness/row validation failed")
    required_keys = {"filename", "start_seconds", "end_seconds"}
    if not required_keys.issubset(feature_rows.columns):
        raise ValueError("M7 feature mapping lacks interval identity columns")
    key_text = "\n".join(
        f"{row.filename}|{int(row.start_seconds)}|{int(row.end_seconds)}"
        for row in feature_rows.itertuples(index=False)
    )
    key_hash = hashlib.sha256(key_text.encode()).hexdigest()
    if key_hash != expected_metadata.get("interval_keys_sha256"):
        raise ValueError("M7 feature cache interval mapping does not match metadata")
    return features.astype(np.float32, copy=False), feature_rows
