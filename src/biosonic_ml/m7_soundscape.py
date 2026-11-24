"""Validated annotation and output utilities for the M7 soundscape task.

This module deliberately does not load soundscape audio or train a model. The
local project currently has annotation metadata but none of the corresponding
soundscape recordings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config


ANNOTATION_COLUMNS = ("filename", "start", "end", "primary_label")
_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})$")


class SoundscapeAnnotationError(ValueError):
    """Raised when soundscape annotations cannot be interpreted safely."""


def timestamp_seconds(value: object) -> int:
    """Parse an annotation timestamp in strict HH:MM:SS form."""
    match = _TIME_RE.fullmatch(str(value).strip())
    if not match:
        raise SoundscapeAnnotationError(f"Invalid HH:MM:SS timestamp: {value!r}")
    hours, minutes, seconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise SoundscapeAnnotationError(f"Out-of-range timestamp component: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def parse_label_cell(value: object) -> tuple[str, ...]:
    """Parse semicolon-separated labels without inventing empty targets."""
    if not isinstance(value, str) or not value.strip():
        raise SoundscapeAnnotationError("An annotated interval has an empty label cell")
    labels = tuple(part.strip() for part in value.split(";"))
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise SoundscapeAnnotationError(f"Malformed or repeated labels in {value!r}")
    return labels


def load_soundscape_annotations(
    annotation_path: str | Path,
    taxonomy: pd.DataFrame,
    *,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """Load, validate, and exact-deduplicate official window annotations.

    Returned rows represent only intervals explicitly present in the CSV.
    Missing intervals are never synthesized as negative examples.
    """
    source = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    return validate_soundscape_annotations(source, taxonomy, interval_seconds=interval_seconds)


def validate_soundscape_annotations(
    source: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """Validate annotation schema, label mappings, intervals, and duplicates."""
    missing = set(ANNOTATION_COLUMNS) - set(source.columns)
    if missing:
        raise SoundscapeAnnotationError(f"Missing annotation columns: {sorted(missing)}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if "primary_label" not in taxonomy.columns or "class_name" not in taxonomy.columns:
        raise SoundscapeAnnotationError("Taxonomy requires primary_label and class_name columns")
    if taxonomy.primary_label.astype(str).duplicated().any():
        raise SoundscapeAnnotationError("Taxonomy primary_label values must be unique")

    rows = source.loc[:, ANNOTATION_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        rows[column] = rows[column].astype(str).str.strip()
    if rows.eq("").any().any():
        raise SoundscapeAnnotationError("Annotation rows may not contain empty required fields")
    if rows.filename.map(lambda name: Path(name).name != name).any():
        raise SoundscapeAnnotationError("Annotation filename must be a basename, not a path")

    rows["labels"] = rows.primary_label.map(parse_label_cell)
    taxonomy_labels = set(taxonomy.primary_label.astype(str))
    observed = {label for labels in rows.labels for label in labels}
    unmapped = observed - taxonomy_labels
    if unmapped:
        raise SoundscapeAnnotationError(f"Labels absent from taxonomy: {sorted(unmapped)}")

    rows["start_seconds"] = rows.start.map(timestamp_seconds)
    rows["end_seconds"] = rows.end.map(timestamp_seconds)
    durations = rows.end_seconds - rows.start_seconds
    if (rows.start_seconds < 0).any() or (durations != interval_seconds).any():
        raise SoundscapeAnnotationError(
            f"Every annotated interval must have duration {interval_seconds} seconds"
        )

    key = ["filename", "start", "end"]
    conflicts = rows.groupby(key, dropna=False).primary_label.nunique()
    if conflicts.gt(1).any():
        raise SoundscapeAnnotationError("Conflicting labels exist for the same soundscape interval")
    rows = rows.drop_duplicates(list(ANNOTATION_COLUMNS), keep="first")
    rows = rows.sort_values(["filename", "start_seconds", "end_seconds"], kind="stable").reset_index(drop=True)

    for filename, group in rows.groupby("filename", sort=False):
        starts = group.start_seconds.to_numpy()
        ends = group.end_seconds.to_numpy()
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
            raise SoundscapeAnnotationError(f"Overlapping annotated intervals in {filename}")
    return rows


def build_multihot_targets(
    annotations: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    class_name: str = "Aves",
    class_labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Build targets only for labeled intervals and requested taxonomy class.

    The returned window table retains file and half-open time boundaries. A
    row containing no requested-class labels receives an all-zero target only
    because that interval itself is present in the supplied annotation file.
    """
    if not {"labels", "filename", "start_seconds", "end_seconds"}.issubset(annotations.columns):
        raise SoundscapeAnnotationError("Use validated annotations before building targets")
    required = {"primary_label", "class_name"}
    if not required.issubset(taxonomy.columns):
        raise SoundscapeAnnotationError("Taxonomy requires primary_label and class_name columns")
    taxonomy_class = taxonomy.set_index(taxonomy.primary_label.astype(str)).class_name.astype(str)
    observed_labels = {label for labels in annotations.labels for label in labels}
    selected = tuple(sorted(
        label for label in observed_labels if taxonomy_class.get(label) == class_name
    )) if class_labels is None else tuple(map(str, class_labels))
    if len(selected) != len(set(selected)):
        raise SoundscapeAnnotationError("Class label order must be unique")
    wrong_class = [label for label in selected if taxonomy_class.get(label) != class_name]
    if wrong_class:
        raise SoundscapeAnnotationError(f"Requested labels are not {class_name}: {wrong_class}")

    index = {label: column for column, label in enumerate(selected)}
    target = np.zeros((len(annotations), len(selected)), dtype=np.uint8)
    for row_index, labels in enumerate(annotations.labels):
        for label in labels:
            if label not in taxonomy_class:
                raise SoundscapeAnnotationError(f"Label absent from taxonomy: {label}")
            if label in index:
                target[row_index, index[label]] = 1
    windows = annotations[["filename", "start_seconds", "end_seconds"]].copy().reset_index(drop=True)
    return windows, target, selected


def assert_group_disjoint(train: pd.DataFrame, validation: pd.DataFrame, group_column: str = "group_id") -> None:
    """Reject any recording/source group represented in both partitions."""
    if group_column not in train or group_column not in validation:
        raise ValueError(f"Both partitions require {group_column!r}")
    if train[group_column].isna().any() or validation[group_column].isna().any():
        raise SoundscapeAnnotationError(f"Partition group column {group_column!r} may not be missing")
    train_groups = set(train[group_column].dropna().astype(str))
    validation_groups = set(validation[group_column].dropna().astype(str))
    if "" in train_groups or "" in validation_groups:
        raise SoundscapeAnnotationError(f"Partition group column {group_column!r} may not be empty")
    overlap = train_groups & validation_groups
    if overlap:
        raise SoundscapeAnnotationError(f"Group leakage detected: {sorted(overlap)}")


def sigmoid_probabilities(logits: np.ndarray) -> np.ndarray:
    """Apply independent sigmoid probabilities (never a cross-class softmax)."""
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite")
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def format_window_predictions(
    windows: pd.DataFrame,
    probabilities: np.ndarray,
    class_labels: Sequence[str],
    thresholds: float | Sequence[float] = 0.5,
) -> list[dict[str, object]]:
    """Format independent per-class probabilities for timestamped windows."""
    required = {"filename", "start_seconds", "end_seconds"}
    if not required.issubset(windows.columns):
        raise ValueError(f"windows require columns: {sorted(required)}")
    labels = tuple(map(str, class_labels))
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("class_labels must be nonempty and unique")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(windows), len(labels)):
        raise ValueError("Probability matrix shape does not match windows and classes")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    threshold = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (len(labels),))
    if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
        raise ValueError("Thresholds must be finite and in [0, 1]")

    predictions: list[dict[str, object]] = []
    for index, row in enumerate(windows.itertuples(index=False)):
        if int(row.start_seconds) < 0 or int(row.end_seconds) <= int(row.start_seconds):
            raise ValueError("Window timestamps must form a positive-duration interval")
        per_species = {label: float(values[index, col]) for col, label in enumerate(labels)}
        active = [label for col, label in enumerate(labels) if values[index, col] >= threshold[col]]
        predictions.append({
            "filename": str(row.filename),
            "start_seconds": int(row.start_seconds),
            "end_seconds": int(row.end_seconds),
            "species": active,
            "probabilities": per_species,
            "max_probability": float(values[index].max(initial=0.0)),
            "no_supported_species_above_threshold": not bool(active),
        })
    return predictions


def audit_local_soundscape_data(
    *,
    annotation_path: str | Path,
    taxonomy_path: str | Path,
    inventory_path: str | Path,
    train_metadata_path: str | Path,
    data_root: str | Path,
) -> dict[str, object]:
    """Return aggregate local-data findings; never reads audio payloads."""
    taxonomy = pd.read_csv(taxonomy_path, dtype=str)
    annotations_raw = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    annotations = validate_soundscape_annotations(annotations_raw, taxonomy)
    inventory = pd.read_csv(inventory_path, dtype={"FileName": str})
    soundscape_inventory = inventory[inventory.FileName.str.startswith("train_soundscapes/")].copy()
    root = Path(data_root)

    def is_present(relative: str) -> bool:
        return any((root / base / relative).is_file() for base in (Path("raw"), Path(".")))

    soundscape_inventory["present"] = soundscape_inventory.FileName.map(is_present)
    annotated_names = set(annotations.filename)
    annotated_inventory = soundscape_inventory[
        soundscape_inventory.FileName.map(lambda path: Path(path).name in annotated_names)
    ]
    class_map = taxonomy.set_index(taxonomy.primary_label.astype(str)).class_name.astype(str)
    label_counts: dict[str, int] = {}
    bird_counts: dict[str, int] = {}
    for labels in annotations.labels:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if class_map[label] == "Aves":
                bird_counts[label] = bird_counts.get(label, 0) + 1
    windows, bird_targets, bird_labels = build_multihot_targets(annotations, taxonomy)

    intervals_by_file = annotations.groupby("filename").size()
    expected_windows_per_soundscape = 12  # Official files are 60 seconds, five-second windows.
    observed_by_file = annotations.groupby("filename").apply(
        lambda frame: set(zip(frame.start_seconds, frame.end_seconds)), include_groups=False
    )
    missing_intervals = sum(
        len({(start, start + 5) for start in range(0, 60, 5)} - observed)
        for observed in observed_by_file
    )
    train_meta = pd.read_csv(train_metadata_path, usecols=["filename", "primary_label"], dtype=str)
    train_names = set(train_meta.filename.map(lambda value: Path(value).name))
    train_labels = set(train_meta.primary_label.astype(str))
    inventory_train_audio = set(
        inventory.loc[inventory.FileName.str.startswith("train_audio/"), "FileName"].map(
            lambda value: Path(value).name
        )
    )
    site = annotations.filename.str.extract(r"_S(\d+)_", expand=False)
    if site.isna().any():
        raise SoundscapeAnnotationError("Cannot derive source site group from every soundscape filename")

    per_window_bird = bird_targets.sum(axis=1)
    observed_bird_labels = sorted(bird_counts)
    soundscape_sites = pd.DataFrame({"filename": annotations.filename, "site": site}).drop_duplicates()
    return {
        "annotation_raw_rows": int(len(annotations_raw)),
        "annotation_exact_duplicates_removed": int(len(annotations_raw) - len(annotations)),
        "unique_annotated_intervals": int(len(annotations)),
        "annotated_soundscapes": int(annotations.filename.nunique()),
        "annotation_columns": list(annotations_raw.columns),
        "interval_duration_seconds": sorted((annotations.end_seconds - annotations.start_seconds).unique().tolist()),
        "interval_count_distribution_per_soundscape": {
            str(k): int(v) for k, v in intervals_by_file.value_counts().sort_index().items()
        },
        "inferred_missing_five_second_intervals": int(missing_intervals),
        "inventory_train_soundscape_files": int(len(soundscape_inventory)),
        "inventory_train_soundscape_bytes": int(soundscape_inventory.SizeBytes.sum()),
        "inventory_train_soundscape_files_present_locally": int(soundscape_inventory.present.sum()),
        "annotated_audio_files_in_inventory": int(len(annotated_inventory)),
        "annotated_audio_expected_bytes": int(annotated_inventory.SizeBytes.sum()),
        "annotated_audio_files_present_locally": int(annotated_inventory.present.sum()),
        "annotated_filename_collisions_with_train_metadata": int(len(annotated_names & train_names)),
        "annotated_filename_collisions_with_train_audio_inventory": int(len(annotated_names & inventory_train_audio)),
        "source_site_group_count": int(site.nunique()),
        "source_site_soundscape_counts": {
            str(k): int(v) for k, v in soundscape_sites.groupby("site").filename.nunique().sort_index().items()
        },
        "unique_taxonomy_labels": int(len(label_counts)),
        "unmapped_labels": [],
        "label_taxonomy_class_counts": {
            str(key): int(value) for key, value in pd.Series({
                class_name: sum(class_map[label] == class_name for label in label_counts)
                for class_name in sorted(set(class_map.values))
            }).items() if value
        },
        "bird_species_count": int(len(observed_bird_labels)),
        "bird_labels_absent_from_train_metadata": int(len(set(observed_bird_labels) - train_labels)),
        "interim_eight_class_positive_intervals": {
            label: int(bird_counts.get(label, 0)) for label in config.WORKING_LABELS
        },
        "bird_label_occurrences": int(sum(bird_counts.values())),
        "bird_positive_intervals": int(np.count_nonzero(per_window_bird)),
        "bird_only_empty_target_intervals": int(np.count_nonzero(per_window_bird == 0)),
        "bird_labels": [
            {"primary_label": label,
             "common_name": str(taxonomy.loc[taxonomy.primary_label.astype(str).eq(label), "common_name"].iloc[0]),
             "positive_intervals": int(bird_counts.get(label, 0))}
            for label in observed_bird_labels
        ],
        "bird_labels_per_interval_distribution": {
            str(k): int(v) for k, v in pd.Series(per_window_bird).value_counts().sort_index().items()
        },
        "test_soundscape_audio_local": False,
    }
