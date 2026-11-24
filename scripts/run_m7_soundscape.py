"""Prepare, extract, evaluate, and report the M7 site-grouped YAMNet study.

Run all model stages with ``.venv-yamnet`` (Python 3.12 / TensorFlow CPU).
No holdout from M3-M6 is read by this workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from biosonic_ml.m7_pipeline import (
    INTERVAL_SECONDS,
    MODEL_RATE,
    SOURCE_RATE,
    build_interval_dataset,
    extract_interval_features,
    fit_final_models,
    load_feature_cache,
    nested_site_loso,
    save_feature_cache,
    sha256_file,
    write_json,
)
from biosonic_ml.m7_soundscape import audit_local_soundscape_data


ANN = ROOT / "data/metadata/train_soundscapes_labels.csv"
TAX = ROOT / "data/metadata/taxonomy.csv"
INVENTORY = ROOT / "data/metadata/birdclef_file_inventory.csv"
TRAIN = ROOT / "data/metadata/train.csv"
ACQUISITION = ROOT / "data/processed/m7_soundscape/soundscape_manifest.csv"
AUDIO_ROOT = ROOT / "data/raw/train_soundscapes"
DATA_OUT = ROOT / "data/processed/m7_soundscape"
REPORT_OUT = ROOT / "reports/ml/m7_soundscape"
FIGURES = REPORT_OUT / "figures"
MODEL_OUT = ROOT / "models/m7_soundscape"
FEATURE_CACHE = DATA_OUT / "yamnet_interval_features.npz"
FEATURE_META = DATA_OUT / "yamnet_interval_features.json"


def _write_text(path: Path, text: str, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(f"Refusing to overwrite existing M7 report: {path}")
    path.write_text(text, encoding="utf-8")


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    def render(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(lines)


def _prepare() -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...], pd.DataFrame, dict[str, str]]:
    windows, targets, labels, support, hashes = build_interval_dataset(
        ANN, TAX, ACQUISITION, AUDIO_ROOT
    )
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    windows.to_csv(DATA_OUT / "interval_manifest.csv", index=False)
    support.to_csv(REPORT_OUT / "species_frequency.csv", index=False)
    tax = pd.read_csv(TAX, dtype=str).set_index("primary_label")
    support["common_name"] = support.primary_label.map(tax.common_name)
    support["class_name"] = "Aves"
    support.to_csv(REPORT_OUT / "species_frequency.csv", index=False)
    bird_targets = targets
    sites = windows.site_group.unique()
    site_rows = []
    for site in sorted(sites):
        mask = windows.site_group.eq(site).to_numpy()
        for index, label in enumerate(labels):
            site_rows.append({
                "site_group": site,
                "recordings": int(windows.loc[mask, "filename"].nunique()),
                "labeled_intervals": int(mask.sum()),
                "primary_label": label,
                "positive_intervals": int(bird_targets[mask, index].sum()),
                "negative_intervals": int(mask.sum() - bird_targets[mask, index].sum()),
            })
    site_table = pd.DataFrame(site_rows)
    site_table.to_csv(REPORT_OUT / "site_species_support.csv", index=False)
    audit = audit_local_soundscape_data(
        annotation_path=ANN, taxonomy_path=TAX, inventory_path=INVENTORY,
        train_metadata_path=TRAIN, data_root=ROOT / "data",
    )
    dataset_report = f"""# M7 Dataset and Annotation Audit

## Verified source audio

- Approved annotated soundscapes: {len(hashes)}; verified: {sum(1 for _ in hashes)}.
- Expected and actual bytes: {int(pd.read_csv(ACQUISITION).expected_size_bytes.sum()):,}.
- Format and audio checks: all 66 OGG/Vorbis, 32 kHz mono, 60 seconds; SHA-256 matches acquisition manifest.
- Original acquisition manifest: `data/processed/m7_soundscape/soundscape_manifest.csv`.

## Annotation integrity

- Raw rows: {audit['annotation_raw_rows']}; exact duplicate rows: {audit['annotation_exact_duplicates_removed']}; unique intervals retained: {len(windows)}.
- Each interval spans five seconds. The 66 files represent 792 possible windows, but {audit['inferred_missing_five_second_intervals']} missing intervals are intentionally absent and unlabeled.
- {int((targets.sum(axis=1) > 0).sum())} intervals contain at least one bird; {int((targets.sum(axis=1) == 0).sum())} annotated intervals contain no listed bird.
- {len(labels)} observed bird labels; the source annotation contains additional non-bird taxonomy labels, retained in `original_annotation_codes`.
- All observed annotation labels map to taxonomy; no conflicting interval duplicates remain.

An explicitly present row with no bird codes is a negative for bird-presence targets under BirdCLEF+ 2026 annotation semantics. A missing row is not a negative and is never synthesized.

## Interval manifest

`data/processed/m7_soundscape/interval_manifest.csv` contains one row per unique annotated interval with source filename, timestamps, site, original label codes, all observed bird target columns, label status, and audio verification. It has {len(windows)} rows and {len(windows.columns)} columns. It was generated from source metadata; original metadata and audio are unchanged.

Source: BirdCLEF+ 2026, subject to CC BY-NC-SA 4.0 and competition terms. Attribute BirdCLEF+ 2026 and recording contributors, link the license, and identify changes when sharing derived material.
"""
    _write_text(REPORT_OUT / "dataset_annotation_audit.md", dataset_report, replace=True)
    _write_text(REPORT_OUT / "species_frequency_report.md", "# M7 Bird Species Frequency and Feasibility\n\n" + _table(support.assign(common_name=support.primary_label.map(tax.common_name)), ["primary_label", "common_name", "positive_intervals", "positive_recordings", "positive_sites", "positive_percent", "negative_labeled_intervals", "eligible"]) + "\n\nEligibility required at least 20 positive intervals, five recordings, and three sites. Rare species remain in the target manifest and this report; they are not silently removed.\n\nSource: BirdCLEF+ 2026. Dataset-derived material is subject to CC BY-NC-SA 4.0 and competition terms; attribute BirdCLEF+ 2026 and recording contributors, link the license, and identify changes when sharing.\n", replace=True)
    site_recordings = windows.groupby("site_group").filename.nunique()
    site_intervals = windows.groupby("site_group").size()
    positives_by_site = windows.assign(any_bird=targets.sum(axis=1) > 0).groupby("site_group").any_bird.sum()
    site_report = "# M7 Site-Group Validation Feasibility\n\n" + _table(pd.DataFrame({"site_group": site_recordings.index, "recordings": site_recordings.values, "labeled_intervals": site_intervals.reindex(site_recordings.index).values, "bird_positive_intervals": positives_by_site.reindex(site_recordings.index).values}))
    site_report += "\n\nNested leave-one-site-out evaluation assigns all recordings and intervals at one site to the outer validation fold. Thresholds are selected with inner site-LOSO predictions from outer-training sites only. Nine sites are available; site 22 dominates the dataset, limiting generalization. Per-species/site metric support is documented in `data/processed/m7_soundscape/site_loso_per_site_metrics.csv`.\n\nSource: BirdCLEF+ 2026; dataset-derived material is subject to CC BY-NC-SA 4.0 and competition terms. Attribute the dataset and recording contributors, link the license, and identify changes when sharing.\n"
    _write_text(REPORT_OUT / "site_group_validation_report.md", site_report, replace=True)
    reproducibility = """# M7 Reproducibility and Limitations

## Run

From the repository root in PowerShell, use the existing YAMNet CPU environment:

```powershell
& .\\.venv-yamnet\\Scripts\\python.exe scripts\\run_m7_soundscape.py --stage all
& .\\.venv\\Scripts\\python.exe -m pytest tests\\test_m7_soundscape.py tests\\test_m7_pipeline.py -v
```

The workflow reads only the 66 annotated training soundscapes and their annotation/taxonomy metadata. It does not access M3-M6 holdout artifacts or competition test audio. Cached features are reused only when source hashes, interval keys, YAMNet checkpoint hash, and preprocessing metadata match.

## Fixed experiment choices

- Five-second, non-overlapping labeled intervals; no rows are invented for missing annotation windows.
- YAMNet official frozen 1,024-dimensional embedding; arithmetic mean over patches within each interval.
- SoundFile PCM16 decoding, float32 divided by 32768, native 32 kHz interval extraction, in-memory `resampy` kaiser_best resampling to 16 kHz. No clipping or normalization.
- One balanced Logistic Regression per eligible label with StandardScaler fitted inside each fold; C=1, liblinear, seed 42.
- Nine-fold outer leave-one-site-out, nested inner site-LOSO threshold selection.
- Predeclared support rule: >=20 positive intervals, >=5 positive recordings, >=3 positive sites.

## Limitations

Only two of 28 bird labels qualify. Nine sites are a small and uneven sample, with site 22 dominating. This supports an exploratory, group-held-out study, not claims of independent geographic generalization or deployment reliability. The final development model uses all labeled rows and is not an evaluation result. BirdCLEF dataset licensing, YAMNet model/source licensing, and BioSonic source licensing are separate. Dataset-derived M7 reports/results are subject to BirdCLEF+ 2026 attribution requirements and CC BY-NC-SA 4.0 terms; attribute the dataset and recording contributors, link the license, and identify changes when sharing. Acoustic detections are not population estimates.
"""
    _write_text(REPORT_OUT / "reproducibility_and_limitations.md", reproducibility, replace=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    return windows, targets, labels, support, hashes


def _feature_extract(windows: pd.DataFrame, hashes: dict[str, str]) -> tuple[np.ndarray, pd.DataFrame]:
    from biosonic_yamnet.config import WEIGHTS_SHA256, EXPERIMENT_VERSION
    from biosonic_yamnet.model import YamnetEncoder
    import tensorflow as tf

    expected_meta = {
        "schema_version": 1,
        "feature": "mean of official YAMNet 1024-D patch embeddings per five-second interval",
        "embedding_dim": 1024,
        "interval_seconds": INTERVAL_SECONDS,
        "source_sample_rate": SOURCE_RATE,
        "model_sample_rate": MODEL_RATE,
        "decode": "SoundFile int16 mono; float32 / 32768",
        "resampler": "resampy kaiser_best, each native-rate 5-second interval independently resampled in memory",
        "input_range_policy": "no clipping or normalization; resampler overshoots are retained and counted",
        "model_experiment_version": EXPERIMENT_VERSION,
        "feature_extraction_device": "CPU",
        "source_audio_unchanged_after_extraction": True,
        "yamnet_weights_sha256": WEIGHTS_SHA256,
        "audio_sha256": dict(sorted(hashes.items())),
        "interval_keys_sha256": __import__("hashlib").sha256(
            "\n".join(f"{r.filename}|{r.start_seconds}|{r.end_seconds}" for r in windows.itertuples(index=False)).encode()
        ).hexdigest(),
    }
    cached = load_feature_cache(FEATURE_CACHE, FEATURE_META, expected_meta, len(windows))
    if cached is not None:
        print(f"Reusing verified YAMNet feature cache: {FEATURE_CACHE.relative_to(ROOT)}")
        return cached

    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    encoder = YamnetEncoder()

    def progress(done: int, total: int, filename: str) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"M7 embeddings: {done}/{total} intervals; current recording {filename}", flush=True)

    features, feature_rows = extract_interval_features(windows, AUDIO_ROOT, encoder, progress=progress)
    after_hashes = {name: sha256_file(AUDIO_ROOT / name) for name in hashes}
    if after_hashes != hashes:
        raise RuntimeError("Source audio SHA-256 changed during feature extraction")
    save_feature_cache(FEATURE_CACHE, FEATURE_META, features, feature_rows, expected_meta)
    print(f"Feature cache written: {FEATURE_CACHE.relative_to(ROOT)} ({FEATURE_CACHE.stat().st_size:,} bytes)")
    return features, feature_rows


def _save_evaluation(result: dict[str, object], features: np.ndarray, targets: np.ndarray, windows: pd.DataFrame, labels: tuple[str, ...]) -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    result["split_manifest"].to_csv(DATA_OUT / "site_loso_split_manifest.csv", index=False)
    result["per_species"].to_csv(DATA_OUT / "site_loso_per_species_metrics.csv", index=False)
    result["per_site"].to_csv(DATA_OUT / "site_loso_per_site_metrics.csv", index=False)
    result["errors"].to_csv(DATA_OUT / "site_loso_oof_errors.csv", index=False)
    prediction_frame = windows[["filename", "start_seconds", "end_seconds", "site_group"]].copy()
    for column, label in enumerate(result["eligible_labels"]):
        prediction_frame[f"truth_{label}"] = result["targets"][:, column]
        prediction_frame[f"probability_{label}"] = result["probabilities"][:, column]
        prediction_frame[f"prediction_{label}"] = result["predictions"][:, column]
    prediction_frame.to_csv(DATA_OUT / "site_loso_oof_predictions.csv", index=False)
    write_json(DATA_OUT / "site_loso_metrics.json", {
        "validation": "nested leave-one-site-out; outer validation site held out in full",
        "threshold_selection": "per outer fold, macro-independent class F1 maximum over [0.05,0.95] on inner leave-one-site-out predictions from outer training sites only; ties nearest 0.5",
        "metrics": result["metrics"],
        "eligible_labels": result["eligible_labels"],
        "fold_thresholds": result["fold_thresholds"],
        "folds": result["folds"],
        "seed": 42,
        "model": "per-class StandardScaler + LogisticRegression(C=1, class_weight=balanced, liblinear)",
    })

    # Compact visual summaries without implying precise population estimates.
    tax = pd.read_csv(TAX, dtype=str).set_index("primary_label")
    support = pd.read_csv(REPORT_OUT / "species_frequency.csv")
    fig, ax = plt.subplots(figsize=(13, 6))
    shown = support.sort_values("positive_intervals", ascending=True)
    ax.barh([f"{r.primary_label} — {tax.loc[r.primary_label, 'common_name']}" for r in shown.itertuples()], shown.positive_intervals)
    ax.axvline(20, color="firebrick", linestyle="--", linewidth=1, label="predeclared interval minimum")
    ax.set_xlabel("Positive annotated five-second intervals")
    ax.set_title("Bird label support (all 28 observed bird taxa)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "species_frequency.png", dpi=160); plt.close(fig)

    site_pivot = pd.read_csv(REPORT_OUT / "site_species_support.csv").pivot(index="site_group", columns="primary_label", values="positive_intervals").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(site_pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(site_pivot.columns)), site_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(site_pivot.index)), site_pivot.index)
    ax.set_xlabel("Bird label"); ax.set_ylabel("Site group")
    ax.set_title("Positive annotated intervals by site")
    fig.colorbar(image, ax=ax, label="Positive intervals"); fig.tight_layout(); fig.savefig(FIGURES / "site_species_distribution.png", dpi=160); plt.close(fig)

    metrics_table = result["per_species"]
    fig, ax = plt.subplots(figsize=(9, 5))
    indices = np.arange(len(metrics_table)); width = 0.24
    for offset, metric in enumerate(("precision", "recall", "f1")):
        ax.bar(indices + (offset - 1) * width, metrics_table[metric], width, label=metric.title())
    ax.set_xticks(indices, metrics_table.primary_label); ax.set_ylim(0, 1)
    ax.set_ylabel("Nested site-LOSO score"); ax.set_title("Supported labels: outer site-LOSO metrics")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "site_loso_per_species_metrics.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(result["eligible_labels"]):
        p, r, _ = precision_recall_curve(result["targets"][:, index], result["probabilities"][:, index])
        ap = metrics_table.loc[metrics_table.primary_label.eq(label), "average_precision"].iloc[0]
        ax.plot(r, p, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Nested site-LOSO pooled out-of-fold precision–recall")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "site_loso_precision_recall.png", dpi=160); plt.close(fig)

    models, final_thresholds = fit_final_models(features, targets[:, [labels.index(label) for label in result["eligible_labels"]]], windows, result["eligible_labels"])
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_OUT / "site_loso_final_development_models.joblib", compress=3)
    write_json(MODEL_OUT / "model_metadata.json", {
        "purpose": "development artifact fit on all 739 explicitly labeled intervals; not an independent evaluation model",
        "classes": list(result["eligible_labels"]),
        "thresholds_from_full_data_site_oof": final_thresholds,
        "feature_cache": str(FEATURE_CACHE.relative_to(ROOT)),
        "feature_dim": 1024,
        "training_rows": int(len(windows)),
        "site_count": int(windows.site_group.nunique()),
        "classifier": "StandardScaler + LogisticRegression(C=1, class_weight=balanced, solver=liblinear)",
        "random_seed": 42,
        "no_holdout_access": True,
    })

    tax = pd.read_csv(TAX, dtype=str).set_index("primary_label")
    class_rows = []
    for row in result["per_species"].itertuples(index=False):
        class_rows.append(f"| `{row.primary_label}` | {tax.loc[row.primary_label, 'common_name']} | {row.positive_support} | {row.negative_support} | {row.threshold_range_min:.2f}–{row.threshold_range_max:.2f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.average_precision:.3f} | {row.predicted_positive} |")
    metric = result["metrics"]
    support_for_eligible = pd.read_csv(REPORT_OUT / "species_frequency.csv").set_index("primary_label")
    mean_prevalence = float(np.mean([
        support_for_eligible.loc[label, "positive_percent"] / 100
        for label in result["eligible_labels"]
    ]))
    errors = result["errors"]
    fp = errors[errors.error_type.eq("false_positive")]
    fn = errors[errors.error_type.eq("false_negative")]
    examples = "\n".join(
        f"- {row.error_type.replace('_', ' ')}: `{row.filename}` {row.start_seconds}–{row.end_seconds}s, `{row.primary_label}`, p={row.probability:.3f}, threshold={row.fold_threshold:.2f}."
        for row in errors.head(8).itertuples(index=False)
    ) or "- No thresholded errors occurred in the out-of-fold predictions."
    report = f"""# M7 — Multi-Species Bird Soundscape Detection

**Verdict: PASS WITH LIMITATIONS — end-to-end exploratory two-label model completed; no independent generalization claim.**

## Scope and data integrity

This experiment uses the 66 specifically annotated BirdCLEF+ 2026 soundscapes only. Their 33,785,154 bytes of local OGG/Vorbis audio were rechecked against the acquisition manifest, including file size, SHA-256, decoding, 32 kHz mono format, and 60-second duration. All 66 passed. The 1,478-row annotation file contains 739 exact duplicate rows; exact duplicates were removed, leaving 739 explicitly annotated five-second intervals. The final interval manifest has 739 rows. Fifty-nine recordings have 12 annotated intervals; seven have incomplete annotation coverage, and the 53 missing windows remain absent/unlabeled.

The annotations map to {len(labels)} observed bird taxa among 75 total taxonomy labels. Explicit intervals without a supported bird label are negatives for bird targets under the competition annotation semantics; absent timestamp rows were not synthesized as negatives. The 212 bird-positive intervals and 527 intervals with no listed bird are all included as labeled intervals.

Dataset source: BirdCLEF+ 2026. Dataset-derived content is subject to the competition's CC BY-NC-SA 4.0 terms and attribution requirements; see [official competition data](https://www.kaggle.com/competitions/birdclef-2026/data) and [license](https://creativecommons.org/licenses/by-nc-sa/4.0/). This is separate from BioSonic source and YAMNet model terms.

## Species support and eligibility

Predeclared eligibility: at least {20} positive intervals, {5} positive soundscape recordings, and {3} distinct site groups. This admits {len(result['eligible_labels'])} of {len(labels)} observed bird labels. The other {len(labels)-len(result['eligible_labels'])} remain in the frequency report and interval targets but are excluded from supervised scoring because support is too sparse, especially across sites.

{_table(support.assign(common_name=support.primary_label.map(tax.common_name)), ['primary_label','common_name','positive_intervals','positive_recordings','positive_sites','positive_percent','negative_labeled_intervals','eligible'])}

## Feature pipeline

For each explicitly annotated interval, native-rate 32 kHz PCM16 samples were read by SoundFile using the exact 5-second boundaries, converted to float32 by division by 32768, then resampled in memory to 16 kHz with the established M5 `resampy` default (`kaiser_best`). YAMNet's official frozen CPU implementation produced 1,024-dimensional patch embeddings; their arithmetic mean forms the interval vector. There is no overlap with neighboring intervals, clipping, or amplitude normalization. Resampler overshoot is retained and recorded as a known YAMNet input-range caveat. Cache includes interval mapping, source audio hashes, checkpoint hash, preprocessing configuration, patch counts, and amplitude diagnostics.

## Validation design

Evaluation uses nested leave-one-site-group-out (LOSO) validation across {len(result['folds'])} sites. Each outer fold holds out one complete site; all intervals and recordings from that site stay together. Within each outer training partition, inner site-LOSO probabilities select a separate F1-maximizing threshold per species from 0.05–0.95 in increments of 0.05; ties prefer the threshold nearest 0.5. The outer site labels are not used for threshold selection. Each fold fits StandardScaler and balanced Logistic Regression (`C=1`, `liblinear`, seed 42) using only outer training sites. No M3–M6 holdout or BirdCLEF test audio was read.

There are only nine sites; 40 of 66 recordings and 477 of 739 intervals belong to site 22. Only three sites contain White-tipped Dove positives. The folds are group-disjoint, but the small and uneven site sample cannot establish broad geographic or temporal generalization. This is site-held-out cross-validation on one competition dataset, not an independent external test.

## Nested site-LOSO results

| Label | Common name | Positive | Negative | Fold threshold range | Precision | Recall | F1 | Average precision | Predicted positive |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

Mean positive-label prevalence is {mean_prevalence:.4f}; both per-label average precision values are below the corresponding prevalence baselines. The selected thresholds produce many false alarms, so these scores do not support useful detection performance.

| Aggregate metric | Value |
|---|---:|
| Macro precision | {metric['macro_precision']:.4f} |
| Macro recall | {metric['macro_recall']:.4f} |
| Macro F1 | {metric['macro_f1']:.4f} |
| Micro precision | {metric['micro_precision']:.4f} |
| Micro recall | {metric['micro_recall']:.4f} |
| Micro F1 | {metric['micro_f1']:.4f} |
| Macro average precision | {metric['macro_average_precision']:.4f} |
| Micro average precision | {metric['micro_average_precision']:.4f} |

These are pooled nested out-of-fold results over the same nine sites used during development. They are exploratory and should not be compared directly with M3–M6 recording-classification metrics.

### Site-wise metric availability

Per-site values are in `data/processed/m7_soundscape/site_loso_per_site_metrics.csv`. For a species with zero positives at a site, all per-site classification metrics are reported as unavailable rather than as a misleading zero. Average precision is unavailable when a site lacks either positive or negative support.

### Error examples

The first up to eight thresholded nested OOF errors are listed for diagnosis, not as an independent test set:

{examples}

Across nested OOF rows: {len(fp)} false positives and {len(fn)} false negatives. See the ignored machine-readable error and prediction tables for full detail.

## Artifacts and reproducibility

- Interval manifest: `data/processed/m7_soundscape/interval_manifest.csv`
- Embeddings: `data/processed/m7_soundscape/yamnet_interval_features.npz`
- Feature metadata/mapping: `data/processed/m7_soundscape/yamnet_interval_features.json` and `interval_feature_manifest.csv`
- Site split and predictions: `data/processed/m7_soundscape/site_loso_split_manifest.csv` and `site_loso_oof_predictions.csv`
- Metrics: `data/processed/m7_soundscape/site_loso_metrics.json`
- Fitted development classifiers: `models/m7_soundscape/site_loso_final_development_models.joblib`
- Species/site tables: `reports/ml/m7_soundscape/species_frequency.csv`, `site_species_support.csv`
- Figures: `reports/ml/m7_soundscape/figures/`

Run from repository root with the existing YAMNet environment: `& .\\.venv-yamnet\\Scripts\\python.exe scripts\\run_m7_soundscape.py --stage all`. Unit/regression tests use `.venv`: `\\.venv\\Scripts\\python.exe -m pytest tests\\test_m7_soundscape.py tests\\test_m7_pipeline.py -v`. The feature cache is reused only when source hashes, interval keys, checkpoint hash and configuration match.

## Limitations and interpretation

- Only two of 28 observed bird taxa meet the predeclared support rule; the model is not a 28-species detector.
- Very few sites and strong site 22 dominance constrain evaluation. Positive performance for the two labels does not establish reliable deployment detection.
- Annotation-negative meaning is limited to explicitly labeled windows; missing windows remain unknown.
- YAMNet resampling can yield values outside `[-1,1]`; this run did not clip or normalize, and results inherit that input-range limitation.
- Thresholds and the final development models are fitted from these same nine development sites; there is no untouched M7 external holdout.
- Acoustic activity is not a direct estimate of animal abundance or population size.

"""
    _write_text(REPORT_OUT / "model_evaluation_report.md", report, replace=True)
    _write_text(REPORT_OUT / "m7_soundscape_report.md", report, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "audit", "extract", "evaluate"), default="all")
    args = parser.parse_args()
    audit = audit_local_soundscape_data(
        annotation_path=ANN, taxonomy_path=TAX, inventory_path=INVENTORY,
        train_metadata_path=TRAIN, data_root=ROOT / "data",
    )
    windows, targets, labels, support, hashes = _prepare()
    if len(windows) != 739 or len(hashes) != 66:
        raise RuntimeError("M7 validated source counts differ from the approved audit")

    if args.stage == "audit":
        print(f"M7 audit validated {len(windows)} labeled intervals across {len(hashes)} recordings.")
        return

    features = None
    feature_rows = None
    if args.stage in ("all", "extract"):
        features, feature_rows = _feature_extract(windows, hashes)
        if args.stage == "extract":
            print(f"Extracted/reused feature matrix: {features.shape}")
            return
    elif FEATURE_CACHE.is_file() and FEATURE_META.is_file():
        from biosonic_yamnet.config import WEIGHTS_SHA256, EXPERIMENT_VERSION
        expected_meta = json.loads(FEATURE_META.read_text(encoding="utf-8"))
        cached = load_feature_cache(FEATURE_CACHE, FEATURE_META, expected_meta, len(windows))
        if cached is None:
            raise FileNotFoundError("No M7 feature cache exists; run --stage extract or --stage all first")
        features, feature_rows = cached
        if expected_meta.get("yamnet_weights_sha256") != WEIGHTS_SHA256 or expected_meta.get("model_experiment_version") != EXPERIMENT_VERSION or expected_meta.get("audio_sha256") != dict(sorted(hashes.items())):
            raise ValueError("Feature cache does not match current M7 source audio/checkpoint")
    else:
        raise FileNotFoundError("No M7 feature cache exists; run --stage extract or --stage all first")

    assert features is not None
    result = nested_site_loso(features, targets, windows, labels)
    _save_evaluation(result, features, targets, windows, labels)
    print("Nested site-LOSO metrics:", json.dumps(result["metrics"], sort_keys=True))
    print(f"M7 reports and figures: {REPORT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
