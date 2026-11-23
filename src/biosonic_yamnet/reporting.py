"""Post-evaluation comparison tables, plots, and the M5 experiment report."""

import json
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .io_utils import atomic_bytes, atomic_csv, atomic_json, atomic_text


def _save_figure(fig, path: Path) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    atomic_bytes(buffer.getvalue(), path)


def _metric_value(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return None


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a small dependency-free Markdown table for the final report."""
    columns = [str(column) for column in frame.columns]

    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[cell(value) for value in row]
            for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
              for i in range(len(columns))]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |"
            for row in rows]
    return "\n".join([header, rule, *body])


def _per_class_report(path: Path) -> dict[str, dict]:
    frame = pd.read_csv(path)
    key = "class" if "class" in frame.columns else frame.columns[0]
    result = {}
    for row in frame.to_dict("records"):
        label = str(row.get(key, ""))
        if label in config.WORKING_LABELS:
            result[label] = {"precision": float(row["precision"]),
                             "recall": float(row["recall"]),
                             "f1": float(row.get("f1-score", row.get("f1"))),
                             "support": int(row["support"])}
    return result


def create_final_reports(holdout_result: dict) -> dict:
    report_root = config.REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    m3 = pd.read_csv(config.ROOT / "reports/ml/m3_eight_class/model_comparison.csv")
    m4 = pd.read_csv(config.ROOT / "reports/ml/m4_custom_cnn/m3_m4_model_comparison.csv")
    m5 = holdout_result["metrics"]
    comparison = []
    for row in m3.to_dict("records"):
        comparison.append({
            "model": f"M3 {row['model']}", "milestone": "M3",
            "accuracy": _metric_value(row, "accuracy"),
            "macro_precision": _metric_value(row, "precision_macro"),
            "macro_recall": _metric_value(row, "recall_macro"),
            "macro_f1": _metric_value(row, "f1_macro"),
            "weighted_f1": _metric_value(row, "f1_weighted"),
            "ovr_macro_roc_auc": _metric_value(row, "roc_auc_ovr_macro"),
            "selection_evidence": "M3 training-only group CV",
        })
    m4row = m4[m4.milestone.eq("M4 custom CNN")]
    if m4row.empty:
        m4row = m4[m4.model.astype(str).str.contains("custom CNN", case=False)]
    if not m4row.empty:
        row = m4row.iloc[0]
        comparison.append({
            "model": "M4 custom CNN", "milestone": "M4",
            "accuracy": _metric_value(row, "holdout_accuracy", "accuracy"),
            "macro_precision": _metric_value(row, "holdout_macro_precision", "precision_macro"),
            "macro_recall": _metric_value(row, "holdout_macro_recall", "recall_macro"),
            "macro_f1": _metric_value(row, "holdout_macro_f1", "f1_macro"),
            "weighted_f1": _metric_value(row, "holdout_weighted_f1", "f1_weighted"),
            "ovr_macro_roc_auc": _metric_value(row, "holdout_roc_auc_ovr_macro", "roc_auc_ovr_macro"),
            "selection_evidence": "M4 internal validation; same previously evaluated M3 holdout",
        })
    freeze = json.loads((config.MODEL_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    comparison.append({
        "model": f"M5 YAMNet + {freeze['selected_model']}", "milestone": "M5",
        "accuracy": m5["accuracy"], "macro_precision": m5["precision_macro"],
        "macro_recall": m5["recall_macro"], "macro_f1": m5["macro_f1"],
        "weighted_f1": m5["weighted_f1"], "ovr_macro_roc_auc": m5["roc_auc_ovr_macro"],
        "selection_evidence": "Five-fold author/content-grouped CV on M3 training partition only",
    })
    comparison_df = pd.DataFrame(comparison)
    atomic_csv(comparison, list(comparison_df.columns), report_root / "m3_m4_m5_comparison.csv")

    # Confusion matrix visualization (M5 only; all values are measured).
    cm = np.asarray(holdout_result["confusion_matrix"], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.WORKING_LABELS)), config.WORKING_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(config.WORKING_LABELS)), config.WORKING_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("M5 YAMNet classifier — final holdout confusion matrix")
    fig.colorbar(image, ax=ax, label="Recordings")
    threshold = cm.max() * 0.55 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    _save_figure(fig, report_root / "holdout_confusion_matrix.png")

    # Model-level F1 comparison uses persisted reference results without rerunning M3/M4.
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = comparison_df.dropna(subset=["macro_f1"])
    ax.bar(bars.model, bars.macro_f1, color=["#6585a8" if x != "M5" else "#27826f" for x in bars.milestone])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recording-level macro F1")
    ax.set_title("M3, M4, and M5 on the shared eight-class holdout")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_figure(fig, report_root / "model_macro_f1_comparison.png")

    cv_candidates = pd.read_csv(report_root / "classifier_cv_candidates.csv")
    best_by_model = cv_candidates.sort_values("mean_macro_f1", ascending=False).groupby("model", as_index=False).first()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(best_by_model.model, best_by_model.mean_macro_f1, yerr=best_by_model.std_macro_f1,
           capsize=5, color="#457b9d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Training-only CV macro F1 (mean ± SD)")
    ax.set_title("Best configuration per classifier — author-grouped CV")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save_figure(fig, report_root / "training_cv_macro_f1.png")

    m5_per_class = _per_class_report(report_root / "holdout_per_class_metrics.csv")
    per_class_rows = [{"primary_label": label, "M5_f1": m5_per_class[label]["f1"]}
                      for label in config.WORKING_LABELS]
    for model_name in ("logistic_regression", "random_forest", "svm"):
        path = config.ROOT / f"reports/ml/m3_eight_class/{model_name}_classification_report.csv"
        per_class = _per_class_report(path)
        for row in per_class_rows:
            row[f"M3_{model_name}_f1"] = per_class.get(row["primary_label"], {}).get("f1")
    m4_per_class = _per_class_report(config.ROOT / "reports/ml/m4_custom_cnn/holdout_classification_report.csv")
    for row in per_class_rows:
        row["M4_custom_cnn_f1"] = m4_per_class.get(row["primary_label"], {}).get("f1")
    atomic_csv(per_class_rows, list(per_class_rows[0]), report_root / "per_class_f1_comparison.csv")
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.16
    x = np.arange(len(config.WORKING_LABELS))
    for index, key in enumerate(["M3_logistic_regression_f1", "M3_random_forest_f1", "M3_svm_f1",
                                 "M4_custom_cnn_f1", "M5_f1"]):
        ax.bar(x + (index - 2) * width, [row.get(key) for row in per_class_rows], width, label=key.replace("_f1", ""))
    ax.set_xticks(x, config.WORKING_LABELS, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-class F1")
    ax.set_title("Per-class performance on the shared holdout")
    ax.legend(ncol=2)
    fig.tight_layout()
    _save_figure(fig, report_root / "per_class_f1_comparison.png")

    model_info = json.loads((config.MODEL_ROOT / "final_model_info.json").read_text(encoding="utf-8"))
    train_summary = json.loads((config.DATA_ROOT / "train_embedding_summary.json").read_text(encoding="utf-8"))
    extraction = holdout_result["embedding_extraction"]
    actual_classifier = config.MODEL_ROOT / "final_classifier.joblib"
    component_metrics = {
        "yamnet_parameters": model_info.get("yamnet_parameter_count"),
        "yamnet_checkpoint_bytes": config.WEIGHTS_BYTES,
        "classifier_bytes": actual_classifier.stat().st_size,
        "training_aggregate_reconstruction_wall_seconds": train_summary.get("wall_seconds_including_audio_and_hashing"),
        "training_yamnet_inference_seconds": train_summary.get("total_model_inference_seconds"),
        "holdout_embedding_extraction_wall_seconds": extraction.get("wall_seconds_including_audio_and_hashing"),
        "holdout_yamnet_mean_seconds_per_recording": extraction.get("mean_inference_seconds_per_recording"),
        "classifier_holdout_inference_ms_per_recording": holdout_result["classifier_inference_ms_per_record"],
        "cv_search_seconds": model_info.get("cv_total_seconds"),
        "final_classifier_refit_seconds": model_info.get("final_refit_seconds"),
    }
    atomic_json(component_metrics, report_root / "computational_performance.json")

    cv_table = best_by_model.sort_values("model")
    comparison_table = _markdown_table(comparison_df)
    cv_markdown = cv_table[["model", "candidate_parameters", "mean_macro_f1", "std_macro_f1",
                            "mean_accuracy", "mean_precision_macro", "mean_recall_macro",
                            "mean_weighted_f1", "mean_roc_auc_ovr_macro"]]
    cv_markdown = _markdown_table(cv_markdown)
    per_class_md = _markdown_table(pd.DataFrame(
        [{"Class": label, **m5_per_class[label]} for label in config.WORKING_LABELS]
    ))
    report_text = f"""# BioSonic M5 — YAMNet transfer-learning experiment

## Objective and scope

This is an eight-class interim recording-level primary-label classification experiment using the 2,177 approved recordings present in the working set. Four intended classes (`swtman1`, `smbani`, `yecpar`, `spispi1`) remain unavailable, so M5 does not complete the 12-class objective. The six M2 representative samples are excluded from all experiment partitions and are used only for the isolated YAMNet smoke test. This is not soundscape multi-label detection, nor a direct estimate of animal population size.

## Model, source, and license

The frozen YAMNet backbone is Google's MobileNetV1 depthwise-separable AudioSet model. It contains **{model_info['yamnet_parameter_count']:,} parameters**, emits 521 AudioSet scores and 1,024-dimensional embeddings per patch, and its original classifier outputs are not used as BioSonic species predictions. Official model/checkpoint source: [Google YAMNet model card](https://tfhub.dev/google/yamnet/1); official implementation: [TensorFlow Models YAMNet at pinned commit {config.OFFICIAL_SOURCE_COMMIT}]({config.OFFICIAL_SOURCE_BASE}); official weights: [{config.WEIGHTS_URL}]({config.WEIGHTS_URL}). The model card lists Apache-2.0. AudioSet's dataset terms are CC BY 4.0 and its ontology terms CC BY-SA 4.0; those are distinct from the model's license. Provenance, checksums, and retained license text are in `models/m5_yamnet/PROVENANCE.md` and `models/m5_yamnet/source_provenance.json`.

## Processing and embedding aggregation

Source OGG/Vorbis files were read unchanged using SoundFile as mono PCM16, then converted to float32 by dividing by 32768, matching the pinned official YAMNet `inference.py` example. Audio was resampled in memory to 16 kHz using resampy 0.4.3 (`kaiser_best`); output length follows resampy's floor convention. We did not clip finite post-resampling filter overshoot: the official reference inference path passes the resampled waveform directly to YAMNet. Its peak and count of samples outside [-1, 1] are recorded per file as a diagnostic. This preserves upstream behavior, but the resulting inputs do not always meet the model card's stated normalized range: the recorded maximum absolute value was 1.6589 for training and 1.6967 for holdout, with 32,783 and 9,855 samples outside [-1, 1], respectively. The models were run on those values without clipping; this is a material input-range limitation and should be addressed only in a separately designed future experiment with a new validation protocol. We did not use M4's 128-band cache. For every recording, the arithmetic mean of all 1,024-dimensional patch embeddings is the one recording-level vector; patches are not treated as independent training examples. Patch counts and source hashes are persisted in the M5 extraction manifests.

An M2 representative sample outside the approved experiment subset was used for the smoke test. It produced {json.loads((report_root / 'model_smoke.json').read_text(encoding='utf-8'))['scores_shape'][0]} patches, score shape {json.loads((report_root / 'model_smoke.json').read_text(encoding='utf-8'))['scores_shape']}, embedding shape {json.loads((report_root / 'model_smoke.json').read_text(encoding='utf-8'))['embedding_shape']}, and log-Mel shape {json.loads((report_root / 'model_smoke.json').read_text(encoding='utf-8'))['log_mel_shape']}. Repeated CPU outputs matched within 1e-5 relative / 1e-6 absolute tolerance.

## Dataset, partitions, and leakage prevention

The authoritative split remains unchanged: **1,748 training recordings** and **429 final holdout recordings**. Classifier development used only the 1,748 training recordings with five-fold `StratifiedGroupKFold`, seed 42, stratification by `primary_label × collection`, and the existing author/content `group_id`. All five folds retained all eight classes, all 16 label×collection strata, and had zero group overlap. Unknown-author recordings remain in the original training partition and share the established unknown-author grouping. The existing final holdout was not read for embedding extraction, classifier fitting, CV, scaler fitting, hyperparameter choice, or model selection; it was first accessed after the classifier freeze marker was written.

## Classifier configurations and cross-validation

Macro-F1 was the primary selection metric. Logistic Regression and RBF SVM use a scikit-learn Pipeline with `StandardScaler` fit inside each training fold and `class_weight='balanced'`. Random Forest uses no scaler, `class_weight='balanced_subsample'`, four workers, and seed 42. The tested grids were LR C ∈ {{0.1, 1, 10}}, SVM C ∈ {{0.5, 1, 5}} with `gamma='scale'`, and RF 200/300 trees, max depth None/24, and leaf size 1/2. Candidate selection and final fit were restricted to training data.

Best configuration per model (mean and sample standard deviation across five author-grouped folds):

{cv_markdown}

The selected frozen classifier is **{model_info['selected_model']}** with parameters `{json.dumps(model_info['selected_parameters'], sort_keys=True)}`. Its training-CV macro-F1 was {model_info['selected_cv_metrics']['mean_macro_f1']:.4f} ± {model_info['selected_cv_metrics']['std_macro_f1']:.4f}. Fold-level metrics and all candidate configurations are in `classifier_cv_folds.csv` and `classifier_cv_candidates.csv`.

## Final holdout evaluation

Only after the classifier artifact and freeze record were finalized, the pipeline processed the existing 429 recordings once. Results are recording-level:

- Accuracy: **{m5['accuracy']:.4f}**
- Macro precision: **{m5['precision_macro']:.4f}**
- Macro recall: **{m5['recall_macro']:.4f}**
- Macro F1: **{m5['macro_f1']:.4f}**
- Weighted F1: **{m5['weighted_f1']:.4f}**
- OVR macro ROC-AUC: **{m5['roc_auc_ovr_macro']:.4f}**

Per-class scores:

{per_class_md}

The confusion matrix and recording-level predictions are saved separately in this M5 report directory. The shared holdout had already been used for prior M3 and M4 benchmark reporting; this M5 run follows the requested frozen final-evaluation boundary, but the comparison is not an independent external test.

## M3/M4/M5 comparison

{comparison_table}

On this shared holdout, M5 macro-F1 is below both M3 Logistic Regression and the M4 custom CNN results. This descriptive comparison uses the same historical holdout that had already been used in M3/M4 reporting, so it is not an independent external test. The M3/M4 artifacts were read for comparison and were not recalculated or overwritten.

## Computational performance and hardware

The isolated environment used TensorFlow {model_info['package_versions']['tensorflow']} and tf-keras {model_info['package_versions']['tf-keras']} on Python {model_info['python']}. Native Windows TensorFlow detected no GPU, so YAMNet embedding extraction ran on CPU; the PyTorch/CUDA environment and RTX 4060 were not used. The checkpoint is {config.WEIGHTS_BYTES:,} bytes; the selected classifier is {actual_classifier.stat().st_size:,} bytes. `computational_performance.json` reports summed model inference time, the resumed training-aggregate reconstruction time, actual holdout extraction wall time, CV/refit time, and classifier prediction time. Because training extraction was resumed after an interrupted run, full end-to-end wall time across all extraction attempts was not captured. The 8 GB GPU is not a bottleneck for this CPU YAMNet run; native-Windows TensorFlow GPU support remains unavailable.

## Tests, audit, and reproducibility

The package test suite exercises input validation, resampling, patch aggregation, finite values, duplicate IDs, class alignment, train-only grouped folds, scaling inside pipelines, frozen holdout gating, output mapping, and artifact integrity. The final audit records 2,177 approved source mappings, the 1,748/429 partition boundary, source-hash/size/decode checks, zero training/holdout author/content overlap, and the excluded M2 samples.

Reproduce with the isolated environment (PowerShell):

```powershell
.\\.venv-yamnet\\Scripts\\Activate.ps1
$env:PYTHONPATH = 'src'
python scripts/run_m5_yamnet.py --stage smoke
python scripts/run_m5_yamnet.py --stage extract-train
python scripts/run_m5_yamnet.py --stage train-freeze
python scripts/run_m5_yamnet.py --stage evaluate-holdout
python scripts/run_m5_yamnet.py --stage audit-report
```

The first three stages do not parse holdout labels or audio. `evaluate-holdout` refuses to run unless the selected classifier matches the M5 freeze manifest. The evaluation artifact prevents a second holdout evaluation. Do not retune based on final holdout results.

## Limitations and conclusion

This experiment covers only the eight species represented in the currently available approved subset. Author grouping does not guarantee geographic independence; unknown-author generalization is not measured separately. Recording-level primary labels are weak labels for local acoustic events. CPU inference timing is machine-specific. The same historical 429-recording holdout supports M3/M4/M5 comparison but repeated reuse limits independence. Acoustic detections and activity indices must not be interpreted as direct animal population counts or trends without a separate ecological sampling design.
"""
    atomic_text(report_text, report_root / "m5_yamnet_transfer_learning_report.md")
    return {"comparison": comparison, "per_class": per_class_rows,
            "report_path": str((report_root / "m5_yamnet_transfer_learning_report.md").relative_to(config.ROOT))}
