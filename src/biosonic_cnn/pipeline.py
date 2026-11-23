import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from biosonic_ml.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from . import config
from .data import make_internal_partition, make_segment_manifest, source_table
from .mels import (extract_all_mels, fit_train_normalizer, make_segment_index, mel_config)
from .model import BioSonicCNN, architecture_config
from .train import _plot_architecture, train_cnn, write_training_plots


def prepare():
    if config.M4_SPLIT.exists() or config.M4_SPLIT_SUMMARY.exists() or config.M4_SEGMENTS.exists():
        raise FileExistsError("M4 split outputs already exist. Inspect them and use the existing manifest; refusing to overwrite.")
    recording_split, summary = make_internal_partition(source_table())
    segments = pd.read_csv(config.M3_SEGMENTS)
    segment_index = make_segment_manifest(recording_split, segments)
    return recording_split, segment_index, summary


def _ensure_m4_split():
    recording_split = pd.read_csv(config.M4_SPLIT, dtype={"filename": str, "primary_label": str,
                                                           "group_id": str, "sha256": str})
    segments = pd.read_csv(config.M3_SEGMENTS, dtype={"filename": str, "primary_label": str})
    needed_columns = {"expected_size_bytes", "actual_size_bytes", "sample_count", "author_group_id"}
    if not config.M4_SEGMENTS.exists():
        return recording_split, make_segment_manifest(recording_split, segments)
    segment_index = pd.read_csv(config.M4_SEGMENTS)
    if not needed_columns.issubset(segment_index.columns):
        # Earlier in this same setup pass the generated M4 mapping omitted source
        # validation fields; rebuild that M4-only artifact from authoritative M3 inputs.
        return recording_split, make_segment_manifest(recording_split, segments)
    if len(segment_index) != len(segments) or segment_index.filename.duplicated().sum() != 0:
        # One recording repeats across segment rows; uniqueness is by file + segment index.
        if len(segment_index) != len(segments) or segment_index.duplicated(["filename", "segment_index"]).any():
            raise RuntimeError("Existing M4 segment manifest is inconsistent")
    if set(segment_index.m4_partition) != {"internal_train", "internal_validation", "holdout"}:
        raise RuntimeError("M4 segment manifest partitions are inconsistent")
    return recording_split, segment_index


def _write_sample_spectrograms(segment_index, normalizer):
    from .dataset import MelWindowDataset
    sample_rows = segment_index.groupby("primary_label", sort=True).head(1).head(4).reset_index(drop=True)
    ds = MelWindowDataset(sample_rows, normalizer)
    fig, axes = plt.subplots(len(sample_rows), 1, figsize=(10, 2.2 * len(sample_rows)), squeeze=False)
    for idx, ax in enumerate(axes[:, 0]):
        row = sample_rows.iloc[idx]
        item = ds[idx]["x"].squeeze(0).numpy()
        valid = int(row.valid_frames)
        ax.imshow(item[:, :valid], origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"{row.primary_label} · {row.filename} · segment {row.segment_index}; training-only standardized log-power Mel")
        ax.set_ylabel("Mel band")
    axes[-1, 0].set_xlabel("STFT frame (hop = 256 samples / 8 ms)")
    fig.tight_layout()
    fig.savefig(config.M4_REPORT_DIR / "sample_mel_spectrograms.png", dpi=160)
    plt.close(fig)


def _persist_evaluation(result):
    config.M4_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, metrics in (("internal_validation", result["internal_validation"]), ("holdout", result["holdout"])):
        report = pd.DataFrame(metrics["classification_report"]).T.rename_axis("class").reset_index()
        atomic_write_csv(report, config.M4_REPORT_DIR / f"{name}_classification_report.csv")
        pd.DataFrame(metrics["confusion_matrix"], index=config.INDEX_TO_LABEL,
                     columns=config.INDEX_TO_LABEL).to_csv(config.M4_REPORT_DIR / f"{name}_confusion_matrix.csv")
        atomic_write_json({k: v for k, v in metrics.items() if k not in {"predictions", "classification_report",
                                                                          "confusion_matrix", "probabilities",
                                                                          "y_true", "names"}},
                          config.M4_REPORT_DIR / f"{name}_metrics.json")
    holdout = result["holdout"]
    rows = []
    for i, filename in enumerate(holdout["names"]):
        rows.append({"filename": filename, "true_label": config.INDEX_TO_LABEL[int(holdout["y_true"][i])],
                     "predicted_label": config.INDEX_TO_LABEL[int(holdout["predictions"][i])],
                     **{f"prob_{label}": float(holdout["probabilities"][i, idx])
                        for idx, label in enumerate(config.INDEX_TO_LABEL)}})
    atomic_write_csv(pd.DataFrame(rows), config.M4_REPORT_DIR / "holdout_recording_predictions.csv")
    _plot_holdout_diagnostics()


def _plot_holdout_diagnostics():
    """Render diagnostics from already-persisted recording-level reports; no inference."""
    labels = config.INDEX_TO_LABEL
    matrix = pd.read_csv(config.M4_REPORT_DIR / "holdout_confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("M4 CNN recording-level confusion matrix (existing holdout)")
    fig.colorbar(image, ax=ax, label="Recordings")
    midpoint = matrix.to_numpy().max() / 2 if matrix.to_numpy().size else 0
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = int(matrix.iloc[row, col])
            ax.text(col, row, str(value), ha="center", va="center",
                    color="white" if value > midpoint else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.M4_REPORT_DIR / "holdout_confusion_matrix.png", dpi=160)
    plt.close(fig)

    reports = []
    for model_name, report_name in (
        ("Logistic Regression", "logistic_regression_classification_report.csv"),
        ("Random Forest", "random_forest_classification_report.csv"),
        ("SVM", "svm_classification_report.csv"),
    ):
        report = pd.read_csv(config.M3_REPORT_DIR / report_name)
        reports.append((model_name, report.set_index("class")["f1-score"].reindex(labels)))
    cnn_report = pd.read_csv(config.M4_REPORT_DIR / "holdout_classification_report.csv")
    reports.append(("Custom CNN", cnn_report.set_index("class")["f1-score"].reindex(labels)))
    comparison = pd.DataFrame({name: scores for name, scores in reports}, index=labels)
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.plot(kind="bar", ax=ax, color=["#4d7597", "#7b9e6b", "#b0a06a", "#b77945"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recording-level holdout F1")
    ax.set_xlabel("Species label")
    ax.set_title("Per-class F1 on the shared 429-recording holdout")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(config.M4_REPORT_DIR / "per_class_holdout_f1_comparison.png", dpi=160)
    plt.close(fig)


def _compare_m3(result):
    m3_rows = []
    for model_name in ("logistic_regression", "random_forest", "svm"):
        data = json.loads((config.M3_REPORT_DIR / f"{model_name}_results.json").read_text(encoding="utf-8"))
        hold = data["holdout"]
        m3_rows.append({"model": f"M3 {model_name}", "milestone": "M3", "cv_macro_f1": data["best_cv_macro_f1"],
                        "internal_validation_macro_f1": np.nan, "holdout_accuracy": hold["accuracy"],
                        "holdout_macro_precision": hold["precision_macro"], "holdout_macro_recall": hold["recall_macro"],
                        "holdout_macro_f1": hold["f1_macro"], "holdout_weighted_f1": hold["f1_weighted"],
                        "holdout_roc_auc_ovr_macro": hold.get("roc_auc_ovr_macro"),
                        "note": "Existing M3 holdout result; same 429 recordings; M3 CV only inside M3 training."})
    internal = result["internal_validation"]
    hold = result["holdout"]
    m3_rows.append({"model": "M4 custom CNN", "milestone": "M4", "cv_macro_f1": np.nan,
                    "internal_validation_macro_f1": internal["f1_macro"], "holdout_accuracy": hold["accuracy"],
                    "holdout_macro_precision": hold["precision_macro"], "holdout_macro_recall": hold["recall_macro"],
                    "holdout_macro_f1": hold["f1_macro"], "holdout_weighted_f1": hold["f1_weighted"],
                    "holdout_roc_auc_ovr_macro": hold["roc_auc_ovr_macro"],
                    "note": "Selected on internal validation, then scored on prior M3 holdout; not a newly untouched test."})
    comparison = pd.DataFrame(m3_rows)
    atomic_write_csv(comparison, config.M4_REPORT_DIR / "m3_m4_model_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    names = comparison.model.tolist()
    values = comparison.holdout_macro_f1.to_numpy(dtype=float)
    colors = ["#4d7597"] * 3 + ["#b77945"]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recording-level holdout macro F1")
    ax.set_title("Existing M3 and M4 models on the same 429-recording holdout")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(config.M4_REPORT_DIR / "m3_m4_holdout_comparison.png", dpi=160)
    plt.close(fig)
    return comparison


def _make_report(split_summary, result, comparison):
    model = result["model"]
    arch = architecture_config(model)
    val, hold = result["internal_validation"], result["holdout"]
    history = result["history"]
    best_row = history.loc[history.epoch.eq(result["best_epoch"])].iloc[-1]
    m3_train = int(split_summary["internal_train_recordings"])
    m3_val = int(split_summary["internal_validation_recordings"])
    hold_count = int(split_summary["final_holdout_recordings"])
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback"
    cuda_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if torch.cuda.is_available() else 0
    mel_cfg = mel_config()
    lines = ["# M4 Custom CNN — eight-class interim experiment", "",
             "## Scope and existing benchmark", "",
             "This experiment continues the eight-class interim M3 scope using exactly the 2,177 approved recordings. The four unavailable approved species (`swtman1`, `smbani`, `yecpar`, `spispi1`) remain excluded; this is not a 12-class result. The original M3 recording split was preserved: 1,748 M3 training recordings and the same 429-recording holdout. M3 has already evaluated that holdout, so CNN scores below are a holdout evaluation on the existing benchmark, not a newly untouched external test.", "",
             "## Internal validation split", "",
             f"A deterministic author/content-grouped `StratifiedGroupKFold(n_splits={config.INTERNAL_SPLITS}, shuffle=True, random_state=42)` subdivided only the 1,748 M3 training recordings. The selected feasible fold was {split_summary['selected_fold']}: **{m3_train:,} internal training**, **{m3_val:,} internal validation** ({split_summary['internal_validation_fraction_of_m3_train']:.2%}), with the original **{hold_count:,} holdout** left out of all training, early stopping, and tuning. All eight classes and all 16 label×collection strata were represented internally on both sides. There was no author/content-group overlap; all 109 literal Unknown-author training recordings remain in internal training.", "",
             "## Mel representation and preprocessing", "",
             f"One spectrogram is generated for each of the {len(pd.read_csv(config.M4_SEGMENTS)):,} windows in the M3 segment manifest. The cache contains log-power values as float16 (disk cache); values are converted to float32 and normalized in the Dataset. Configuration: 32 kHz mono, five-second / 160,000-sample non-overlapping windows, `n_fft=win_length=1024`, `hop_length=256`, periodic Hann, 128 Slaney-scale Slaney-normalized Mel bands, 0–16 kHz, no centered padding. Incomplete/short tails are right-zero-padded and carry valid sample/frame counts. The extractor uses the same 128-band Slaney power-Mel definition as M3; it uses an explicit `10*log10(max(power, 1e-10))` transform and periodic Hann consistent with M3's feature code. The persisted M3 exploratory plots used a distinct visualization convention, so no claim of bitwise equality to their plotted matrices is made.", "",
             "Per-band mean and standard deviation are fit only on valid frames in the internal-training recordings. Invalid padded time frames are set to zero after normalization. The final holdout contributes no normalization statistics.", "",
             "## Custom architecture", "",
             "The model is trained from scratch with no pretrained weights. Four blocks each contain two 3×3 Conv2D–BatchNorm–ReLU layers, 2× max pooling, and Dropout2D. Channels grow 1→16→32→64→128. Adaptive average pooling produces one 128-value vector, followed by a 128-unit ReLU/dropout layer and eight raw logits trained with CrossEntropyLoss.", "",
             f"Trainable parameters: **{arch['trainable_parameters']:,}**. Model input is `[batch, 1, 128, 622]`; output is `[batch, 8]`.", "",
             "## Training configuration", "",
             f"Device: **{result['device']}** ({cuda_name}, {cuda_total:.1f} GiB reported VRAM). Batch size {config.BATCH_SIZE}; AdamW (lr={config.LEARNING_RATE:g}, weight decay={config.WEIGHT_DECAY:g}); ReduceLROnPlateau on internal-validation recording macro F1; up to {config.MAX_EPOCHS} epochs with early stopping patience {config.EARLY_STOPPING_PATIENCE}. Seed 42 and deterministic cuDNN settings. CUDA mixed precision is enabled when CUDA is available. A weighted segment sampler gives equal total sampling weight per class and per recording, preventing long recordings or large classes from dominating the minibatch stream.", "",
             f"Training stopped at epoch {int(history.epoch.max())}; best epoch **{result['best_epoch']}**. Best-epoch training loss: {best_row.train_loss_segment_balanced:.4f}; sampled recording-aggregate training macro F1: {best_row.train_macro_f1_sampled_recording_aggregate:.4f}; internal-validation segment loss: {best_row.internal_validation_loss_segment_mean:.4f}; internal-validation recording macro F1: {best_row.internal_validation_macro_f1_recording:.4f}. Training duration: {result['training_seconds']:.1f} seconds. Peak CUDA allocated memory: {result['peak_cuda_memory_bytes'] / 1024**2:.1f} MiB ({result['peak_cuda_memory_bytes']:.0f} bytes).", "",
             "Training loss is segment-sampled; train macro F1 aggregates sampled segment probabilities by recording. Validation and holdout losses are segment-level diagnostics, while their headline metrics average segment probabilities per recording before scoring so each recording counts once.", "",
             "## Recording-level results", "",
             "CNN segment probabilities were averaged across all windows from each recording. The resulting recording-level scores are:", "",
             "| Partition | Recordings | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | OVR macro ROC-AUC | Inference time | ms / recording |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| Internal validation | {val['recording_count']} | {val['accuracy']:.4f} | {val['precision_macro']:.4f} | {val['recall_macro']:.4f} | {val['f1_macro']:.4f} | {val['f1_weighted']:.4f} | {val['roc_auc_ovr_macro']:.4f} | {val['inference_seconds']:.2f}s | {val['inference_ms_per_record']:.3f} |",
             f"| Existing M3 holdout | {hold['recording_count']} | {hold['accuracy']:.4f} | {hold['precision_macro']:.4f} | {hold['recall_macro']:.4f} | {hold['f1_macro']:.4f} | {hold['f1_weighted']:.4f} | {hold['roc_auc_ovr_macro']:.4f} | {hold['inference_seconds']:.2f}s | {hold['inference_ms_per_record']:.3f} |", "",
             "### Comparison with traditional ML", "",
             "The M3 models use their existing five-fold training-CV mean macro F1 for model selection; the CNN uses internal-validation macro F1 for early stopping. All holdout columns below refer to the same previously evaluated M3 429 recordings.", "",
             "| Model | Selection metric value | Holdout accuracy | Holdout macro F1 | Holdout weighted F1 | Holdout OVR ROC-AUC |", "| --- | ---: | ---: | ---: | ---: | ---: |",
             *[f"| {r.model} | {('CV macro-F1 ' + format(r.cv_macro_f1, '.4f')) if pd.notna(r.cv_macro_f1) else ('Internal validation macro-F1 ' + format(r.internal_validation_macro_f1, '.4f'))} | {r.holdout_accuracy:.4f} | {r.holdout_macro_f1:.4f} | {r.holdout_weighted_f1:.4f} | {'unsupported' if pd.isna(r.holdout_roc_auc_ovr_macro) else f'{r.holdout_roc_auc_ovr_macro:.4f}'} |" for r in comparison.itertuples(index=False)], "",
             "The different selection scores are not directly interchangeable. No unsupported claim that the CNN is better is made. The shared holdout has previously influenced M3 development, so the M4 holdout comparison is benchmark reuse rather than independent confirmation.", "",
             "## Limitations and next work", "",
             "The per-window target is inherited from each recording's primary label and may be weak for silent or mixed-species windows. Padding may affect CNN predictions despite the explicit frame mask and neutral normalized padding. Unknown authors remain training-only, so generalization to Unknown contributors cannot be assessed. Author grouping does not ensure spatial independence. This is recording-level primary-label classification, not soundscape multi-label detection or a direct population estimate. Results apply only to the current eight-class subset.", "",
             "## Reproducibility", "",
             "`.venv\\Scripts\\python.exe -m pytest tests\\test_m4_cnn.py tests\\test_m3_pipeline.py tests\\test_download_approved_audio.py -q`", "",
             "`.venv\\Scripts\\python.exe scripts\\run_m4_cnn.py --stage prepare`", "",
             "`.venv\\Scripts\\python.exe scripts\\run_m4_cnn.py --stage mels`", "",
             "`.venv\\Scripts\\python.exe scripts\\run_m4_cnn.py --stage train`", "",
             "To resume an interrupted training run, add `--resume` to the train command. To resume feature generation, rerun the mels stage; completed atomic per-recording shards are checked against source SHA-256 and the feature configuration. No model or feature output should be manually relabeled or edited.", ""]
    atomic_write_text(config.M4_REPORT, "\n".join(lines))


def run(stage="all", resume=False):
    if stage in {"prepare", "all"}:
        recording_split, segment_index, split_summary = prepare()
        print(f"M4 split ready: {split_summary['internal_train_recordings']} internal train, {split_summary['internal_validation_recordings']} internal validation, {split_summary['final_holdout_recordings']} original holdout; selected fold {split_summary['selected_fold']}", flush=True)
        if stage == "prepare":
            return
    else:
        if not config.M4_SPLIT.exists() or not config.M4_SPLIT_SUMMARY.exists():
            raise FileNotFoundError("M4 split missing; run --stage prepare first")
        recording_split, segment_index = _ensure_m4_split()
        split_summary = json.loads(config.M4_SPLIT_SUMMARY.read_text(encoding="utf-8"))
    if stage in {"mels", "all"}:
        def progress(done, total):
            print(f"M4 Mel generation: {done:,}/{total:,} recordings", flush=True)
        extract_all_mels(segment_index, progress=progress)
        mel_segments = make_segment_index(segment_index)
        normalizer = fit_train_normalizer(mel_segments)
        print(f"Mel cache ready: {len(mel_segments):,} windows; normalization fitted on {normalizer['fit_recordings']:,} internal-training recordings", flush=True)
        if stage == "mels":
            return
    else:
        if not (config.M4_DATA_ROOT / "mel_normalizer.json").exists():
            raise FileNotFoundError("Mel normalization missing; run --stage mels first")
        mel_segments = pd.read_csv(config.M4_DATA_ROOT / "mel_segment_index.csv")
        normalizer = json.loads((config.M4_DATA_ROOT / "mel_normalizer.json").read_text(encoding="utf-8"))
        if normalizer.get("fit_partition") != "internal_train":
            raise RuntimeError("Existing normalizer was not fit exclusively on internal training")
    if stage in {"train", "all"}:
        if config.M4_MODEL_ROOT.exists() and not resume and any(config.M4_MODEL_ROOT.iterdir()):
            raise FileExistsError("M4 model outputs exist. Pass --resume for an interrupted training run; refusing overwrite.")
        config.M4_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        arch = architecture_config(BioSonicCNN())
        atomic_write_json(arch, config.M4_REPORT_DIR / "architecture.json")
        _plot_architecture(config.M4_REPORT_DIR / "cnn_architecture.png")
        _write_sample_spectrograms(mel_segments, normalizer)
        result = train_cnn(mel_segments, normalizer, resume=resume)
        write_training_plots(result["history"])
        _persist_evaluation(result)
        comparison = _compare_m3(result)
        _make_report(split_summary, result, comparison)
        print(f"CNN finished: best epoch {result['best_epoch']}; internal validation macro F1={result['internal_validation']['f1_macro']:.4f}; existing holdout macro F1={result['holdout']['f1_macro']:.4f}; peak CUDA={result['peak_cuda_memory_bytes'] / 1024**2:.1f} MiB", flush=True)
