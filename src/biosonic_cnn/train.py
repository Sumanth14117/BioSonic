import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from biosonic_ml.io_utils import atomic_write_csv, atomic_write_json
from . import config
from .dataset import MelWindowDataset
from .model import BioSonicCNN, architecture_config


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def aggregate_recording_probabilities(probabilities, filenames, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    filenames = np.asarray(filenames, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(filenames) or len(labels) != len(filenames):
        raise ValueError("Segment predictions and recording metadata have inconsistent dimensions")
    agg, y = {}, {}
    for idx, name in enumerate(filenames):
        if name in y and y[name] != labels[idx]:
            raise ValueError(f"Inconsistent labels among segments for recording {name}")
        y[name] = int(labels[idx])
        agg[name] = agg.get(name, np.zeros(probabilities.shape[1], dtype=np.float64)) + probabilities[idx]
    names = sorted(agg)
    counts = {name: int(np.count_nonzero(filenames == name)) for name in names}
    mean_probs = np.vstack([agg[name] / counts[name] for name in names])
    return names, np.asarray([y[name] for name in names], dtype=np.int64), mean_probs


def calculate_recording_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(config.INDEX_TO_LABEL):
        raise ValueError("Expected eight-class recording probabilities")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid prediction probabilities")
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(config.INDEX_TO_LABEL))
    try:
        auc = float(roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {"accuracy": float(accuracy_score(y_true, predictions)),
            "precision_macro": float(precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
            "roc_auc_ovr_macro": auc,
            "predictions": predictions,
            "classification_report": classification_report(y_true, predictions, labels=labels,
                                                             target_names=config.INDEX_TO_LABEL, output_dict=True,
                                                             zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels)}


def _atomic_torch_save(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _loader(dataset, *, shuffle=False, sampler=None):
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle, sampler=sampler,
                      num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, drop_last=False)


def _forward_recordings(model, loader, device, criterion=None):
    model.eval()
    sums, counts, labels_by_name = {}, {}, {}
    loss_sum, loss_count = 0.0, 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                loss_sum += float(criterion(logits, y).item()) * len(y)
                loss_count += len(y)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            y_cpu = y.cpu().numpy()
            for name, prob, label in zip(batch["filename"], probs, y_cpu):
                if name in labels_by_name and labels_by_name[name] != int(label):
                    raise ValueError(f"Recording label mismatch for {name}")
                labels_by_name[name] = int(label)
                sums[name] = sums.get(name, np.zeros(len(config.INDEX_TO_LABEL), dtype=np.float64)) + prob
                counts[name] = counts.get(name, 0) + 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    names = sorted(sums)
    probabilities = np.vstack([sums[n] / counts[n] for n in names])
    y_true = np.asarray([labels_by_name[n] for n in names], dtype=np.int64)
    metrics = calculate_recording_metrics(y_true, probabilities)
    metrics.update({"names": names, "y_true": y_true, "probabilities": probabilities,
                    "segment_loss": loss_sum / max(loss_count, 1), "recording_count": len(names),
                    "inference_seconds": elapsed, "inference_ms_per_record": elapsed * 1000 / max(len(names), 1)})
    return metrics


def _class_balance_sampler(train_index):
    rec_counts = train_index.groupby("primary_label").filename.nunique().to_dict()
    segments_per_record = train_index.groupby("filename").size().to_dict()
    weights = np.asarray([1.0 / (rec_counts[row.primary_label] * segments_per_record[row.filename])
                          for row in train_index.itertuples(index=False)], dtype=np.float64)
    weights /= weights.sum()
    generator = torch.Generator().manual_seed(config.SEED)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights),
                                    replacement=True, generator=generator)
    return sampler, generator, rec_counts


def _sampled_recording_f1(prob_sums, counts, y_by_name):
    names = sorted(prob_sums)
    probs = np.vstack([prob_sums[n] / counts[n] for n in names])
    true = np.asarray([y_by_name[n] for n in names])
    return float(f1_score(true, probs.argmax(axis=1), labels=np.arange(8), average="macro", zero_division=0))


def _plot_architecture(path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    boxes = [(0.02, "Input\n1 × 128 × 622"), (0.20, "Conv blocks\n1→16→32\n2× Conv-BN-ReLU + Pool"),
             (0.43, "Conv blocks\n32→64→128\n2× Conv-BN-ReLU + Pool"),
             (0.67, "Adaptive average\npool to 1×1"), (0.84, "FC 128\nReLU + Dropout"), (0.96, "8 logits")]
    for x, label in boxes:
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "fc": "#edf2f7", "ec": "#506070"}, transform=ax.transAxes)
    for a, b in zip([0.08, 0.27, 0.51, 0.75, 0.91], [0.14, 0.36, 0.60, 0.78, 0.93]):
        ax.annotate("", xy=(b, .5), xytext=(a, .5), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#35465a", "lw": 1.5})
    ax.set_title("Custom BioSonicCNN architecture (four convolutional stages)")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _load_model_state(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = BioSonicCNN()
    model.load_state_dict(state["model_state"])
    return model.to(device), state


def train_cnn(index: pd.DataFrame, normalizer: dict, resume=False):
    seed_everything(config.SEED)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BioSonicCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_idx = index[index.m4_partition.eq("internal_train")].reset_index(drop=True)
    val_idx = index[index.m4_partition.eq("internal_validation")].reset_index(drop=True)
    sampler, sampler_generator, class_counts = _class_balance_sampler(train_idx)
    train_ds = MelWindowDataset(train_idx, normalizer)
    val_ds = MelWindowDataset(val_idx, normalizer)
    train_loader = _loader(train_ds, sampler=sampler)
    train_eval_loader = _loader(train_ds, shuffle=False)
    val_loader = _loader(val_ds, shuffle=False)
    best_path = config.M4_MODEL_ROOT / "best_model.pt"
    latest_path = config.M4_MODEL_ROOT / "latest_checkpoint.pt"
    history_path = config.M4_MODEL_ROOT / "training_history.csv"
    history = []
    start_epoch, best_f1, best_epoch, stale = 0, -1.0, 0, 0
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["amp_scaler_state"])
        sampler_generator.set_state(state["sampler_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        start_epoch = int(state["epoch"])
        best_f1, best_epoch, stale = state["best_f1"], state["best_epoch"], state["stale_epochs"]
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError("CNN checkpoint exists. Pass --resume to continue safely; refusing to overwrite.")
    model_config = architecture_config(model)
    config.M4_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json({**model_config, "seed": config.SEED, "batch_size": config.BATCH_SIZE,
                       "max_epochs": config.MAX_EPOCHS, "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                       "optimizer": "AdamW", "learning_rate": config.LEARNING_RATE,
                       "weight_decay": config.WEIGHT_DECAY, "loss": "CrossEntropyLoss",
                       "sampler": "class-balanced, recording-balanced weighted segment sampler",
                       "normalization_fit_partition": "internal_train", "device": str(device)},
                      config.M4_MODEL_ROOT / "training_config.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    for epoch in range(start_epoch, config.MAX_EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        loss_total, seen = 0.0, 0
        train_probs, train_n, train_y = {}, {}, {}
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach().item()) * len(y)
            seen += len(y)
            probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
            yvals = y.detach().cpu().numpy()
            for name, prob, label in zip(batch["filename"], probs, yvals):
                train_probs[name] = train_probs.get(name, np.zeros(8)) + prob
                train_n[name] = train_n.get(name, 0) + 1
                train_y[name] = int(label)
        train_loss = loss_total / max(seen, 1)
        train_f1 = _sampled_recording_f1(train_probs, train_n, train_y)
        val_metrics = _forward_recordings(model, val_loader, device, criterion)
        scheduler.step(val_metrics["f1_macro"])
        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        row = {"epoch": epoch + 1, "train_loss_segment_balanced": train_loss,
               "train_macro_f1_sampled_recording_aggregate": train_f1,
               "internal_validation_loss_segment_mean": val_metrics["segment_loss"],
               "internal_validation_macro_f1_recording": val_metrics["f1_macro"],
               "internal_validation_accuracy_recording": val_metrics["accuracy"],
               "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": epoch_seconds,
               "peak_cuda_memory_bytes": peak_memory}
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), history_path)
        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1, best_epoch, stale = val_metrics["f1_macro"], epoch + 1, 0
            best_state = {"epoch": epoch + 1, "model_state": model.state_dict(), "best_f1": best_f1,
                          "architecture": model_config, "seed": config.SEED, "normalizer": normalizer}
            _atomic_torch_save(best_state, best_path)
        else:
            stale += 1
        checkpoint = {"epoch": epoch + 1, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                      "amp_scaler_state": scaler.state_dict(), "sampler_generator_state": sampler_generator.get_state(),
                      "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                      "python_rng_state": random.getstate(), "best_f1": best_f1,
                      "best_epoch": best_epoch, "stale_epochs": stale, "history": history}
        _atomic_torch_save(checkpoint, latest_path)
        print(f"Epoch {epoch+1:02d}/{config.MAX_EPOCHS}: train_loss={train_loss:.4f} train_macroF1={train_f1:.4f} internal_val_loss={val_metrics['segment_loss']:.4f} internal_val_macroF1={val_metrics['f1_macro']:.4f} time={epoch_seconds:.1f}s peak_vram={peak_memory/1048576:.0f}MB", flush=True)
        if stale >= config.EARLY_STOPPING_PATIENCE:
            break
    train_seconds = time.perf_counter() - total_start
    # Re-evaluate best checkpoint on internal validation only; this is still model selection data.
    best_model, best_state = _load_model_state(best_path, device)
    internal_metrics = _forward_recordings(best_model, val_loader, device, criterion)
    # Only after best epoch and all training decisions are frozen, evaluate the prior M3 holdout.
    holdout_idx = index[index.m4_partition.eq("holdout")].reset_index(drop=True)
    holdout_ds = MelWindowDataset(holdout_idx, normalizer)
    holdout_loader = _loader(holdout_ds, shuffle=False)
    holdout_metrics = _forward_recordings(best_model, holdout_loader, device, criterion)
    return {"model": best_model, "best_epoch": int(best_state["epoch"]), "best_internal_validation_macro_f1": best_f1,
            "training_seconds": train_seconds, "history": pd.DataFrame(history),
            "internal_validation": internal_metrics, "holdout": holdout_metrics,
            "peak_cuda_memory_bytes": max((float(x.get("peak_cuda_memory_bytes", 0)) for x in history), default=0),
            "device": str(device), "class_counts": class_counts}


def write_training_plots(history: pd.DataFrame, report_dir=config.M4_REPORT_DIR):
    report_dir.mkdir(parents=True, exist_ok=True)
    for train_col, val_col, title, filename, ylabel in [
        ("train_loss_segment_balanced", "internal_validation_loss_segment_mean", "Training and internal validation loss", "training_validation_loss.png", "Cross-entropy loss"),
        ("train_macro_f1_sampled_recording_aggregate", "internal_validation_macro_f1_recording", "Training and internal validation macro F1", "training_validation_macro_f1.png", "Recording-level macro F1")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history.epoch, history[train_col], label="Training", marker="o")
        ax.plot(history.epoch, history[val_col], label="Internal validation", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(report_dir / filename, dpi=160)
        plt.close(fig)
