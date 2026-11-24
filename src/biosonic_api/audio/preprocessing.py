"""Exact M4 CNN window and Mel normalization used during training."""

from __future__ import annotations

import math

import numpy as np
import torch

from biosonic_cnn import config as m4_config
from biosonic_cnn.mels import compute_mel_window


def build_m4_inputs(waveform: np.ndarray, sample_rate: int, normalizer: dict) -> tuple[torch.Tensor, list[int]]:
    """Create one normalized [1, 128, 622] input per five-second window."""
    source = np.asarray(waveform, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("Expected a non-empty finite mono waveform")
    if sample_rate != m4_config.SR:
        raise ValueError(f"Expected {m4_config.SR} Hz audio")
    mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(m4_config.N_MELS, 1)
    std = np.asarray(normalizer["std"], dtype=np.float32).reshape(m4_config.N_MELS, 1)
    if mean.shape != (128, 1) or std.shape != (128, 1) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid M4 training normalizer")
    if normalizer.get("fit_partition") != "internal_train":
        raise ValueError("M4 normalizer is not marked as training-only")

    segments: list[np.ndarray] = []
    valid_frames_per_segment: list[int] = []
    segment_size = m4_config.SEGMENT_SAMPLES
    for start in range(0, len(source), segment_size):
        chunk = source[start:start + segment_size]
        valid_samples = len(chunk)
        if valid_samples <= 0:
            continue
        padded = chunk if valid_samples == segment_size else np.pad(chunk, (0, segment_size - valid_samples))
        logmel, valid_frames = compute_mel_window(padded, valid_samples)
        values = (logmel - mean) / std
        values[:, valid_frames:] = 0.0
        if values.shape != (m4_config.N_MELS, 622) or not np.isfinite(values).all():
            raise ValueError("M4 preprocessing produced invalid Mel input")
        segments.append(values[None, :, :])
        valid_frames_per_segment.append(valid_frames)
    if not segments:
        raise ValueError("Audio contains no processable segments")
    batch = torch.from_numpy(np.stack(segments).astype(np.float32, copy=False))
    return batch, valid_frames_per_segment


def expected_segment_count(sample_count: int, sample_rate: int = 32_000) -> int:
    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("sample_count and sample_rate must be positive")
    return math.ceil(sample_count / (5 * sample_rate))
