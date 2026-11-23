"""Input validation and in-memory audio preparation for YAMNet."""

from pathlib import Path

import numpy as np
import resampy
import soundfile as sf

from . import config


class AudioValidationError(ValueError):
    pass


def prepare_waveform(path: Path, expected_sample_rate: int = config.SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    path = Path(path)
    try:
        info = sf.info(str(path))
        if info.frames <= 0:
            raise AudioValidationError(f"Empty audio file: {path}")
        if info.samplerate != expected_sample_rate:
            raise AudioValidationError(
                f"Expected {expected_sample_rate} Hz input, got {info.samplerate} Hz: {path.name}"
            )
        if info.channels != 1:
            raise AudioValidationError(
                f"M5 expects mono source audio; got {info.channels} channels: {path.name}"
            )
        pcm, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(f"Could not decode {path}: {type(exc).__name__}: {exc}") from exc

    if sample_rate != expected_sample_rate or pcm.shape[1] != 1:
        raise AudioValidationError(f"Decoded audio metadata changed while reading: {path.name}")
    # This matches the official TensorFlow Models YAMNet inference example:
    # SoundFile PCM16 decode followed by division by 32768.0.
    mono = (pcm[:, 0].astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    if not np.isfinite(mono).all():
        raise AudioValidationError(f"Non-finite source samples: {path.name}")

    if sample_rate == config.MODEL_SAMPLE_RATE:
        resampled = mono.copy()
    else:
        resampled = resampy.resample(mono, sample_rate, config.MODEL_SAMPLE_RATE).astype(np.float32)
    expected_count = (len(mono) * config.MODEL_SAMPLE_RATE) // sample_rate
    if resampled.shape != (expected_count,):
        raise AudioValidationError(
            f"Unexpected resampled sample count for {path.name}: {resampled.size} != {expected_count}"
        )
    if not np.isfinite(resampled).all():
        raise AudioValidationError(f"Non-finite resampled samples: {path.name}")
    # Do not clip filter overshoot after resampling. The official reference
    # inference path also passes the resampled waveform directly to YAMNet.
    source_int32 = pcm[:, 0].astype(np.int32)
    metadata = {
        "sample_rate": int(sample_rate),
        "channels": int(info.channels),
        "sample_count": int(info.frames),
        "duration_seconds": float(info.frames / sample_rate),
        "model_sample_rate": config.MODEL_SAMPLE_RATE,
        "model_sample_count": int(resampled.size),
        "source_decode": "SoundFile int16 then float32 / 32768, matching official inference.py",
        "source_full_scale_sample_count": int(np.count_nonzero(np.abs(source_int32) == 32768)),
        "resampler": f"resampy {resampy.__version__}, default kaiser_best",
        "resampled_length_convention": "floor(source_samples * target_rate / source_rate)",
        "resampled_peak_abs": float(np.max(np.abs(resampled), initial=0.0)),
        "resampled_samples_outside_unit_interval": int(np.count_nonzero(np.abs(resampled) > 1.0)),
        "source_min": float(np.min(mono)),
        "source_max": float(np.max(mono)),
    }
    return resampled, metadata
