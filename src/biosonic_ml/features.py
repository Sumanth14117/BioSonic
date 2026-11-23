import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.signal import get_window

from . import config
from .dataset import sha256_file
from .io_utils import atomic_write_csv, atomic_write_json


def feature_configuration() -> dict:
    return {"version": config.FEATURE_VERSION, "sample_rate": config.SR, "segment_seconds": config.SEGMENT_SECONDS,
            "segment_samples": config.SEGMENT_SAMPLES, "tail_policy": "right-zero-pad final partial window and short recordings; summarize complete source frames, using one zero-padded STFT frame if source tail is shorter than n_fft; report padded sample counts",
            "n_fft": config.N_FFT, "win_length": config.WIN, "hop_length": config.HOP,
            "window": "periodic Hann", "mel_bands": config.N_MELS, "fmin": 0, "fmax": config.SR / 2,
            "mel_scale": "Slaney (htk=False)", "mel_norm": "Slaney", "spectrogram": "power",
            "log_mel": "10*log10(max(power_mel, 1e-10)) dB relative to 1 power unit",
            "mfcc": "DCT-II, orthonormal, first 13 including coefficient 0",
            "delta": "Librosa Savitzky-Golay delta, width=9, order 1 and 2 when >=9 frames; otherwise finite-difference gradient",
            "aggregation": "for each segment summarize frame-wise features with mean/std, then aggregate each statistic across source segments with mean/std",
            "feature_names": list(config.FEATURE_NAMES)}


def _stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate([np.mean(values, axis=-1), np.std(values, axis=-1)])


def _delta(x: np.ndarray, order: int) -> np.ndarray:
    if x.shape[-1] >= 9:
        return librosa.feature.delta(x, width=9, order=order, axis=-1, mode="interp")
    result = x
    for _ in range(order):
        result = np.gradient(result, axis=-1) if result.shape[-1] > 1 else np.zeros_like(result)
    return result


@lru_cache(maxsize=1)
def _mel_filter():
    return librosa.filters.mel(sr=config.SR, n_fft=config.N_FFT, n_mels=config.N_MELS,
                               fmin=0, fmax=config.SR / 2, htk=False, norm="slaney", dtype=np.float64)


def segment_feature(wave: np.ndarray, valid_samples: int, sr: int = config.SR) -> tuple[np.ndarray, int]:
    """Return 102 per-segment stats and count of complete source frames."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if sr != config.SR:
        raise ValueError(f"Expected {config.SR} Hz, got {sr}")
    if len(wave) != config.SEGMENT_SAMPLES:
        raise ValueError("Feature segment must be exactly five seconds after padding")
    if not np.isfinite(wave).all() or not 0 < valid_samples <= len(wave):
        raise ValueError("Invalid segment samples")
    complete_frames = max(1, 1 + (valid_samples - config.N_FFT) // config.HOP)
    stft = librosa.stft(wave, n_fft=config.N_FFT, hop_length=config.HOP, win_length=config.WIN,
                        window="hann", center=False)
    stft = stft[:, :complete_frames]
    magnitude = np.abs(stft).astype(np.float64)
    power = magnitude ** 2
    mel_power = _mel_filter() @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:config.N_MFCC]
    delta, delta2 = _delta(mfcc, 1), _delta(mfcc, 2)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
    denom = np.maximum(magnitude.sum(axis=0), 1e-12)
    centroid = (freqs[:, None] * magnitude).sum(axis=0) / denom
    bandwidth = np.sqrt((magnitude * (freqs[:, None] - centroid[None, :]) ** 2).sum(axis=0) / denom)
    cumsum = np.cumsum(magnitude, axis=0)
    roll_idx = (cumsum >= (0.85 * cumsum[-1:])).argmax(axis=0)
    rolloff = freqs[roll_idx]
    frame_source = wave[:max(valid_samples, config.N_FFT)]
    framed = librosa.util.frame(frame_source, frame_length=config.N_FFT, hop_length=config.HOP)
    framed = framed[:, :complete_frames]
    zcr = np.mean(np.abs(np.diff(np.signbit(framed), axis=0)), axis=0)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=0))
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr, n_bands=6, fmin=200,
                                                  quantile=0.02, linear=False)
    # Pack segment-level mean/std for each feature in a stable order.
    blocks = [(mfcc, config.N_MFCC), (delta, config.N_MFCC), (delta2, config.N_MFCC),
              (centroid[None, :], 1), (bandwidth[None, :], 1), (rolloff[None, :], 1),
              (zcr[None, :], 1), (rms[None, :], 1), (contrast, 7)]
    packed = np.concatenate([np.stack([np.mean(arr, axis=-1), np.std(arr, axis=-1)], axis=1).ravel() for arr, _ in blocks])
    if packed.size != len(config.FEATURE_NAMES):
        raise AssertionError(f"Unexpected segment feature size {packed.size}")
    return packed.astype(np.float32), complete_frames


def recording_features(path: Path, expected_sr=config.SR) -> tuple[np.ndarray, dict]:
    segment_summaries, segment_valid_samples, frame_total = [], [], 0
    with sf.SoundFile(str(path), mode="r") as audio:
        if audio.samplerate != expected_sr or audio.channels < 1:
            raise ValueError(f"Invalid audio shape/rate for {path.name}")
        original_count = len(audio)
        duration = original_count / audio.samplerate
        while True:
            block = audio.read(config.SEGMENT_SAMPLES, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite values in {path.name}")
            mono = block.mean(axis=1, dtype=np.float64).astype(np.float32) if block.shape[1] > 1 else block[:, 0]
            valid = len(mono)
            if valid < config.SEGMENT_SAMPLES:
                mono = np.pad(mono, (0, config.SEGMENT_SAMPLES - valid))
            vector, nframes = segment_feature(mono, valid, audio.samplerate)
            segment_summaries.append(vector)
            segment_valid_samples.append(valid)
            frame_total += nframes
    if not segment_summaries:
        raise ValueError(f"Empty audio file: {path}")
    per_segment = np.vstack(segment_summaries).astype(np.float64)
    # Per-recording vector: mean and standard deviation across segment-level stats.
    out = np.concatenate([per_segment.mean(axis=0), per_segment.std(axis=0)]).astype(np.float32)
    info = {"sample_count": original_count, "duration_seconds": duration,
            "segment_count": len(segment_summaries), "complete_five_second_segments": int(original_count // config.SEGMENT_SAMPLES),
            "partial_or_short_segments": int(original_count % config.SEGMENT_SAMPLES != 0 or original_count < config.SEGMENT_SAMPLES),
            "padded_samples": int(sum(config.SEGMENT_SAMPLES - x for x in segment_valid_samples)),
            "source_sample_coverage": 1.0, "analyzed_stft_frames": frame_total,
            "sub_fft_tail_segments": int(sum(x < config.N_FFT for x in segment_valid_samples))}
    return out, info


def _config_hash() -> str:
    payload = json.dumps(feature_configuration(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_feature_table(split: pd.DataFrame, root=config.FEATURE_ROOT, progress=None) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(root)
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash()
    config_path = root / "feature_config.json"
    if config_path.exists():
        current = json.loads(config_path.read_text(encoding="utf-8"))
        if current.get("config_hash") != config_hash:
            raise RuntimeError("Existing feature configuration differs; refusing to overwrite artifacts")
    else:
        atomic_write_json({**feature_configuration(), "config_hash": config_hash}, config_path)
    vectors, records = [], []
    for index, row in enumerate(split.itertuples(index=False), start=1):
        source = Path(row.audio_path)
        if source.stat().st_size != int(row.actual_size_bytes) or sha256_file(source) != str(row.sha256):
            raise RuntimeError(f"Source file changed after split validation: {row.filename}")
        # 96-bit path key keeps filenames compact on Windows while retaining
        # ample collision resistance for this small, fixed working set.
        key = hashlib.sha256(str(row.filename).encode()).hexdigest()[:24]
        shard = shards / f"{key}.npz"
        metadata = {"filename": row.filename, "primary_label": row.primary_label, "collection": row.collection,
                    "partition": row.partition, "group_id": row.group_id, "author_group_id": row.author_group_id,
                    "sha256": row.sha256, "feature_config_hash": config_hash}
        if shard.exists():
            with np.load(shard, allow_pickle=False) as saved:
                if (str(saved["feature_config_hash"].item()) != config_hash or
                        str(saved["filename"].item()) != row.filename or
                        str(saved["source_sha256"].item()) != str(row.sha256)):
                    raise RuntimeError(f"Invalid/stale feature shard for {row.filename}; preserved at {shard}")
                vector = saved["features"].astype(np.float32)
                info = json.loads(str(saved["info"].item()))
        else:
            vector, info = recording_features(Path(row.audio_path))
            tmp = shard.with_name(shard.name + ".tmp.npz")
            shard.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, features=vector, feature_config_hash=np.array(config_hash),
                                    filename=np.array(row.filename), source_sha256=np.array(row.sha256),
                                    info=np.array(json.dumps(info)))
            os.replace(tmp, shard)
        if vector.shape != (len(config.FEATURE_NAMES) * 2,):
            raise ValueError(f"Unexpected feature dimension for {row.filename}: {vector.shape}")
        vectors.append(vector)
        records.append({**metadata, **info})
        if progress and (index % 50 == 0 or index == len(split)):
            progress(index, len(split))
    matrix = np.vstack(vectors).astype(np.float32)
    meta = pd.DataFrame(records)
    if len(matrix) != len(split) or not np.isfinite(matrix).all():
        raise ValueError("Feature table has missing rows or non-finite values")
    root.mkdir(parents=True, exist_ok=True)
    npz = root / "features.npz"
    tmp = root / "features.tmp.npz"
    np.savez_compressed(tmp, X=matrix, y=meta.primary_label.to_numpy(dtype=str), feature_names=np.array(
        [f"{n}_{s}" for n in config.FEATURE_NAMES for s in ("segment_mean", "segment_std")]),
        filenames=meta.filename.to_numpy(dtype=str))
    os.replace(tmp, npz)
    atomic_write_csv(meta, root / "feature_metadata.csv")
    return matrix, meta
