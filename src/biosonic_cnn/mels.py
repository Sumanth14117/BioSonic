import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import get_window

from biosonic_ml.dataset import sha256_file
from biosonic_ml.io_utils import atomic_write_csv, atomic_write_json
from . import config
from .data import sha_key


def mel_config() -> dict:
    return {"version": config.MEL_VERSION, "sample_rate": config.SR, "channels": "mono; current corpus is mono",
            "window_seconds": config.SEGMENT_SECONDS, "window_samples": config.SEGMENT_SAMPLES,
            "n_fft": config.N_FFT, "win_length": config.WIN_LENGTH, "hop_length": config.HOP_LENGTH,
            "window": "periodic Hann (SciPy get_window fftbins=True; same periodic Hann as M3 feature extraction)",
            "center": False, "frames_per_window": 622, "mel_bands": config.N_MELS,
            "fmin": config.FMIN, "fmax": config.FMAX, "mel_scale": "Slaney (htk=False)",
            "mel_filter_normalization": "Slaney", "spectrogram": "power",
            "model_input_transform": "10*log10(max(power_mel, 1e-10)); per-band z-score from internal-training valid frames only",
            "tail_policy": "right-zero-pad incomplete and short windows to 160000 samples; record valid source samples and valid analysis frames; sub-FFT tails get one padded analysis frame; padded frames zeroed after train-only normalization",
            "source_hash_required": True}


def mel_config_hash() -> str:
    body = json.dumps(mel_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def compute_mel_window(samples: np.ndarray, valid_samples: int) -> tuple[np.ndarray, int]:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size != config.SEGMENT_SAMPLES or not np.isfinite(samples).all():
        raise ValueError("Mel window must be finite and exactly five seconds")
    if not 0 < valid_samples <= config.SEGMENT_SAMPLES:
        raise ValueError("valid_samples outside window range")
    frames = librosa.util.frame(samples, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH, axis=-1)
    if frames.shape[1] != 622:
        raise AssertionError(f"Expected 622 STFT frames; got {frames.shape[1]}")
    window = get_window("hann", config.WIN_LENGTH, fftbins=True).astype(np.float32)
    spectra = np.fft.rfft(frames * window[:, None], n=config.N_FFT, axis=0)
    power = (spectra.real ** 2 + spectra.imag ** 2).astype(np.float32)
    mel = _mel_filters() @ power
    logmel = (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)
    valid_frames = max(1, 1 + (valid_samples - config.N_FFT) // config.HOP_LENGTH)
    valid_frames = min(valid_frames, 622)
    return logmel, int(valid_frames)


@lru_cache(maxsize=1)
def _mel_filters():
    return librosa.filters.mel(sr=config.SR, n_fft=config.N_FFT, n_mels=config.N_MELS,
                               fmin=config.FMIN, fmax=config.FMAX, htk=False, norm="slaney", dtype=np.float32)


def _shard_paths(filename: str, cache_root: Path):
    key = sha_key(filename)
    directory = cache_root / key
    return directory, directory / "mels.npy", directory / "metadata.json"


def _read_complete_shard(directory: Path, filename: str, digest: str, signature: str):
    npy_path, meta_path = directory / "mels.npy", directory / "metadata.json"
    if not npy_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(meta["segments"]), config.N_MELS, 622)
        valid = not (meta.get("filename") != filename or meta.get("source_sha256") != digest or
                     meta.get("config_hash") != signature or arr.shape != expected_shape or
                     arr.dtype != np.float16 or len(meta.get("valid_frames", [])) != expected_shape[0] or
                     not np.isfinite(arr).all())
        if isinstance(arr, np.memmap) and arr._mmap is not None:
            arr._mmap.close()
        if not valid:
            return None
        return meta
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _promote_complete_shard(source: Path, destination: Path, filename: str, digest: str, signature: str):
    meta = _read_complete_shard(source, filename, digest, signature)
    if meta is None:
        return None
    last_error = None
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return meta
        except PermissionError as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(0.1 * (2 ** attempt))
    raise PermissionError(f"Could not atomically promote complete Mel shard for {filename} after 5 attempts; preserved at {source}") from last_error


def extract_recording_mels(record: pd.DataFrame, cache_root: Path) -> dict:
    row = record.iloc[0]
    filename = str(row.filename)
    cache_root.mkdir(parents=True, exist_ok=True)
    audio_path = config.AUDIO_ROOT.joinpath(*filename.split("/"))
    if audio_path.stat().st_size != int(row.expected_size_bytes):
        raise RuntimeError(f"Source size changed: {filename}")
    digest = sha256_file(audio_path)
    if digest != str(row.sha256):
        raise RuntimeError(f"Source SHA-256 changed: {filename}")
    shard_dir, npy_path, meta_path = _shard_paths(filename, cache_root)
    signature = mel_config_hash()
    if npy_path.exists() or meta_path.exists():
        if not npy_path.exists() or not meta_path.exists():
            raise RuntimeError(f"Incomplete existing Mel shard preserved for diagnosis: {filename}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_sha256") != digest or meta.get("config_hash") != signature:
            raise RuntimeError(f"Existing Mel shard metadata mismatch; refusing overwrite: {filename}")
        arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        valid_shard = (arr.shape == (len(record), config.N_MELS, 622) and arr.dtype == np.float16 and
                       len(meta.get("valid_frames", [])) == len(record) and np.isfinite(arr).all())
        if isinstance(arr, np.memmap) and arr._mmap is not None:
            arr._mmap.close()
        if not valid_shard:
            raise RuntimeError(f"Existing Mel shard has wrong shape/dtype: {filename}")
        return meta
    # Resume a complete orphaned temporary shard from a prior interrupted run.
    # Invalid or partial temp folders are preserved; extraction uses a fresh unique name.
    for orphan in sorted(cache_root.glob(f"{sha_key(filename)}.tmp-*")):
        resumed = _promote_complete_shard(orphan, shard_dir, filename, digest, signature)
        if resumed is not None:
            return resumed
    mels = np.empty((len(record), config.N_MELS, 622), dtype=np.float16)
    valid_frame_counts = []
    with sf.SoundFile(str(audio_path), mode="r") as source:
        if source.samplerate != config.SR or source.channels != 1 or len(source) != int(row.sample_count):
            raise RuntimeError(f"Source properties changed during Mel extraction: {filename}")
        for idx, segment in enumerate(record.itertuples(index=False)):
            block = source.read(config.SEGMENT_SAMPLES, dtype="float32", always_2d=True)
            if block.size == 0 or not np.isfinite(block).all():
                raise RuntimeError(f"Empty/non-finite audio segment: {filename} #{idx}")
            mono = block[:, 0]
            valid_samples = len(mono)
            if valid_samples < config.SEGMENT_SAMPLES:
                mono = np.pad(mono, (0, config.SEGMENT_SAMPLES - valid_samples))
            spec, valid_frames = compute_mel_window(mono, valid_samples)
            mels[idx] = spec.astype(np.float16)
            valid_frame_counts.append(valid_frames)
            if valid_samples != int(segment.source_sample_count):
                raise RuntimeError(f"Segment sample count mismatch: {filename} #{idx}")
    # Keep data and identity metadata inside a temp directory. Promotion is one
    # same-volume rename, so a crash cannot leave a half-written final shard.
    temp_dir = cache_root / f"{sha_key(filename)}.tmp-{os.getpid()}-{time.time_ns()}"
    temp_dir.mkdir(parents=True)
    temp_npy = temp_dir / "mels.npy"
    with temp_npy.open("wb") as stream:
        np.save(stream, mels, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    meta = {"filename": filename, "source_sha256": digest, "config_hash": signature,
            "segments": len(record), "valid_frames": valid_frame_counts,
            "dtype": "float16", "shape": list(mels.shape),
            "partition": str(row.m4_partition), "primary_label": str(row.primary_label)}
    # This file is inside a private temp directory that is promoted as a whole;
    # write it directly rather than creating an additional nested temp sibling.
    with (temp_dir / "metadata.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(meta, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _promote_complete_shard(temp_dir, shard_dir, filename, digest, signature)
    return meta


def extract_all_mels(segment_manifest: pd.DataFrame, progress=None, cache_root=config.M4_DATA_ROOT / "mels") -> pd.DataFrame:
    cache_root = Path(cache_root)
    records = segment_manifest.groupby("filename", sort=False)
    summaries = []
    total = records.ngroups
    for index, (_, rows) in enumerate(records, start=1):
        rows = rows.sort_values("segment_index", kind="stable").reset_index(drop=True)
        meta = extract_recording_mels(rows, cache_root)
        summaries.append({"filename": meta["filename"], "source_sha256": meta["source_sha256"],
                          "config_hash": meta["config_hash"], "segments": meta["segments"],
                          "m4_partition": meta["partition"], "primary_label": meta["primary_label"],
                          "mel_path": str(_shard_paths(meta["filename"], cache_root)[0].relative_to(config.ROOT))})
        if progress and (index % 25 == 0 or index == total):
            progress(index, total)
    manifest = pd.DataFrame(summaries)
    if len(manifest) != config.EXPECTED_RECORDINGS or manifest.filename.duplicated().any():
        raise RuntimeError("Mel extraction did not produce one unique shard per recording")
    atomic_write_csv(manifest, config.M4_DATA_ROOT / "mel_recording_manifest.csv")
    atomic_write_json(mel_config() | {"config_hash": mel_config_hash(), "recording_count": total,
                                      "segment_count": int(len(segment_manifest)), "storage_dtype": "float16"},
                      config.M4_DATA_ROOT / "mel_config.json")
    return manifest


def fit_train_normalizer(segment_manifest: pd.DataFrame, cache_root=config.M4_DATA_ROOT / "mels",
                         output=config.M4_DATA_ROOT / "mel_normalizer.json") -> dict:
    train_segments = segment_manifest[segment_manifest.m4_partition.eq("internal_train")]
    sums = np.zeros(config.N_MELS, dtype=np.float64)
    sumsquares = np.zeros(config.N_MELS, dtype=np.float64)
    frame_count = 0
    for filename, group in train_segments.groupby("filename", sort=False):
        _, npy_path, meta_path = _shard_paths(str(filename), Path(cache_root))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["partition"] != "internal_train":
            raise RuntimeError(f"Normalization attempted on non-training recording {filename}")
        arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        for row in group.itertuples(index=False):
            values = arr[int(row.segment_index), :, :int(row.valid_frames)]
            x = np.asarray(values, dtype=np.float64)
            sums += x.sum(axis=1)
            sumsquares += np.square(x).sum(axis=1)
            frame_count += x.shape[1]
    if frame_count <= 0:
        raise RuntimeError("No internal-training valid frames for normalization")
    mean = sums / frame_count
    var = np.maximum(sumsquares / frame_count - mean ** 2, 1e-12)
    std = np.sqrt(var)
    result = {"method": "per-Mel-band mean/std z-score; invalid padded frames set to zero",
              "fit_partition": "internal_train", "fit_recordings": int(train_segments.filename.nunique()),
              "fit_segments": int(len(train_segments)), "valid_frames": frame_count,
              "mean": mean.tolist(), "std": std.tolist(), "config_hash": mel_config_hash()}
    atomic_write_json(result, output)
    return result


def make_segment_index(segment_manifest: pd.DataFrame, cache_root=config.M4_DATA_ROOT / "mels",
                       output=config.M4_DATA_ROOT / "mel_segment_index.csv") -> pd.DataFrame:
    result = segment_manifest.copy()
    result["valid_frames"] = -1
    for filename, indices in result.groupby("filename", sort=False).groups.items():
        _, _, meta_path = _shard_paths(str(filename), Path(cache_root))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(meta["valid_frames"]) != len(indices):
            raise RuntimeError(f"Mel frame mask count mismatch for {filename}")
        ordered = result.loc[indices].sort_values("segment_index", kind="stable")
        for row_index, valid_frames in zip(ordered.index, meta["valid_frames"]):
            result.loc[row_index, "valid_frames"] = int(valid_frames)
    if result.valid_frames.le(0).any():
        raise RuntimeError("One or more segments has no valid mel frames")
    atomic_write_csv(result, output)
    return result
