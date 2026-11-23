"""CPU wrapper around the pinned official Google YAMNet implementation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config
from .io_utils import sha256_file


def _load_official_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pinned official YAMNet module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_official_modules():
    source = config.OFFICIAL_SOURCE_DIR
    for name, expected_hash in config.OFFICIAL_SOURCE_SHA256.items():
        path = source / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Pinned official YAMNet source integrity check failed: {name}")
    params = _load_official_module("biosonic_yamnet_official_params", source / "params.py")
    features = _load_official_module("biosonic_yamnet_official_features", source / "features.py")
    previous = sys.modules.get("features")
    sys.modules["features"] = features
    try:
        yamnet = _load_official_module("biosonic_yamnet_official_yamnet", source / "yamnet.py")
    finally:
        if previous is None:
            sys.modules.pop("features", None)
        else:
            sys.modules["features"] = previous
    return params, yamnet


class YamnetEncoder:
    def __init__(self, weights_path: Path = config.WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Official YAMNet checkpoint is missing: {self.weights_path}")
        if self.weights_path.stat().st_size != config.WEIGHTS_BYTES:
            raise ValueError("Official YAMNet checkpoint size does not match Google Storage metadata")
        if sha256_file(self.weights_path) != config.WEIGHTS_SHA256:
            raise ValueError("Official YAMNet checkpoint SHA-256 mismatch")
        import hashlib

        md5 = hashlib.md5(self.weights_path.read_bytes()).hexdigest()
        if md5 != config.WEIGHTS_MD5:
            raise ValueError("Official YAMNet checkpoint MD5 mismatch")

        self.params_module, self.yamnet_module = load_official_modules()
        self.params = self.params_module.Params()
        with tf.device("/CPU:0"):
            self.model = self.yamnet_module.yamnet_frames_model(self.params)
            self.model.load_weights(str(self.weights_path))
        self.parameter_count = int(self.model.count_params())

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wave = np.asarray(waveform)
        if wave.ndim != 1 or wave.dtype != np.float32:
            raise ValueError("YAMNet input must be a one-dimensional float32 waveform")
        if wave.size == 0 or not np.isfinite(wave).all():
            raise ValueError("YAMNet input must be non-empty and finite")
        with tf.device("/CPU:0"):
            scores, embeddings, log_mel = self.model(tf.convert_to_tensor(wave, dtype=tf.float32))
        scores = np.asarray(scores.numpy(), dtype=np.float32)
        embeddings = np.asarray(embeddings.numpy(), dtype=np.float32)
        log_mel = np.asarray(log_mel.numpy(), dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != 521:
            raise ValueError(f"Unexpected YAMNet class-score shape: {scores.shape}")
        if embeddings.ndim != 2 or embeddings.shape[1] != config.EMBEDDING_DIM:
            raise ValueError(f"Unexpected YAMNet embedding shape: {embeddings.shape}")
        if log_mel.ndim != 2 or log_mel.shape[1] != 64:
            raise ValueError(f"Unexpected YAMNet log-Mel shape: {log_mel.shape}")
        if not (np.isfinite(scores).all() and np.isfinite(embeddings).all() and np.isfinite(log_mel).all()):
            raise ValueError("YAMNet produced non-finite outputs")
        return scores, embeddings, log_mel
