"""Lifecycle-managed inference wrapper for the existing M4 CNN checkpoint."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from biosonic_cnn import config as m4_config
from biosonic_cnn.mels import mel_config_hash
from biosonic_cnn.model import BioSonicCNN, architecture_config
from biosonic_api.audio.preprocessing import build_m4_inputs
from biosonic_api.config import Settings
from biosonic_api.exceptions import InferenceError, ModelUnavailableError

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class M4ModelService:
    """Loads the M4 checkpoint once and performs recording-level aggregation."""

    def __init__(self, settings: Settings, *, model: Any | None = None, normalizer: dict | None = None):
        self.settings = settings
        self._model = model
        self._normalizer = normalizer
        self._device: torch.device | None = None
        self._model_sha256: str | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._ready = model is not None and normalizer is not None
        if self._ready:
            self._device = torch.device("cpu")
            self._model_sha256 = "injected-test-model"

    def load(self) -> bool:
        if self._ready:
            return True
        if self._load_error:
            return False
        with self._lock:
            if self._ready:
                return True
            try:
                path = self.settings.checkpoint_path
                if not path.is_file():
                    raise FileNotFoundError("M4 checkpoint is unavailable")
                checkpoint_hash = _sha256(path)
                # weights_only=True avoids unrestricted pickle globals.
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or not {"model_state", "architecture", "normalizer"}.issubset(checkpoint):
                    raise ValueError("M4 checkpoint schema is incompatible")
                expected_architecture = architecture_config()
                recorded = checkpoint["architecture"]
                if recorded != expected_architecture:
                    raise ValueError("M4 checkpoint architecture does not match current code")
                if tuple(recorded.get("class_order", ())) != tuple(m4_config.INDEX_TO_LABEL):
                    raise ValueError("M4 checkpoint class order is incompatible")
                if recorded.get("input_shape") != [1, 128, 622] or recorded.get("num_classes") != 8:
                    raise ValueError("M4 checkpoint input or output dimensions are incompatible")
                normalizer = checkpoint["normalizer"]
                if normalizer.get("fit_partition") != "internal_train" or normalizer.get("config_hash") != mel_config_hash():
                    raise ValueError("M4 checkpoint normalizer does not match the training feature configuration")
                mean = np.asarray(normalizer.get("mean"), dtype=np.float32)
                std = np.asarray(normalizer.get("std"), dtype=np.float32)
                if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                    raise ValueError("M4 checkpoint normalizer dimensions or values are invalid")

                device_setting = self.settings.device
                if device_setting == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is unavailable")
                device = torch.device("cuda" if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available()) else "cpu")
                model = BioSonicCNN(num_classes=len(m4_config.INDEX_TO_LABEL))
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.to(device)
                model.eval()
                with torch.inference_mode():
                    smoke = model(torch.zeros((1, 1, 128, 622), dtype=torch.float32, device=device))
                if smoke.shape != (1, 8) or not torch.isfinite(smoke).all():
                    raise ValueError("M4 checkpoint failed output compatibility validation")
                self._model = model
                self._normalizer = normalizer
                self._device = device
                self._model_sha256 = checkpoint_hash
                self._ready = True
                self._load_error = None
                logger.info("model_loaded model=biosonic-m4-cnn version=%s device=%s classes=%d", checkpoint_hash[:12], device.type, 8)
                return True
            except Exception as exc:
                self._load_error = type(exc).__name__
                logger.error("model_load_failed model=biosonic-m4-cnn error_type=%s", type(exc).__name__)
                return False

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    @property
    def readiness_error(self) -> str | None:
        return self._load_error

    @property
    def version(self) -> str | None:
        return self._model_sha256[:12] if self._model_sha256 else None

    @property
    def device_name(self) -> str | None:
        if not self._device:
            return None
        if self._device.type == "cuda":
            return torch.cuda.get_device_name(self._device)
        return "CPU"

    def predict(self, waveform: np.ndarray, sample_rate: int) -> dict[str, object]:
        if not self._ready and not self.load():
            raise ModelUnavailableError("The primary M4 model is unavailable.")
        assert self._model is not None and self._normalizer is not None and self._device is not None
        try:
            inputs, valid_frames = build_m4_inputs(waveform, sample_rate, self._normalizer)
            segment_probabilities: list[np.ndarray] = []
            with self._lock, torch.inference_mode():
                for segment in inputs:
                    logits = self._model(segment.unsqueeze(0).to(self._device))
                    if logits.shape != (1, len(m4_config.INDEX_TO_LABEL)) or not torch.isfinite(logits).all():
                        raise InferenceError("Model returned invalid logits")
                    probs = torch.softmax(logits.float(), dim=1)[0].detach().to("cpu").numpy().astype(np.float64)
                    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
                        raise InferenceError("Model returned invalid probabilities")
                    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
                        raise InferenceError("Model probabilities are not normalized")
                    segment_probabilities.append(probs)
            if not segment_probabilities:
                raise InferenceError("No segment predictions were produced")
            # M4 evaluation averages per-window softmax scores at recording level.
            aggregate = np.mean(np.vstack(segment_probabilities), axis=0)
            if not np.isfinite(aggregate).all() or not np.isclose(aggregate.sum(), 1.0, atol=1e-6):
                raise InferenceError("Aggregated probabilities are invalid")
            best = int(np.argmax(aggregate))
            return {
                "model": "biosonic-m4-cnn",
                "model_version": self.version,
                "prediction": m4_config.INDEX_TO_LABEL[best],
                "confidence": float(aggregate[best]),
                "probabilities": {label: float(aggregate[i]) for i, label in enumerate(m4_config.INDEX_TO_LABEL)},
                "segment_count": len(segment_probabilities),
                "valid_mel_frames_per_segment": valid_frames,
            }
        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.error("inference_failed model=biosonic-m4-cnn error_type=%s", type(exc).__name__)
            if isinstance(exc, InferenceError):
                raise
            raise InferenceError("Audio inference failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._normalizer = None
            self._ready = False
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
            self._device = None
