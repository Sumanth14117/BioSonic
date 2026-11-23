"""Single-recording YAMNet load and CPU inference smoke test."""

import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config
from .audio import prepare_waveform
from .io_utils import atomic_json, sha256_file
from .model import YamnetEncoder


def run_smoke_test(sample_path: Path = config.SAMPLE_AUDIO, output_path: Path | None = None) -> dict:
    sample_path = Path(sample_path)
    relative = sample_path.resolve().relative_to(config.AUDIO_ROOT.resolve()).as_posix()
    if relative not in config.M2_SAMPLE_FILES:
        raise ValueError("YAMNet smoke test must use an out-of-scope M2 representative recording")
    before = sha256_file(sample_path)
    waveform, audio_info = prepare_waveform(sample_path)
    encoder = YamnetEncoder()
    start = time.perf_counter()
    output1 = encoder.infer(waveform)
    elapsed1 = time.perf_counter() - start
    start = time.perf_counter()
    output2 = encoder.infer(waveform)
    elapsed2 = time.perf_counter() - start
    if sha256_file(sample_path) != before:
        raise RuntimeError("M2 smoke-test recording changed during inference")
    scores, embeddings, log_mel = output1
    repeats = [np.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(output1, output2)]
    if not all(repeats):
        raise RuntimeError("Repeated CPU inference outputs were not reproducible within tolerance")
    if scores.shape[0] != embeddings.shape[0]:
        raise RuntimeError("YAMNet patch score and embedding counts differ")
    report = {
        "status": "passed", "sample_filename": relative,
        "sample_sha256_before_after_identical": True, "source_sha256": before,
        "audio": audio_info, "waveform_shape": list(waveform.shape),
        "waveform_dtype": str(waveform.dtype), "waveform_finite": bool(np.isfinite(waveform).all()),
        "execution_device": "/CPU:0",
        "tensorflow_gpu_detected": bool(tf.config.list_physical_devices("GPU")),
        "checkpoint_bytes": config.WEIGHTS_BYTES, "checkpoint_sha256": config.WEIGHTS_SHA256,
        "parameter_count": encoder.parameter_count, "scores_shape": list(scores.shape),
        "embedding_shape": list(embeddings.shape), "log_mel_shape": list(log_mel.shape),
        "outputs_finite": bool(all(np.isfinite(x).all() for x in output1)),
        "repeatability_allclose_rtol_1e-5_atol_1e-6": repeats,
        "first_inference_seconds": elapsed1, "repeat_inference_seconds": elapsed2,
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "tf_keras": __import__("importlib.metadata", fromlist=["version"]).version("tf-keras"),
    }
    if output_path:
        atomic_json(report, output_path)
    return report
