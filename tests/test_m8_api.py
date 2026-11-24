"""M8 validation/model unit tests; HTTP tests run when optional API deps exist."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_api.audio.validation import cleanup_temp_file, stage_upload, validate_audio_file, validate_client_filename
from biosonic_api.config import Settings
from biosonic_api.exceptions import AudioUploadError, InferenceError
from biosonic_api.services.m4_model import M4ModelService
from biosonic_api.services.registry import ModelRegistry


def write_ogg(path: Path, *, sample_rate: int = 32_000, channels: int = 1, seconds: float = 5.0) -> None:
    samples = int(sample_rate * seconds)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        signal = np.repeat(signal[:, None], channels, axis=1)
    sf.write(path, signal, sample_rate, format="OGG", subtype="VORBIS")


def default_settings(**changes) -> Settings:
    values = {
        "max_upload_bytes": 1024 * 1024,
        "min_duration_seconds": 1.0,
        "max_duration_seconds": 180.0,
        "checkpoint_path": ROOT / "models/m4_custom_cnn/best_model.pt",
        "auth_database_path": Path(tempfile.gettempdir()) / "biosonic-test-unused.sqlite3",
    }
    values.update(changes)
    return Settings(**values)


class FakeModel(torch.nn.Module):
    def __init__(self, output=None):
        super().__init__()
        self.output = torch.tensor(output if output is not None else [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.input_shapes = []

    def forward(self, values):
        self.input_shapes.append(tuple(values.shape))
        return self.output.to(values.device).reshape(1, -1)


def ready_service(model=None):
    normalizer = {
        "fit_partition": "internal_train",
        "mean": [0.0] * 128,
        "std": [1.0] * 128,
    }
    return M4ModelService(default_settings(), model=model or FakeModel(), normalizer=normalizer)


def test_supported_filename_is_validated_and_never_used_as_a_path():
    assert validate_client_filename("soundscape.ogg") == "soundscape.ogg"
    for invalid in ("../soundscape.ogg", r"..\soundscape.ogg", "C:soundscape.ogg", "bad.wav", "", None):
        with pytest.raises(AudioUploadError):
            validate_client_filename(invalid)


def test_valid_ogg_upload_stages_decodes_and_cleans_up():
    with tempfile.TemporaryDirectory() as temporary:
        source_path = Path(temporary) / "source.ogg"
        write_ogg(source_path)
        settings = default_settings()
        temp_path, size = stage_upload(io.BytesIO(source_path.read_bytes()), "recording.ogg", settings)
        try:
            audio = validate_audio_file(temp_path, settings)
            assert size == temp_path.stat().st_size
            assert audio.sample_rate == 32_000 and audio.channels == 1
            assert audio.sample_count == 160_000
            assert audio.duration_seconds == 5.0
            assert audio.audio_format == "OGG/VORBIS"
            assert np.isfinite(audio.waveform).all()
        finally:
            cleanup_temp_file(temp_path)
        assert not temp_path.exists()


def test_upload_size_limit_removes_partial_temp_file():
    with tempfile.TemporaryDirectory() as temporary:
        with patch.object(tempfile, "tempdir", temporary):
            with pytest.raises(AudioUploadError) as error:
                stage_upload(io.BytesIO(b"x" * 100), "recording.ogg", default_settings(max_upload_bytes=10))
        assert error.value.status_code == 413
        assert list(Path(temporary).glob("biosonic-upload-*.ogg")) == []


def test_empty_corrupt_wrong_rate_wrong_channels_and_short_audio_fail():
    settings = default_settings()
    with pytest.raises(AudioUploadError, match="empty"):
        stage_upload(io.BytesIO(b""), "recording.ogg", settings)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corrupt = root / "corrupt.ogg"
        corrupt.write_bytes(b"this is not an audio stream")
        with pytest.raises(AudioUploadError):
            validate_audio_file(corrupt, settings)
        for name, rate, channels, duration, expected_code in (
            ("rate.ogg", 22_050, 1, 2.0, "unsupported_sample_rate"),
            ("stereo.ogg", 32_000, 2, 2.0, "unsupported_channel_count"),
            ("short.ogg", 32_000, 1, 0.5, "unsupported_duration"),
        ):
            path = root / name
            write_ogg(path, sample_rate=rate, channels=channels, seconds=duration)
            with pytest.raises(AudioUploadError) as error:
                validate_audio_file(path, settings)
            assert error.value.code == expected_code


def test_decoded_nonfinite_waveform_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "valid.ogg"
        write_ogg(path)
        bad = np.zeros((160_000, 1), dtype=np.float32)
        bad[10, 0] = np.nan
        with patch("biosonic_api.audio.validation.sf.read", return_value=(bad, 32_000)):
            with pytest.raises(AudioUploadError) as error:
                validate_audio_file(path, default_settings())
        assert error.value.code == "non_finite_audio"


def test_m4_preprocessing_and_probabilities_match_recording_contract():
    service = ready_service()
    waveform = np.zeros(6 * 32_000 + 123, dtype=np.float32)
    result = service.predict(waveform, 32_000)
    assert result["model"] == "biosonic-m4-cnn"
    assert result["prediction"] == "burowl"
    probabilities = result["probabilities"]
    assert list(probabilities) == ["compot1", "rufhor2", "saytan1", "crbthr1", "rutjac1", "rufnig1", "chobla1", "burowl"]
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert result["segment_count"] == 2
    assert service._model.input_shapes == [(1, 1, 128, 622), (1, 1, 128, 622)]
    assert result["valid_mel_frames_per_segment"][1] < 622


def test_invalid_logits_fail_without_returning_unvalidated_probabilities():
    service = ready_service(FakeModel(output=[float("nan")] * 8))
    with pytest.raises(InferenceError):
        service.predict(np.zeros(32_000, dtype=np.float32), 32_000)


def test_model_is_unavailable_gracefully_and_registry_marks_m7_experimental_disabled():
    with tempfile.TemporaryDirectory() as temporary:
        service = M4ModelService(default_settings(checkpoint_path=Path(temporary) / "missing.pt"))
        registry = ModelRegistry(service.settings, m4_service=service)
        registry.startup()
        assert not service.ready
        assert service.readiness_error == "FileNotFoundError"
        assert str(Path(temporary)) not in str(registry.list_models())
        m7 = next(item for item in registry.list_models() if item["id"] == "biosonic-m7-soundscape")
        assert m7["status"] == "disabled"
        assert "exploratory" in " ".join(m7["limitations"])
        registry.shutdown()


if importlib.util.find_spec("fastapi") and importlib.util.find_spec("multipart"):
    from fastapi.testclient import TestClient
    from biosonic_api.main import create_app

    class ReadyFakeService(M4ModelService):
        def __init__(self, settings):
            super().__init__(settings, model=FakeModel(), normalizer={"fit_partition": "internal_train", "mean": [0.0] * 128, "std": [1.0] * 128})

    def test_http_health_models_and_prediction_contract():
        with tempfile.TemporaryDirectory() as temporary:
            settings = default_settings(auth_database_path=Path(temporary) / "auth.sqlite3")
            service = ReadyFakeService(settings)
            registry = ModelRegistry(settings, m4_service=service)
            app = create_app(settings, registry)
            path = Path(temporary) / "upload.ogg"
            write_ogg(path)
            with TestClient(app) as client:
                assert client.get("/health").json() == {"status": "ok"}
                assert client.get("/api/v1/health").json()["status"] == "ready"
                assert len(client.get("/api/v1/models").json()["models"]) == 2
                csrf = client.get("/api/v1/auth/csrf").json()["csrf_token"]
                registered = client.post("/api/v1/auth/register", json={"email": "bird@example.org", "password": "strong password 42"}, headers={"Origin": "http://127.0.0.1:5173", "X-CSRF-Token": csrf})
                assert registered.status_code == 201
                csrf = registered.json()["csrf_token"]
                response = client.post("/api/v1/predict", files={"file": ("bird.ogg", path.read_bytes(), "audio/ogg")}, headers={"Origin": "http://127.0.0.1:5173", "X-CSRF-Token": csrf})
                assert response.status_code == 200
                body = response.json()
                assert body["prediction"] == "burowl"
                assert body["prediction_id"]
                assert len(body["probabilities"]) == 8
                history = client.get("/api/v1/history").json()["items"]
                assert len(history) == 1 and history[0]["id"] == body["prediction_id"]
                assert "audio_bytes" not in history[0]
                assert "C:\\BioSonic" not in response.text
                assert "temporary" not in response.text.lower()

    def test_http_filename_traversal_is_rejected():
        with tempfile.TemporaryDirectory() as temporary:
            settings = default_settings(auth_database_path=Path(temporary) / "auth.sqlite3")
            app = create_app(settings, ModelRegistry(settings, m4_service=ReadyFakeService(settings)))
            with TestClient(app) as client:
                csrf = client.get("/api/v1/auth/csrf").json()["csrf_token"]
                client.post("/api/v1/auth/register", json={"email": "bird@example.org", "password": "strong password 42"}, headers={"Origin": "http://127.0.0.1:5173", "X-CSRF-Token": csrf})
                csrf = client.get("/api/v1/auth/csrf").json()["csrf_token"]
                response = client.post("/api/v1/predict", files={"file": ("../escape.ogg", b"x", "audio/ogg")}, headers={"Origin": "http://127.0.0.1:5173", "X-CSRF-Token": csrf})
                assert response.status_code == 400
                assert "C:\\BioSonic" not in response.text
else:
    @pytest.mark.skip(reason="FastAPI, Starlette/Pydantic, and python-multipart are not installed in the existing .venv; no packages were installed.")
    def test_http_health_models_and_prediction_contract():
        pass

    @pytest.mark.skip(reason="FastAPI and python-multipart are not installed in the existing .venv; no packages were installed.")
    def test_http_filename_traversal_is_rejected():
        pass
