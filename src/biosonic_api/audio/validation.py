"""Safe upload staging and decoding checks for the M4 inference contract."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

from biosonic_api.config import Settings
from biosonic_api.exceptions import AudioUploadError


@dataclass(frozen=True)
class ValidatedAudio:
    waveform: np.ndarray
    sample_rate: int
    channels: int
    sample_count: int
    duration_seconds: float
    audio_format: str
    subtype: str


def validate_client_filename(filename: str | None) -> str:
    """Accept only a safe basename with a supported extension; never use it as a path."""
    if not filename or "\x00" in filename or "/" in filename or "\\" in filename or ":" in filename:
        raise AudioUploadError(400, "invalid_filename", "Provide an audio filename without path separators.")
    if filename in {".", ".."} or ".." in filename:
        raise AudioUploadError(400, "invalid_filename", "The uploaded filename is invalid.")
    if Path(filename).suffix.lower() != ".ogg":
        raise AudioUploadError(415, "unsupported_extension", "Only .ogg audio uploads are supported.")
    return filename


def stage_upload(source: BinaryIO, filename: str | None, settings: Settings) -> tuple[Path, int]:
    """Copy bounded upload bytes to an unpredictable temporary path."""
    validate_client_filename(filename)
    temp_path: Path | None = None
    total = 0
    try:
        handle = tempfile.NamedTemporaryFile(prefix="biosonic-upload-", suffix=".ogg", delete=False)
        temp_path = Path(handle.name)
        with handle:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise AudioUploadError(413, "upload_too_large", "The audio file exceeds the configured upload limit.")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if total == 0:
            raise AudioUploadError(400, "empty_upload", "The uploaded audio file is empty.")
        return temp_path, total
    except AudioUploadError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise AudioUploadError(400, "upload_read_failed", "The uploaded file could not be read.") from exc


def validate_audio_file(path: Path, settings: Settings) -> ValidatedAudio:
    """Validate file contents and decode the complete mono 32 kHz waveform."""
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise AudioUploadError(422, "invalid_audio", "The upload is not a decodable audio file.") from exc

    if info.format != "OGG" or info.subtype != "VORBIS":
        raise AudioUploadError(415, "unsupported_audio_format", "The file must contain OGG/Vorbis audio.")
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise AudioUploadError(422, "invalid_audio_metadata", "The audio file has invalid or empty audio metadata.")
    if info.samplerate != settings.sample_rate:
        raise AudioUploadError(422, "unsupported_sample_rate", "Audio must use a 32 kHz sample rate; resampling is not applied by this API.")
    if info.channels != settings.channels:
        raise AudioUploadError(422, "unsupported_channel_count", "Audio must be mono; channel conversion is not applied by this API.")
    duration = info.frames / info.samplerate
    if not settings.min_duration_seconds <= duration <= settings.max_duration_seconds:
        raise AudioUploadError(422, "unsupported_duration", "Audio duration is outside the configured accepted range.")
    try:
        waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioUploadError(422, "audio_decode_failed", "The complete audio stream could not be decoded.") from exc
    if rate != settings.sample_rate or waveform.shape != (info.frames, 1):
        raise AudioUploadError(422, "audio_shape_mismatch", "Decoded audio dimensions do not match the file metadata.")
    if not np.isfinite(waveform).all():
        raise AudioUploadError(422, "non_finite_audio", "Audio contains invalid non-finite sample values.")
    return ValidatedAudio(
        waveform=waveform[:, 0],
        sample_rate=int(rate),
        channels=1,
        sample_count=int(info.frames),
        duration_seconds=float(duration),
        audio_format=f"{info.format}/{info.subtype}",
        subtype=str(info.subtype),
    )


def cleanup_temp_file(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)
