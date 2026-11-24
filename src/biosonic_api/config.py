"""Environment-based API settings; paths resolve relative to the repository."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    app_name: str = "BioSonic Inference API"
    api_version: str = "0.1.0"
    max_upload_bytes: int = 25 * 1024 * 1024
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 180.0
    sample_rate: int = 32_000
    channels: int = 1
    checkpoint_path: Path = ROOT / "models/m4_custom_cnn/best_model.pt"
    request_body_overhead_bytes: int = 256 * 1024
    device: str = "auto"
    auth_database_path: Path = ROOT / "data/auth/biosonic_auth.sqlite3"
    auth_session_ttl_seconds: int = 12 * 60 * 60
    auth_cookie_secure: bool = False
    auth_cookie_name: str = "biosonic_session"
    auth_allowed_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")

    @classmethod
    def from_env(cls) -> "Settings":
        device = os.getenv("BIOSONIC_DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("BIOSONIC_DEVICE must be auto, cpu, or cuda")
        checkpoint = Path(os.getenv("BIOSONIC_M4_CHECKPOINT", str(ROOT / "models/m4_custom_cnn/best_model.pt"))).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        min_duration = float(os.getenv("BIOSONIC_MIN_DURATION_SECONDS", "1.0"))
        max_duration = float(os.getenv("BIOSONIC_MAX_DURATION_SECONDS", "180.0"))
        if min_duration <= 0 or max_duration < min_duration:
            raise ValueError("Audio duration settings must be positive and max must be >= min")
        database_path = Path(os.getenv("BIOSONIC_AUTH_DATABASE", str(ROOT / "data/auth/biosonic_auth.sqlite3"))).expanduser()
        if not database_path.is_absolute():
            database_path = ROOT / database_path
        allowed_origins = tuple(
            item.strip().rstrip("/")
            for item in os.getenv("BIOSONIC_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
            if item.strip()
        )
        secure_value = os.getenv("BIOSONIC_AUTH_COOKIE_SECURE", "false").strip().lower()
        if secure_value not in {"true", "false", "1", "0"}:
            raise ValueError("BIOSONIC_AUTH_COOKIE_SECURE must be true or false")
        return cls(
            max_upload_bytes=_positive_int("BIOSONIC_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            checkpoint_path=checkpoint,
            device=device,
            auth_database_path=database_path,
            auth_cookie_secure=secure_value in {"true", "1"},
            auth_allowed_origins=allowed_origins,
        )
