"""Per-application model lifecycle and public capability descriptions."""

from __future__ import annotations

import logging

from biosonic_cnn.config import INDEX_TO_LABEL
from biosonic_api.config import Settings
from biosonic_api.services.m4_model import M4ModelService

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings, *, m4_service: M4ModelService | None = None):
        self.settings = settings
        self.m4 = m4_service or M4ModelService(settings)
        self._started = False

    def startup(self) -> None:
        if self._started:
            return
        self.m4.load()
        self._started = True
        logger.info("api_models_initialized m4_ready=%s", self.m4.ready)

    def shutdown(self) -> None:
        self.m4.close()
        self._started = False

    def readiness(self) -> dict[str, object]:
        return {
            "status": "ready" if self.m4.ready else "degraded",
            "models": {"biosonic-m4-cnn": "ready" if self.m4.ready else "unavailable"},
        }

    def list_models(self) -> list[dict[str, object]]:
        return [
            {
                "id": "biosonic-m4-cnn",
                "task": "recording-level eight-class bird classification",
                "status": "ready" if self.m4.ready else "unavailable",
                "version": self.m4.version,
                "classes": list(INDEX_TO_LABEL),
                "limitations": [
                    "Predicts only among its eight trained classes; it has no unknown-species rejection capability.",
                    "Interim BirdCLEF+ 2026 experiment; not validated as a general-purpose bird detector.",
                    "Input must be mono 32 kHz OGG/Vorbis; the API does not resample or downmix.",
                ],
                "device": self.m4.device_name if self.m4.ready else None,
            },
            {
                "id": "biosonic-m7-soundscape",
                "task": "experimental multi-label five-second soundscape detection",
                "status": "disabled",
                "version": None,
                "classes": [],
                "limitations": [
                    "M7 evaluation was exploratory and did not establish useful detection performance.",
                    "YAMNet extraction requires the separate TensorFlow CPU environment; no runtime bridge is configured.",
                ],
                "device": None,
            },
        ]
