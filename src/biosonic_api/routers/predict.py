"""Recording-level M4 inference route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from biosonic_api.auth import get_auth_store, get_authenticated_csrf_user
from biosonic_api.auth_store import AuthStore
from biosonic_api.audio.validation import cleanup_temp_file, stage_upload, validate_audio_file
from biosonic_api.config import Settings
from biosonic_api.dependencies import get_model_registry
from biosonic_api.exceptions import AudioUploadError, InferenceError, ModelUnavailableError
from biosonic_api.schemas import AudioMetadata, ErrorResponse, PredictionResponse
from biosonic_api.services.registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["inference"],
)
def predict(
    file: UploadFile = File(...),
    user: dict[str, str] = Depends(get_authenticated_csrf_user),
    registry: ModelRegistry = Depends(get_model_registry),
    store: AuthStore = Depends(get_auth_store),
):
    settings = registry.settings
    temporary_path = None
    try:
        temporary_path, upload_size = stage_upload(file.file, file.filename, settings)
        audio = validate_audio_file(temporary_path, settings)
        result = registry.m4.predict(audio.waveform, audio.sample_rate)
        response = PredictionResponse(
            **{key: result[key] for key in ("model", "model_version", "prediction", "confidence", "probabilities")},
            prediction_id="pending",
            audio=AudioMetadata(
                duration_seconds=audio.duration_seconds,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                sample_count=audio.sample_count,
                format=audio.audio_format,
            ),
        )
        prediction_id = store.create_prediction(user["id"], file.filename or "recording.ogg", upload_size, response.model_dump(exclude={"prediction_id"}))
        return response.model_copy(update={"prediction_id": prediction_id})
    except AudioUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "The primary classification model is not ready."}) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail={"code": "inference_failed", "message": "Audio inference could not be completed."}) from exc
    except Exception as exc:
        logger.error("prediction_request_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail={"code": "request_failed", "message": "The prediction request could not be completed."}) from exc
    finally:
        cleanup_temp_file(temporary_path)
        try:
            file.file.close()
        except Exception:
            pass
