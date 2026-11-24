"""Public Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str


class ProcessHealthResponse(BaseModel):
    status: str


class APIReadinessResponse(BaseModel):
    status: str
    models: dict[str, str]


class ModelInfo(BaseModel):
    id: str
    task: str
    status: str
    version: str | None = None
    classes: list[str]
    limitations: list[str]
    device: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class AudioMetadata(BaseModel):
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    format: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    model_version: str
    prediction_id: str
    prediction: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    audio: AudioMetadata


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class AuthUser(BaseModel):
    id: str
    email: str


class CsrfResponse(BaseModel):
    csrf_token: str
    user: AuthUser | None = None


class AuthCredentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(AuthCredentials):
    password: str = Field(min_length=12, max_length=128)


class EmailUpdateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class PasswordUpdateRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)
