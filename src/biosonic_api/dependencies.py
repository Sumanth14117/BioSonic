"""FastAPI request dependencies."""

from fastapi import Request

from biosonic_api.services.registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry
