"""FastAPI application factory and lifecycle."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError as exc:  # make missing local setup explicit without a partial API
    raise ImportError("BioSonic API dependencies are missing; see requirements-api.txt. No packages were installed automatically.") from exc

from biosonic_api import __version__
from biosonic_api.auth_store import AuthStore
from biosonic_api.config import Settings
from biosonic_api.middleware import RequestBodyLimitMiddleware
from biosonic_api.routers.auth import router as auth_router
from biosonic_api.routers.health import router as health_router
from biosonic_api.routers.history import router as history_router
from biosonic_api.routers.predict import router as predict_router
from biosonic_api.services.registry import ModelRegistry


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    if not any(isinstance(existing, logging.StreamHandler) and isinstance(existing.formatter, JsonLogFormatter) for existing in root.handlers):
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    model_registry = registry or ModelRegistry(settings)
    auth_store = AuthStore(settings.auth_database_path, settings.auth_session_ttl_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        auth_store.initialize()
        model_registry.startup()
        app.state.model_registry = model_registry
        yield
        model_registry.shutdown()

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description="Local inference API for BioSonic's interim eight-class M4 classifier.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.auth_store = auth_store
    app.state.model_registry = model_registry
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_upload_bytes + settings.request_body_overhead_bytes)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(history_router)
    app.include_router(predict_router)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            logging.getLogger("biosonic_api.request").info(
                "request_complete method=%s path=%s status=%d elapsed_ms=%.2f",
                request.method, request.url.path, status, (time.perf_counter() - started) * 1000,
            )

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": {"code": "invalid_request", "message": "The request fields are invalid."}})

    return app


app = create_app()
