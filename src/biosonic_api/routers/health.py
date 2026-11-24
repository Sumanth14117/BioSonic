from fastapi import APIRouter, Depends

from biosonic_api.dependencies import get_model_registry
from biosonic_api.schemas import APIReadinessResponse, ModelInfo, ModelsResponse, ProcessHealthResponse, RootResponse
from biosonic_api.services.registry import ModelRegistry

router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["system"])
def root():
    return RootResponse(name="BioSonic Inference API", version="0.1.0", docs="/docs")


@router.get("/health", response_model=ProcessHealthResponse, tags=["system"])
def process_health():
    return ProcessHealthResponse(status="ok")


@router.get("/api/v1/health", response_model=APIReadinessResponse, tags=["system"])
def api_health(registry: ModelRegistry = Depends(get_model_registry)):
    return APIReadinessResponse(**registry.readiness())


@router.get("/api/v1/models", response_model=ModelsResponse, tags=["models"])
def models(registry: ModelRegistry = Depends(get_model_registry)):
    return ModelsResponse(models=[ModelInfo(**item) for item in registry.list_models()])
