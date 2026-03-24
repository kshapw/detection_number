from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.middleware.logging import LoggingMiddleware, configure_structlog
from app.routes.detect import router as detect_router
from app.schemas import HealthResponse, ReadyResponse
from app.services.ocr import OCRService

settings = get_settings()
configure_structlog(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup.begin", use_gpu=settings.use_gpu, lang=settings.ocr_lang)
    app.state.model_loaded = False
    app.state.ocr = OCRService(settings)
    app.state.model_loaded = True
    logger.info("startup.complete", model_loaded=True)
    yield
    logger.info("shutdown")
    app.state.model_loaded = False


def create_app() -> FastAPI:
    app = FastAPI(
        title="Phone Number Detection API",
        description="Offline OCR microservice that extracts phone numbers from uploaded images.",
        version="1.0.0",
        lifespan=lifespan,
        # Disable interactive docs in production; enable via DETECT_ENABLE_DOCS=true
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.include_router(detect_router)

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/health", "/ready", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(request: Request):
        loaded = getattr(request.app.state, "model_loaded", False)
        return ReadyResponse(status="ready" if loaded else "loading", model_loaded=loaded)

    return app


app = create_app()
