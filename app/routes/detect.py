"""POST /detect endpoint."""

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import magic
import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.schemas import DetectResponse
from app.services.phone_extractor import extract_phone_numbers
from app.services.preprocessor import preprocess
from typing import List, Tuple
logger = structlog.get_logger(__name__)
router = APIRouter(tags=["detection"])
settings = get_settings()

# Bounded executor: one thread per Gunicorn worker cap prevents thread/GPU exhaustion
# under concurrent load. Sized at runtime via WORKERS env var.
import os as _os
_EXECUTOR = ThreadPoolExecutor(max_workers=int(_os.getenv("WORKERS", 2)))

# Strip non-printable/control characters from user-supplied filenames before logging
_SAFE_FILENAME_RE = re.compile(r"[^\x20-\x7E]")


def _sanitise_filename(name: str | None) -> str:
    if not name:
        return "<no filename>"
    return _SAFE_FILENAME_RE.sub("?", name)[:256]


def _run_inference(ocr_service, image_bytes: bytes) -> Tuple[str, List[float]]:
    """CPU/GPU-bound work executed in the bounded thread pool executor."""
    rgb = preprocess(image_bytes)
    return ocr_service.run(rgb)


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Extract phone numbers from an uploaded image",
    responses={
        200: {"description": "Successful extraction (phone_numbers may be empty list)"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Validation error"},
        503: {"description": "Model not yet loaded"},
    },
)
async def detect(request: Request, file: UploadFile) -> DetectResponse:
    t0 = time.perf_counter()

    # Guard: model must be loaded
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready")

    # Read bytes with size cap
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_bytes // (1024*1024)} MB limit",
        )
    if len(data) == 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")

    # MIME validation (inspect actual bytes, not Content-Type header)
    detected_mime = magic.from_buffer(data[:2048], mime=True)
    if detected_mime not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {detected_mime}. Allowed: {settings.allowed_content_types}",
        )

    safe_filename = _sanitise_filename(file.filename)
    logger.info("detect.start", filename=safe_filename, mime=detected_mime, size=len(data))

    # Run blocking OCR in bounded thread pool so event loop stays free.
    # asyncio.get_running_loop() is the correct API in Python 3.10+ async contexts.
    loop = asyncio.get_running_loop()
    try:
        full_text, confidences = await loop.run_in_executor(
            _EXECUTOR, partial(_run_inference, request.app.state.ocr, data)
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("detect.ocr_error", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OCR failed")

    phone_numbers, _ = extract_phone_numbers(full_text, settings.default_phone_region)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "detect.complete",
        phone_count=len(phone_numbers),
        phones=phone_numbers,
        latency_ms=latency_ms,
    )

    return DetectResponse(
        phone_numbers=phone_numbers,
        raw_text=full_text,
        confidence_scores=confidences,
        latency_ms=latency_ms,
    )
