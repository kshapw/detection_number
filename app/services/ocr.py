"""PaddleOCR singleton wrapper.

Loaded once at application startup via lifespan; shared across all requests.
Thread-safe for concurrent read access once initialised.
"""

import threading
from typing import Optional

import numpy as np
import structlog
from paddleocr import PaddleOCR

from app.config import Settings

logger = structlog.get_logger(__name__)

# Language map: expand as needed
_LANG_MAP = {
    "en": "en",
    "hi": "hi",       # Hindi (Devanagari)
    "kn": "kn",       # Kannada — use 'en' model but keep Kannada dict
    "multilang": "en", # fall back to English model (handles mixed scripts well)
}


class OCRService:
    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        lang = _LANG_MAP.get(settings.ocr_lang, "en")
        logger.info("ocr.init", use_gpu=settings.use_gpu, lang=lang)
        # use_angle_cls=True rotates text regions before recognition
        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=settings.use_gpu,
            show_log=False,
            det_db_thresh=settings.ocr_det_db_thresh,
            det_db_box_thresh=settings.ocr_det_db_box_thresh,
            # Download models to a fixed location so they survive container restarts
            # when the volume is mounted at /models
            det_model_dir="/models/det",
            rec_model_dir="/models/rec",
            cls_model_dir="/models/cls",
        )
        self._confidence_threshold = settings.ocr_confidence_threshold
        logger.info("ocr.ready")

    def run(self, image: np.ndarray) -> tuple[str, list[float]]:
        """Run OCR on a preprocessed RGB numpy array.

        Returns:
            full_text: all recognised text joined by newlines
            confidences: per-line confidence scores (filtered by threshold)
        """
        with self._lock:
            result = self._engine.ocr(image, cls=True)

        lines: list[str] = []
        confidences: list[float] = []

        if not result or result[0] is None:
            return "", []

        for page in result:
            if page is None:
                continue
            for item in page:
                # item: [bbox, (text, confidence)]
                try:
                    text, conf = item[1]
                except (IndexError, TypeError, ValueError):
                    continue
                if conf < self._confidence_threshold:
                    continue
                lines.append(text.strip())
                confidences.append(round(float(conf), 4))

        full_text = "\n".join(lines)
        logger.debug("ocr.run.complete", lines=len(lines), text_preview=full_text[:120])
        return full_text, confidences
