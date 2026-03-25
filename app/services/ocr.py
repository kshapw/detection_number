"""PaddleOCR singleton wrapper.

Loaded once at application startup via lifespan; shared across all requests.
Thread-safe for concurrent read access once initialised.
"""

import threading
from typing import List, Optional
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
        logger.info("ocr.ready")

    def run(self, image: np.ndarray) -> str:
        """Run OCR on a preprocessed RGB numpy array.

        Returns:
            full_text: all recognised text joined by newlines
        """
        with self._lock:
            result = self._engine.ocr(image, cls=True)

        lines: List[str] = []

        if not result or result[0] is None:
            return ""

        for page in result:
            if page is None:
                continue
            for item in page:
                # item: [bbox, (text, confidence)]
                try:
                    text = item[1][0]
                except (IndexError, TypeError, ValueError):
                    continue
                lines.append(text.strip())

        full_text = "\n".join(lines)
        logger.debug("ocr.run.complete", lines=len(lines), text_preview=full_text[:120])
        return full_text
