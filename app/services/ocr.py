"""PaddleOCR singleton wrapper — two-stage pipeline with early exit.

Stage 1 (detection): one neural-net pass over the full image → bounding boxes.
Stage 2 (recognition): process boxes top-to-bottom, stop as soon as a phone
                        number pattern is matched — QR codes, barcodes, and any
                        content below the phone number are never recognised.

This can reduce per-request latency by 40–60 % on CPU for a labour card where
~25 of ~50 detected boxes belong to the QR code at the bottom of the card.
"""

import re
import threading
from typing import List, Optional

import cv2
import numpy as np
import structlog
from paddleocr import PaddleOCR

from app.config import Settings

logger = structlog.get_logger(__name__)

# Language map: expand as needed
_LANG_MAP = {
    "en": "en",
    "hi": "hi",        # Hindi (Devanagari)
    "kn": "kn",        # Kannada — use 'en' model but keep Kannada dict
    "multilang": "en", # fall back to English model (handles mixed scripts well)
}

# Minimum short-side length (px) for a detected box to be sent to recognition.
# QR-code modules and random noise produce boxes << 15 px; real printed text is
# typically ≥ 20 px tall on a phone-photo of a labour card.
_MIN_BOX_SHORT_SIDE: int = 15

# Quick phone-pattern check used inside the recognition loop for early exit.
# Intentionally simple (no phonenumbers library overhead) — just needs to
# detect "looks like an Indian mobile" to stop the loop.
_PHONE_QUICK_RE = re.compile(
    r"(?:(?:\+|00)?91[\s\-.]?)?[6-9]\d{9}"
)


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def _sort_boxes(boxes: list) -> list:
    """Sort 4-corner boxes top-to-bottom, left-to-right (PaddleOCR convention)."""
    boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    # Swap neighbours on the same row that are right-to-left
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if (abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10
                    and boxes[j + 1][0][0] < boxes[j][0][0]):
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def _box_short_side(box) -> float:
    """Return the shorter dimension of a 4-corner text box in pixels."""
    pts = np.array(box, dtype=np.float32)
    w = float(np.linalg.norm(pts[1] - pts[0]))
    h = float(np.linalg.norm(pts[3] - pts[0]))
    return min(w, h)


def _crop_box(img: np.ndarray, box) -> np.ndarray:
    """Perspective-crop a detected text region to a flat rectangle."""
    pts = np.array(box, dtype=np.float32)
    crop_w = int(max(np.linalg.norm(pts[1] - pts[0]),
                     np.linalg.norm(pts[2] - pts[3])))
    crop_h = int(max(np.linalg.norm(pts[0] - pts[3]),
                     np.linalg.norm(pts[1] - pts[2])))
    if crop_w == 0 or crop_h == 0:
        return img  # degenerate box — pass full image as fallback
    dst = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(pts, dst)
    cropped = cv2.warpPerspective(img, M, (crop_w, crop_h),
                                  borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
    # PaddleOCR convention: rotate tall crops 90°
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        if cropped.shape[0] / cropped.shape[1] >= 1.5:
            cropped = np.rot90(cropped)
    return cropped


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OCRService:
    _lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        lang = _LANG_MAP.get(settings.ocr_lang, "en")
        logger.info("ocr.init", use_gpu=settings.use_gpu, lang=lang)
        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=settings.use_gpu,
            show_log=False,
            det_db_thresh=settings.ocr_det_db_thresh,
            det_db_box_thresh=settings.ocr_det_db_box_thresh,
            det_model_dir="/models/det",
            rec_model_dir="/models/rec",
            cls_model_dir="/models/cls",
        )
        logger.info("ocr.ready")

    def run(self, image: np.ndarray) -> str:
        """Two-stage OCR with early exit on first phone-number match.

        Stage 1 — detection runs once on the full image (fixed cost).
        Stage 2 — recognition iterates boxes top-to-bottom and stops as soon as
                   a phone-number pattern is found, skipping the QR code region
                   and everything below the phone number.

        Returns:
            Text lines joined by newlines, up to and including the phone-number
            line (subsequent lines are not recognised).
        """
        with self._lock:
            # ── Stage 1: detect all text regions ──────────────────────────
            dt_boxes, _ = self._engine.text_detector(image)

            if dt_boxes is None or len(dt_boxes) == 0:
                logger.debug("ocr.no_boxes_detected")
                return ""

            # Sort top→bottom, left→right; drop tiny QR-code / noise boxes
            boxes = _sort_boxes(list(dt_boxes))
            boxes = [b for b in boxes if _box_short_side(b) >= _MIN_BOX_SHORT_SIDE]

            logger.debug("ocr.boxes", total=len(dt_boxes), after_filter=len(boxes))

            lines: List[str] = []

            # ── Stage 2: recognise one box at a time, stop on phone match ──
            for box in boxes:
                crop = _crop_box(image, box)

                # Angle-classify this crop (handles upside-down text regions)
                if self._engine.use_angle_cls and hasattr(self._engine, "text_classifier"):
                    crops, _, _ = self._engine.text_classifier([crop])
                    crop = crops[0]

                rec_res, _ = self._engine.text_recognizer([crop])
                if not rec_res or not rec_res[0]:
                    continue

                text = rec_res[0][0].strip()
                if not text:
                    continue

                lines.append(text)

                # Early exit — stop recognising once a phone number is found
                if _PHONE_QUICK_RE.search(text):
                    logger.debug(
                        "ocr.early_exit",
                        matched=text,
                        recognised=len(lines),
                        skipped=len(boxes) - len(lines),
                    )
                    break

        full_text = "\n".join(lines)
        logger.debug("ocr.run.complete", lines=len(lines), text_preview=full_text[:120])
        return full_text
