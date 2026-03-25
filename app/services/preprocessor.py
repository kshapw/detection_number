"""Image preprocessing pipeline before OCR.

Steps:
  1. Decode raw bytes → OpenCV BGR
  2. Decompression-bomb guard (max pixel dimensions)
  3. Convert to grayscale
  4. CLAHE contrast enhancement (helps with phone-photo lighting)
  5. Deskew via Hough line transform (±15 °)
  6. Return as RGB numpy array ready for PaddleOCR
"""

import math

import cv2
import numpy as np

# Hard cap on pixel dimensions to prevent decompression-bomb attacks.
# A 8000×8000 RGBA image is ~256 MB uncompressed — well within reason.
_MAX_PIXELS = 8000 * 8000  # 64 MP
_MAX_DIM = 8000             # px on longest side


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    h, w = img.shape[:2]
    if h * w > _MAX_PIXELS or max(h, w) > _MAX_DIM:
        raise ValueError(
            f"Image dimensions {w}×{h} exceed the allowed maximum of {_MAX_DIM}px "
            f"on each side ({_MAX_PIXELS // 1_000_000} MP total)."
        )
    return img


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


_DESKEW_SAMPLE_WIDTH = 640  # px — used only for angle detection, not final rotation


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate image to straighten dominant text lines (max ±15°)."""
    h, w = gray.shape

    # Downscale a copy for fast line detection; angle is scale-invariant.
    # A 3024×4032 photo → 640×853 sample = 21× fewer pixels for Canny + HoughLinesP.
    if w > _DESKEW_SAMPLE_WIDTH:
        scale = _DESKEW_SAMPLE_WIDTH / w
        sample = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        sample = gray  # already small enough

    edges = cv2.Canny(sample, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -15 <= angle <= 15:
                angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray  # already straight — skip warpAffine entirely

    # Rotate the FULL original image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _resize_if_small(img: np.ndarray, min_dim: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Return preprocessed image as HxWx3 uint8 RGB numpy array."""
    bgr = _decode(image_bytes)
    bgr = _resize_if_small(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe(gray)
    gray = _deskew(gray)
    # PaddleOCR accepts RGB numpy arrays; keep full-colour for better accuracy
    # but apply CLAHE result as luminance channel only
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = gray  # replace L channel with enhanced version
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return rgb
