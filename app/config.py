from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECT_", env_file=".env", extra="ignore")

    # OCR
    use_gpu: bool = True
    ocr_lang: str = "en"
    ocr_confidence_threshold: float = 0.6
    ocr_det_db_thresh: float = 0.3
    ocr_det_db_box_thresh: float = 0.5

    # Upload limits
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_content_types: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ]

    # Phone extraction
    default_phone_region: str = "IN"

    # Server
    log_level: str = "INFO"
    workers: int = 2

    # CORS — restrict to known origins in production; "*" only for local dev
    cors_origins: List[str] = ["*"]

    # Set to False in production to disable /docs and /redoc
    enable_docs: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
