from pydantic import BaseModel, Field
from typing import List

class DetectResponse(BaseModel):
    phone_numbers: List[str] = Field(
        description="Deduplicated list of phone numbers found in the image (E.164 or raw format)"
    )
    raw_text: str = Field(description="All text extracted from the image via OCR")
    confidence_scores: List[float] = Field(
        description="OCR confidence per recognised phone-number token (0–1)"
    )
    latency_ms: float = Field(description="Total server-side processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
