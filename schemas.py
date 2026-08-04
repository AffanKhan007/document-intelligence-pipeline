from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ExtractedData(BaseModel):
    items: list[dict]
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    overall_confidence: float
    raw: dict


class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_path: str
    status: str
    extracted_data: Optional[ExtractedData] = None
    overall_confidence: Optional[float] = None
    human_corrections: Optional[dict] = None
    created_at: datetime
    updated_at: datetime


class VerificationRequest(BaseModel):
    corrected_data: dict


class StatisticsResponse(BaseModel):
    total_documents: int
    by_status: dict[str, int]
    average_confidence: Optional[float] = None
