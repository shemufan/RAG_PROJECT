"""Models for retrieved evidence and classification results."""

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One knowledge passage supporting a classification."""

    content: str
    source: str
    article: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    chunk_id: str | None = None


class RegulationRetrievalResult(BaseModel):
    """Regulation evidence paired with the store's unmodified score."""

    evidence: Evidence
    raw_score: float


class ClassificationOutput(BaseModel):
    """Strict structure returned by the language model."""

    is_personal: bool
    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    need_review: bool


class ClassificationResult(BaseModel):
    """Complete API result including retrieval evidence."""

    field_name: str
    is_personal: bool | None = None
    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4", "UNKNOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    need_review: bool
    decision_path: str


class ClassifyResponse(BaseModel):
    """Stable envelope for the classification endpoint."""

    code: int = 200
    message: str = "success"
    data: ClassificationResult | None = None
