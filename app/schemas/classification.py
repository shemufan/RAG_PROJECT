"""Models for retrieved evidence and classification results."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.semantic import (
    ObjectiveValueProfile,
    SemanticCard,
    SemanticRetrievalResult,
)


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


class SemanticBridgeTrace(BaseModel):
    """Internal layer-by-layer trace exported by Experiment E."""

    profiling: ObjectiveValueProfile = Field(default_factory=ObjectiveValueProfile)
    semantic_query: str = ""
    semantic_retrieval: list[SemanticRetrievalResult] = Field(default_factory=list)
    selected_semantic_type: str | None = None
    selected_card: SemanticCard | None = None
    top1_top2_score_gap: float | None = None
    regulation_query: str = ""
    regulation_retrieval: list[RegulationRetrievalResult] = Field(
        default_factory=list
    )
    failed_stage: str | None = None


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
    experiment_trace: SemanticBridgeTrace | None = Field(default=None, exclude=True)


class ClassifyResponse(BaseModel):
    """Stable envelope for the classification endpoint."""

    code: int = 200
    message: str = "success"
    data: ClassificationResult | None = None
