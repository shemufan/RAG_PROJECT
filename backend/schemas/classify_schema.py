"""Shared request, inference, and response models for the Week3 baseline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldProfile(BaseModel):
    """Metadata describing one database field."""

    model_config = ConfigDict(extra="ignore")

    field_name: str = Field(min_length=1)
    field_cn: str | None = None
    field_comment: str | None = None
    data_type: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    business_domain: str = "general"
    table_name: str | None = None
    database_name: str | None = None
    source_system: str | None = None

    @field_validator("field_name")
    @classmethod
    def field_name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field_name 不能为空")
        return value


class Evidence(BaseModel):
    """One retrieved knowledge-base passage."""

    content: str
    source: str = "未知来源"
    article: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def document_name(self) -> str:
        """Compatibility name used by the legacy batch evaluator."""
        return self.source

    @property
    def hierarchy_level(self) -> str | None:
        """Compatibility name used by the legacy batch evaluator."""
        return self.article


class ClassificationOutput(BaseModel):
    """Strict structure returned by the LLM, before retrieval metadata is added."""

    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    need_review: bool


class ClassificationResult(BaseModel):
    """Complete business result returned by the classifier and API."""

    field_name: str
    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4", "UNKNOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    need_review: bool
    decision_path: str


class ClassifyResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: ClassificationResult | None = None
