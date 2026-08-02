"""Validated records for database classification runs and query responses."""

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.classification import ClassificationOutput, Evidence
from app.schemas.field import FieldProfile

RunStatus = Literal["RUNNING", "SUCCESS", "PARTIAL_FAILED", "FAILED"]
ClassificationLevel = Literal["L1", "L2", "L3", "L4"]
TableName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


def stable_field_id(profile: FieldProfile) -> UUID:
    """Return the repeatable identity of one physical database field."""
    identity = ":".join(
        (
            profile.source_system,
            profile.database_name,
            profile.table_name,
            profile.field_name,
        )
    )
    return uuid5(NAMESPACE_URL, identity)


class FieldClassificationRecord(BaseModel):
    """Combine one validated physical field with one successful classification."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    field_id: UUID
    field_profile: FieldProfile
    classification: ClassificationOutput
    evidence: list[Evidence] = Field(default_factory=list)
    decision_path: str
    model_name: str
    knowledge_base_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineRequest(BaseModel):
    """Options accepted by one synchronous database scan."""

    model_config = ConfigDict(extra="forbid")

    sample_limit: int = Field(default=3, ge=0, le=5)
    table_names: list[TableName] | None = None
    continue_on_error: bool = True


class PipelineSummary(BaseModel):
    """Lifecycle and counters returned for one pipeline run."""

    run_id: UUID
    source_database: str
    total_fields: int = Field(ge=0)
    success_fields: int = Field(ge=0)
    review_fields: int = Field(ge=0)
    failed_fields: int = Field(ge=0)
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None


class RunDetail(PipelineSummary):
    """Persisted run status returned by the target repository."""

    source_system: str
    model_name: str
    knowledge_base_version: str
    error_message: str | None = None


class ClassificationResultRow(BaseModel):
    """Queryable relational classification result joined with field identity."""

    result_id: int
    run_id: UUID
    field_id: UUID
    source_system: str
    database_name: str
    table_name: str
    column_name: str
    business_domain: str
    category: str
    subcategory: str | None = None
    level: ClassificationLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    need_review: bool
    decision_path: str
    created_at: datetime


class ClassificationEvidenceRow(BaseModel):
    """One persisted evidence row ordered within a result."""

    evidence_id: int
    result_id: int
    rank_no: int = Field(ge=1)
    document_name: str
    article: str | None = None
    content: str
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    chunk_id: str | None = None
