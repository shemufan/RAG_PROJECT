"""Validated models for personal-information benchmark results and scoring."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.classification import Evidence


class BenchmarkMetricInput(BaseModel):
    """Minimal prediction state consumed by the pure evaluator."""

    expected_personal: bool | None
    predicted_personal: bool | None


class BenchmarkMetrics(BaseModel):
    """Confusion matrix, execution counts, and derived scores."""

    total_cases: int = 0
    success_cases: int = 0
    failed_cases: int = 0
    labeled_cases: int = 0
    unlabeled_cases: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision_score: float | None = 0.0
    recall_score: float | None = 0.0
    f1_score: float | None = 0.0
    accuracy_score: float | None = 0.0
    coverage_score: float = 0.0
    effective_recall_score: float | None = 0.0


class BenchmarkRunSummary(BenchmarkMetrics):
    """Lifecycle and scores for one benchmark execution."""

    run_id: UUID
    batch_name: str
    personal_limit: int | None = None
    non_personal_limit: int | None = None
    source_type: Literal["mysql_benchmark", "csv"] = "mysql_benchmark"
    input_mode: Literal["catalog", "tabular"] | None = None
    source_name: str | None = None
    source_fingerprint: str | None = None
    label_fingerprint: str | None = None
    status: Literal["RUNNING", "SUCCESS", "PARTIAL_FAILED", "FAILED"]
    model_name: str
    knowledge_base_version: str
    started_at: datetime
    finished_at: datetime | None = None
    error_message: str | None = None


class BenchmarkPrediction(BaseModel):
    """Persistable case-level prediction or explicit failure."""

    run_id: UUID
    benchmark_id: int
    field_name_snapshot: str
    sample_values: list[str] = Field(default_factory=list)
    expected_personal: bool | None
    predicted_personal: bool | None = None
    outcome: Literal["TP", "FP", "TN", "FN", "FAILED", "UNLABELED"]
    category: str | None = None
    subcategory: str | None = None
    level: str | None = None
    confidence: float | None = None
    reason: str | None = None
    need_review: bool | None = None
    decision_path: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["SUCCESS", "FAILED"]
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def metric_input(self) -> BenchmarkMetricInput:
        return BenchmarkMetricInput(
            expected_personal=self.expected_personal,
            predicted_personal=self.predicted_personal,
        )


class BenchmarkPredictionRow(BaseModel):
    """Validated read model for one persisted benchmark prediction."""

    prediction_id: int
    run_id: UUID
    benchmark_id: int
    field_name_snapshot: str
    sample_values: list[str] = Field(default_factory=list)
    expected_personal: bool | None
    predicted_personal: bool | None = None
    outcome: Literal["TP", "FP", "TN", "FN", "FAILED", "UNLABELED"]
    category: str | None = None
    subcategory: str | None = None
    level: str | None = None
    confidence: float | None = None
    reason: str | None = None
    need_review: bool | None = None
    decision_path: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["SUCCESS", "FAILED"]
    error_message: str | None = None
    created_at: datetime
