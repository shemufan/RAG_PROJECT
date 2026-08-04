"""Validated models for personal-information benchmark import and scoring."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.field import FieldProfile

DatasetLabel = Literal["personal", "non_personal"]


class BenchmarkImportRow(BaseModel):
    """One validated source CSV row before database insertion."""

    source_dataset: DatasetLabel
    source_row_number: int = Field(ge=2)
    field_name: str = Field(min_length=1, max_length=128)
    sample_values: list[str] = Field(default_factory=list, max_length=5)
    expected_personal: bool


class BenchmarkCase(BaseModel):
    """One stored benchmark case plus its hidden evaluation label."""

    benchmark_id: int = Field(ge=1)
    batch_name: str = Field(min_length=1, max_length=64)
    expected_personal: bool
    field_profile: FieldProfile


class BenchmarkMetricInput(BaseModel):
    """Minimal prediction state consumed by the pure evaluator."""

    expected_personal: bool
    predicted_personal: bool | None


class BenchmarkMetrics(BaseModel):
    """Confusion matrix, execution counts, and derived scores."""

    total_cases: int = 0
    success_cases: int = 0
    failed_cases: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision_score: float = 0.0
    recall_score: float = 0.0
    f1_score: float = 0.0
    accuracy_score: float = 0.0
    coverage_score: float = 0.0
    effective_recall_score: float = 0.0
