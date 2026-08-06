"""Canonical models produced by file input adapters."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.field import FieldProfile

InputMode = Literal["auto", "catalog", "tabular"]
ResolvedInputMode = Literal["catalog", "tabular"]


class CSVReadLimits(BaseModel):
    max_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_rows: int = Field(default=1_000_000, ge=1)
    max_columns: int = Field(default=10_000, ge=1)


class CSVInspection(BaseModel):
    source_name: str
    source_fingerprint: str
    encoding: str
    headers: list[str]
    row_count: int = Field(ge=0)


class CSVFieldCase(BaseModel):
    case_index: int = Field(ge=1)
    field_profile: FieldProfile
    expected_personal: bool | None = None


class CSVInputBatch(BaseModel):
    source_name: str
    source_fingerprint: str
    input_mode: ResolvedInputMode
    cases: list[CSVFieldCase] = Field(default_factory=list)


class LabelMatchSummary(BaseModel):
    label_fingerprint: str
    labeled_cases: int = Field(ge=0)
    unlabeled_cases: int = Field(ge=0)
    cases: list[CSVFieldCase] = Field(default_factory=list)
