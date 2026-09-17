"""Typed records for knowledge extraction, cleaning, and quality checks."""

from enum import Enum

from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    """Whether a document may proceed to vector ingestion."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class ExtractedPage(BaseModel):
    """Text extracted from one source page, with its extraction provenance."""

    page_number: int = Field(gt=0)
    text: str = ""
    extraction_method: str
    is_blank: bool = False
    segment_count: int = Field(default=1, ge=1)


class SourceManifest(BaseModel):
    """Stable identity and extraction configuration for one source document."""

    document_name: str
    source_format: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(gt=0)
    extractor_version: str = Field(min_length=1)
    encoding: str | None = None


class PageQuality(BaseModel):
    """Measured quality information for one source page."""

    page_number: int = Field(gt=0)
    extraction_method: str
    character_count: int = Field(ge=0)
    is_blank: bool = False
    segment_count: int = Field(default=1, ge=1)
    issues: list[str] = Field(default_factory=list)


class CleaningAudit(BaseModel):
    """A count and bounded examples of content removed by one cleaning rule."""

    rule: str
    removed_count: int = Field(ge=0)
    samples: list[str] = Field(default_factory=list, max_length=5)


class CleanedKnowledge(BaseModel):
    """Clean text with page boundaries and an audit trail."""

    text: str
    pages: list[ExtractedPage]
    audits: list[CleaningAudit] = Field(default_factory=list)


class KnowledgeQualityReport(BaseModel):
    """Serializable document-level decision and supporting measurements."""

    document_name: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: QualityStatus
    pages: list[PageQuality] = Field(default_factory=list)
    audits: list[CleaningAudit] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int] = Field(default_factory=dict)
