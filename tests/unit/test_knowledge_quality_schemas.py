import pytest
from pydantic import ValidationError

from app.schemas.knowledge_quality import (
    CleaningAudit,
    ExtractedPage,
    KnowledgeQualityReport,
    PageQuality,
    QualityStatus,
    SourceManifest,
)


def test_quality_report_is_json_serializable():
    report = KnowledgeQualityReport(
        document_name="standard.pdf",
        source_sha256="a" * 64,
        status=QualityStatus.PASS,
        pages=[
            PageQuality(
                page_number=1,
                extraction_method="native",
                character_count=20,
            )
        ],
        audits=[CleaningAudit(rule="page_number", removed_count=1)],
    )

    payload = report.model_dump_json()

    assert '"status":"PASS"' in payload
    assert '"page_number":1' in payload


def test_page_numbers_must_be_positive():
    with pytest.raises(ValidationError):
        ExtractedPage(page_number=0, text="正文", extraction_method="ocr")


def test_source_manifest_requires_sha256_and_extractor_version():
    manifest = SourceManifest(
        document_name="standard.pdf",
        source_format="pdf",
        source_sha256="b" * 64,
        page_count=2,
        extractor_version="legal-hybrid-page-v4",
    )

    assert manifest.source_sha256 == "b" * 64
    assert manifest.extractor_version == "legal-hybrid-page-v4"

    with pytest.raises(ValidationError):
        SourceManifest(
            document_name="standard.pdf",
            source_format="pdf",
            source_sha256="short",
            page_count=2,
            extractor_version="v1",
        )
