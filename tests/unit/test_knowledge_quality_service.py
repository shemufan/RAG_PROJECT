from app.schemas.knowledge_quality import (
    CleanedKnowledge,
    ExtractedPage,
    QualityStatus,
    SourceManifest,
)
from app.services.knowledge_quality_service import KnowledgeQualityService


def manifest(page_count: int = 2) -> SourceManifest:
    return SourceManifest(
        document_name="standard.pdf",
        source_format="pdf",
        source_sha256="a" * 64,
        page_count=page_count,
        extractor_version="v1",
    )


def cleaned(*pages: ExtractedPage) -> CleanedKnowledge:
    return CleanedKnowledge(
        text="\n\n".join(page.text for page in pages),
        pages=list(pages),
    )


def page(number: int, text: str, *, blank: bool = False) -> ExtractedPage:
    return ExtractedPage(
        page_number=number,
        text=text,
        extraction_method="ocr",
        is_blank=blank,
    )


def test_complete_structured_document_passes_quality_gate():
    report = KnowledgeQualityService().evaluate(
        manifest(),
        cleaned(page(1, "1 范围\n本标准规定数据处理要求。"), page(2, "2 要求\n处理者应保护数据。")),
    )

    assert report.status is QualityStatus.PASS
    assert report.issues == []


def test_missing_page_and_unconfirmed_empty_page_fail():
    missing = KnowledgeQualityService().evaluate(
        manifest(),
        cleaned(page(1, "1 范围\n正文")),
    )
    empty = KnowledgeQualityService().evaluate(
        manifest(),
        cleaned(page(1, "1 范围\n正文"), page(2, "")),
    )

    assert missing.status is QualityStatus.FAIL
    assert any("缺少页状态" in issue for issue in missing.issues)
    assert empty.status is QualityStatus.FAIL
    assert any("非确认空白页没有文本" in issue for issue in empty.issues)


def test_confirmed_blank_page_is_accepted():
    report = KnowledgeQualityService().evaluate(
        manifest(),
        cleaned(page(1, "1 范围\n正文"), page(2, "", blank=True)),
    )

    assert report.status is QualityStatus.PASS


def test_extreme_page_length_and_excessive_repetition_require_review():
    repeated = "\n".join(["重复表头"] * 30 + ["1 范围", "正文"])
    report = KnowledgeQualityService().evaluate(
        manifest(3),
        cleaned(
            page(1, "1 范围\n短正文"),
            page(2, "2 要求\n短正文"),
            page(3, repeated + "超长" * 100),
        ),
    )

    assert report.status is QualityStatus.REVIEW
    assert any("重复行" in issue for issue in report.issues)
    assert any("字符数异常" in issue for issue in report.issues)


def test_missing_start_section_or_required_appendix_fails():
    service = KnowledgeQualityService()
    no_start = service.evaluate(
        manifest(1),
        cleaned(page(1, "这是没有章节标题的正文。")),
    )
    no_appendix = service.evaluate(
        manifest(1),
        cleaned(page(1, "1 范围\n正文")),
        requires_appendix=True,
    )

    assert no_start.status is QualityStatus.FAIL
    assert no_appendix.status is QualityStatus.FAIL


def test_review_requires_explicit_source_hash_approval():
    service = KnowledgeQualityService()
    report = service.evaluate(
        manifest(2),
        cleaned(page(1, "1 范围\n短"), page(2, "2 要求\n" + "长" * 500)),
    )

    assert report.status is QualityStatus.REVIEW
    assert not service.can_ingest(report)
    assert service.can_ingest(report, approved_hashes={"a" * 64})


def test_segmented_container_pages_always_require_human_review():
    segmented = ExtractedPage(
        page_number=1,
        text="1 范围\n正文",
        extraction_method="ocr",
        segment_count=5,
    )

    report = KnowledgeQualityService().evaluate(
        manifest(1),
        cleaned(segmented),
    )

    assert report.status is QualityStatus.REVIEW
    assert any("分段 OCR" in issue for issue in report.issues)


def test_abnormally_long_line_requires_review():
    report = KnowledgeQualityService().evaluate(
        manifest(1),
        cleaned(page(1, "1 范围\n" + "异常连续文本" * 180)),
    )

    assert report.status is QualityStatus.REVIEW
    assert any("单行长度异常" in issue for issue in report.issues)
