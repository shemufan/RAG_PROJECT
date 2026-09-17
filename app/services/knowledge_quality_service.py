"""Deterministic quality gates for cleaned knowledge documents."""

import re
from collections import Counter
from statistics import median_low

from app.schemas.knowledge_quality import (
    CleanedKnowledge,
    KnowledgeQualityReport,
    PageQuality,
    QualityStatus,
    SourceManifest,
)

_START_SECTION = re.compile(
    r"(?m)^(?:1(?:\.0+)*\s+\S|第一章\s*\S|第一条\s*\S)"
)
_APPENDIX = re.compile(r"(?m)^附录\s*[A-ZＡ-Ｚ一二三四五六七八九十]+")


class KnowledgeQualityService:
    """Measure extraction anomalies and decide whether ingestion is safe."""

    def evaluate(
        self,
        manifest: SourceManifest,
        cleaned: CleanedKnowledge,
        *,
        requires_start_section: bool = True,
        requires_appendix: bool = False,
    ) -> KnowledgeQualityReport:
        failures: list[str] = []
        reviews: list[str] = []
        expected_pages = set(range(1, manifest.page_count + 1))
        actual_pages = {page.page_number for page in cleaned.pages}
        missing = sorted(expected_pages - actual_pages)
        if missing:
            failures.append(f"缺少页状态：{missing}")

        page_reports = []
        lengths = []
        for page in cleaned.pages:
            character_count = len("".join(page.text.split()))
            page_issues = []
            if not page.is_blank and character_count == 0:
                issue = f"第 {page.page_number} 页为非确认空白页没有文本"
                failures.append(issue)
                page_issues.append(issue)
            if not page.is_blank and character_count:
                lengths.append((page.page_number, character_count))
            page_reports.append(
                PageQuality(
                    page_number=page.page_number,
                    extraction_method=page.extraction_method,
                    character_count=character_count,
                    is_blank=page.is_blank,
                    segment_count=page.segment_count,
                    issues=page_issues,
                )
            )
            if page.segment_count > 1:
                reviews.append(
                    f"第 {page.page_number} 页由 {page.segment_count} 个片段进行分段 OCR，"
                    "需要人工核对阅读顺序"
                )

        if requires_start_section and cleaned.text and not _START_SECTION.search(cleaned.text):
            failures.append("未识别到正文起始章节")
        if requires_appendix and not _APPENDIX.search(cleaned.text):
            failures.append("文档预期包含附录，但未识别到附录")

        median_characters = median_low([length for _, length in lengths]) if lengths else 0
        if median_characters:
            for page_number, length in lengths:
                if length > median_characters * 8:
                    reviews.append(
                        f"第 {page_number} 页字符数异常：{length}，"
                        f"超过中位数 {median_characters} 的 8 倍"
                    )

        lines = [line.strip() for line in cleaned.text.splitlines() if line.strip()]
        counts = Counter(lines)
        maximum_repetition = max(counts.values(), default=0)
        repetition_ratio = maximum_repetition / len(lines) if lines else 0.0
        if repetition_ratio > 0.2 and maximum_repetition >= 3:
            reviews.append(f"单一重复行占正文行比例过高：{repetition_ratio:.2%}")
        maximum_line_length = max((len(line) for line in lines), default=0)
        if maximum_line_length > 800:
            reviews.append(f"单行长度异常：{maximum_line_length} 字符，需要核对 OCR 输出")

        if failures:
            status = QualityStatus.FAIL
        elif reviews:
            status = QualityStatus.REVIEW
        else:
            status = QualityStatus.PASS
        return KnowledgeQualityReport(
            document_name=manifest.document_name,
            source_sha256=manifest.source_sha256,
            status=status,
            pages=page_reports,
            audits=cleaned.audits,
            issues=[*failures, *reviews],
            metrics={
                "page_count": manifest.page_count,
                "content_page_count": len(lengths),
                "median_page_characters": median_characters,
                "maximum_line_repetition_ratio": round(repetition_ratio, 4),
                "maximum_line_characters": maximum_line_length,
            },
        )

    @staticmethod
    def can_ingest(
        report: KnowledgeQualityReport,
        *,
        approved_hashes: set[str] | None = None,
    ) -> bool:
        """Allow PASS, or REVIEW explicitly approved by immutable source hash."""
        if report.status is QualityStatus.PASS:
            return True
        return (
            report.status is QualityStatus.REVIEW
            and report.source_sha256 in (approved_hashes or set())
        )
