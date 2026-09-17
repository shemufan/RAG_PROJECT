from pathlib import Path

import pytest

from app.rag.chunker import split_knowledge_text
from app.schemas.knowledge_quality import ExtractedPage, QualityStatus
from app.services.knowledge_service import (
    KnowledgeQualityError,
    load_knowledge_documents,
    prepare_knowledge_documents,
)


class FakeOCRService:
    def __init__(self, text: str = "第一章 总则\n第一条 PDF法规内容"):
        self.text = text
        self.paths = []

    def extract_pdf(self, path: Path) -> str:
        self.paths.append(path)
        return self.text


def test_knowledge_loader_routes_pdf_to_ocr_and_preserves_name(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    pdf_path = laws / "新标准.pdf"
    pdf_path.write_bytes(b"fake pdf handled by injected OCR")
    ocr = FakeOCRService()

    documents = load_knowledge_documents(tmp_path, version="pdf-v1", ocr_service=ocr)

    assert ocr.paths == [pdf_path]
    assert any("PDF法规内容" in item.page_content for item in documents)
    assert all(item.metadata["document_name"] == "新标准.pdf" for item in documents)
    assert all(item.metadata["source_format"] == "pdf" for item in documents)


def test_knowledge_loader_requires_ocr_service_when_pdf_exists(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "新标准.pdf").write_bytes(b"pdf")

    with pytest.raises(RuntimeError, match="新标准.pdf.*OCR"):
        load_knowledge_documents(tmp_path, version="v1")


def test_knowledge_loader_rejects_duplicate_txt_and_pdf_stems(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "same-law.txt").write_text("第一条 文本法规", encoding="utf-8")
    (laws / "same-law.pdf").write_bytes(b"pdf")

    with pytest.raises(RuntimeError, match="same-law.*duplicate"):
        load_knowledge_documents(tmp_path, version="v1", ocr_service=FakeOCRService())


def test_chunker_records_explicit_pdf_source_format():
    chunks = split_knowledge_text(
        "第一条 PDF法规内容",
        "新标准.pdf",
        source_type="legal_document",
        source_format="pdf",
        version="pdf-v1",
    )

    assert chunks[0].metadata["document_name"] == "新标准.pdf"
    assert chunks[0].metadata["source_format"] == "pdf"


def test_prepare_pipeline_cleans_validates_then_splits_pdf(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    pdf = laws / "standard.pdf"
    pdf.write_bytes(b"fake pdf handled by injected OCR")

    class PageOCR:
        def extract_pdf_pages(self, path):
            assert path == pdf
            return [
                ExtractedPage(
                    page_number=1,
                    extraction_method="ocr",
                    text="GB/T 00000—2026\n1 范围\n正文",
                ),
                ExtractedPage(
                    page_number=2,
                    extraction_method="ocr",
                    text="GB/T 00000—2026\n2 要求\n处理者应保护数据",
                ),
            ]

    prepared = prepare_knowledge_documents(
        tmp_path,
        version="clean-v1",
        ocr_service=PageOCR(),
    )

    assert prepared.reports[0].status is QualityStatus.PASS
    assert len(prepared.documents) == 2
    assert all("GB/T 00000—2026" not in doc.page_content for doc in prepared.documents)
    assert prepared.documents[0].metadata["page_start"] == 1
    assert "source_sha256" in prepared.documents[0].metadata


def test_loader_rejects_document_that_fails_quality_gate(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    pdf = laws / "broken.pdf"
    pdf.write_bytes(b"fake pdf handled by injected OCR")

    class EmptyPageOCR:
        def extract_pdf_pages(self, path):
            return [
                ExtractedPage(
                    page_number=1,
                    extraction_method="ocr",
                    text="",
                )
            ]

    with pytest.raises(KnowledgeQualityError, match="broken.pdf"):
        load_knowledge_documents(
            tmp_path,
            version="clean-v1",
            ocr_service=EmptyPageOCR(),
        )
