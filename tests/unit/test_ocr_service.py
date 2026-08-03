from pathlib import Path

import pytest
from pypdf import PdfWriter

import app.services.ocr_service as ocr_module
from app.services.ocr_service import OCRExtractionError, QwenOCRService


def write_pdf(path: Path, *, pages: int = 1, encrypted: bool = False) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    if encrypted:
        writer.encrypt("secret")
    with path.open("wb") as stream:
        writer.write(stream)


def make_service(tmp_path: Path, extractor, *, model: str = "qwen3.5-ocr"):
    return QwenOCRService(
        api_key="test-key",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model=model,
        cache_dir=tmp_path / "cache",
        extractor=extractor,
    )


def test_extract_pdf_rejects_corrupt_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    service = make_service(tmp_path, lambda _: "never called")

    with pytest.raises(OCRExtractionError, match="broken.pdf"):
        service.extract_pdf(path)


def test_extract_pdf_rejects_encrypted_pdf(tmp_path):
    path = tmp_path / "encrypted.pdf"
    write_pdf(path, encrypted=True)
    service = make_service(tmp_path, lambda _: "never called")

    with pytest.raises(OCRExtractionError, match="encrypted.pdf.*encrypted"):
        service.extract_pdf(path)


def test_extract_pdf_rejects_zero_pages(tmp_path):
    path = tmp_path / "empty.pdf"
    write_pdf(path, pages=0)
    service = make_service(tmp_path, lambda _: "never called")

    with pytest.raises(OCRExtractionError, match="empty.pdf.*page count"):
        service.extract_pdf(path)


def test_extract_pdf_rejects_more_than_50_pages(tmp_path):
    path = tmp_path / "long.pdf"
    write_pdf(path, pages=51)
    service = make_service(tmp_path, lambda _: "never called")

    with pytest.raises(OCRExtractionError, match="long.pdf.*page count"):
        service.extract_pdf(path)


def test_extract_pdf_rejects_oversized_file(tmp_path, monkeypatch):
    path = tmp_path / "large.pdf"
    write_pdf(path)
    monkeypatch.setattr(ocr_module, "MAX_PDF_BYTES", 1)
    service = make_service(tmp_path, lambda _: "never called")

    with pytest.raises(OCRExtractionError, match="large.pdf.*size limit"):
        service.extract_pdf(path)


def test_extract_pdf_uses_cache_without_repeating_extraction(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    calls = []
    service = make_service(
        tmp_path,
        lambda pdf_path: calls.append(pdf_path) or "第一条 测试法规",
    )

    first = service.extract_pdf(path)
    second = service.extract_pdf(path)

    assert first == second == "第一条 测试法规"
    assert calls == [path]
    assert len(list((tmp_path / "cache").glob("*.txt"))) == 1


def test_pdf_content_change_invalidates_cache(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path, pages=1)
    calls = []
    service = make_service(
        tmp_path,
        lambda _: calls.append(len(calls)) or f"文本-{len(calls)}",
    )

    assert service.extract_pdf(path) == "文本-1"
    write_pdf(path, pages=2)
    assert service.extract_pdf(path) == "文本-2"
    assert len(calls) == 2


def test_model_change_invalidates_cache(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    calls = []
    first = make_service(
        tmp_path,
        lambda _: calls.append("first") or "模型一文本",
        model="qwen3.5-ocr",
    )
    second = make_service(
        tmp_path,
        lambda _: calls.append("second") or "模型二文本",
        model="qwen-next-ocr",
    )

    assert first.extract_pdf(path) == "模型一文本"
    assert second.extract_pdf(path) == "模型二文本"
    assert calls == ["first", "second"]


def test_empty_extraction_is_rejected_and_not_cached(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    service = make_service(tmp_path, lambda _: "   ")

    with pytest.raises(OCRExtractionError, match="law.pdf.*empty"):
        service.extract_pdf(path)

    assert not list((tmp_path / "cache").glob("*.txt"))
