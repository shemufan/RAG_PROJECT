"""Opt-in paid integration coverage for one non-sensitive PDF."""

import os
from pathlib import Path

import pytest

from app.services.ocr_service import QwenOCRService

RUN_FLAG = "RUN_QWEN_OCR_INTEGRATION"
PDF_ENV = "QWEN_OCR_TEST_PDF"
KEY_ENV = "QWEN_OCR_API_KEY"
BASE_URL_ENV = "QWEN_OCR_BASE_URL"


def _integration_configuration():
    if os.getenv(RUN_FLAG) != "1":
        pytest.skip(f"{RUN_FLAG}=1 is required for the paid Qwen OCR integration test")
    missing = [name for name in (PDF_ENV, KEY_ENV, BASE_URL_ENV) if not os.getenv(name)]
    if missing:
        pytest.skip(f"Qwen OCR integration variables are not configured: {', '.join(missing)}")
    pdf_path = Path(os.environ[PDF_ENV]).resolve()
    if not pdf_path.is_file():
        pytest.fail(f"Qwen OCR test PDF does not exist: {pdf_path.name}")
    if pdf_path.stat().st_size > 5 * 1024 * 1024:
        pytest.fail("Qwen OCR integration PDF must be no larger than 5 MB")
    return pdf_path


def test_qwen_ocr_extracts_non_empty_text_from_explicit_test_pdf(tmp_path):
    pdf_path = _integration_configuration()
    service = QwenOCRService(
        api_key=os.environ[KEY_ENV],
        base_url=os.environ[BASE_URL_ENV],
        model=os.getenv("QWEN_OCR_MODEL", "qwen3.5-ocr"),
        cache_dir=tmp_path / "ocr-cache",
    )

    text = service.extract_pdf(pdf_path)

    assert text.strip()
