from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.rebuild_knowledge_base import (
    build_ocr_service,
    load_documents,
    rebuild_knowledge_base,
)


class FakeOCRService:
    def __init__(self, *, text="第一条 PDF法规", error=None):
        self.text = text
        self.error = error
        self.paths = []

    def extract_pdf(self, path: Path) -> str:
        self.paths.append(path)
        if self.error:
            raise self.error
        return self.text


class RecordingStore:
    def __init__(self):
        self.reset_called = False
        self.documents = []

    def reset(self):
        self.reset_called = True

    def add_documents(self, documents):
        self.documents.extend(documents)


def settings_for(tmp_path: Path, *, api_key="test-key", base_url="https://test/v1"):
    return SimpleNamespace(
        knowledge_dir=tmp_path,
        knowledge_base_version="pdf-v1",
        qwen_ocr_api_key=api_key,
        qwen_ocr_base_url=base_url,
        qwen_ocr_model="qwen3.5-ocr",
        qwen_ocr_cache_dir=tmp_path / "cache",
        qwen_ocr_timeout_seconds=60,
        qwen_ocr_max_retries=1,
    )


def test_txt_only_load_does_not_require_ocr(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "law.txt").write_text("第一条 文本法规", encoding="utf-8")

    documents = load_documents(tmp_path, version="v1")

    assert documents[0].metadata["document_name"] == "law.txt"


def test_pdf_load_uses_injected_ocr(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    pdf = laws / "law.pdf"
    pdf.write_bytes(b"pdf handled by fake")
    ocr = FakeOCRService()

    documents = load_documents(tmp_path, version="v1", ocr_service=ocr)

    assert ocr.paths == [pdf]
    assert documents[0].metadata["document_name"] == "law.pdf"


def test_pdf_failure_does_not_reset_vector_store(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "law.pdf").write_bytes(b"pdf handled by fake")
    store = RecordingStore()
    ocr = FakeOCRService(error=RuntimeError("OCR unavailable"))

    with pytest.raises(RuntimeError, match="OCR unavailable"):
        rebuild_knowledge_base(
            settings_for(tmp_path),
            vector_store=store,
            ocr_service=ocr,
        )

    assert store.reset_called is False
    assert store.documents == []


def test_build_ocr_service_requires_key_and_workspace_url(tmp_path):
    with pytest.raises(RuntimeError, match="QWEN_OCR_API_KEY"):
        build_ocr_service(settings_for(tmp_path, api_key=""))

    with pytest.raises(RuntimeError, match="QWEN_OCR_BASE_URL"):
        build_ocr_service(settings_for(tmp_path, base_url=""))
