import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
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


class FakeHTTPResponse:
    def __init__(self, payload=None, *, error: Exception | None = None):
        self.payload = payload or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeHTTPSession:
    def __init__(self, policy_response, upload_response=None):
        self.policy_response = policy_response
        self.upload_response = upload_response or FakeHTTPResponse()
        self.get_calls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.policy_response

    def post(self, url, **kwargs):
        files = kwargs["files"]
        self.post_calls.append(
            {
                "url": url,
                "field_names": set(files),
                "file_name": files["file"][0],
                "timeout": kwargs["timeout"],
            }
        )
        return self.upload_response


class FakeResponsesClient:
    def __init__(self, output_text="第一章 总则\n第一条 测试法规", error=None):
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


def upload_policy():
    return {
        "data": {
            "upload_dir": "tmp/upload-secret",
            "upload_host": "https://upload.example.test",
            "oss_access_key_id": "temporary-access-id",
            "signature": "signed-secret",
            "policy": "policy-secret",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
        }
    }


def make_transport_service(tmp_path, session, responses):
    return QwenOCRService(
        api_key="test-api-key",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.5-ocr",
        cache_dir=tmp_path / "cache",
        http_session=session,
        responses_client=responses,
    )


def test_cache_miss_uploads_pdf_and_returns_qwen_output(tmp_path):
    path = tmp_path / "新标准.pdf"
    write_pdf(path)
    session = FakeHTTPSession(FakeHTTPResponse(upload_policy()))
    responses = FakeResponsesClient()
    service = make_transport_service(tmp_path, session, responses)

    text = service.extract_pdf(path)

    assert text == "第一章 总则\n第一条 测试法规"
    assert session.get_calls[0][0] == "https://dashscope.aliyuncs.com/api/v1/uploads"
    assert session.get_calls[0][1]["params"] == {
        "action": "getPolicy",
        "model": "qwen3.5-ocr",
    }
    assert session.post_calls[0]["url"] == "https://upload.example.test"
    assert session.post_calls[0]["file_name"] == "新标准.pdf"


def test_qwen_request_uses_document_parsing_and_legal_prompt(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    session = FakeHTTPSession(FakeHTTPResponse(upload_policy()))
    responses = FakeResponsesClient()
    service = make_transport_service(tmp_path, session, responses)

    service.extract_pdf(path)

    request = responses.calls[0]
    assert request["model"] == "qwen3.5-ocr"
    assert request["extra_body"] == {"ocr_options": {"task": "document_parsing"}}
    content = request["input"][0]["content"]
    assert content[0] == {
        "type": "input_file",
        "file_url": "oss://tmp/upload-secret/law.pdf",
    }
    assert content[1]["type"] == "input_text"
    assert "不要总结" in content[1]["text"]
    assert "章节" in content[1]["text"]


def test_empty_qwen_output_is_rejected_and_not_cached(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    session = FakeHTTPSession(FakeHTTPResponse(upload_policy()))
    service = make_transport_service(tmp_path, session, FakeResponsesClient("   "))

    with pytest.raises(OCRExtractionError, match="law.pdf.*empty"):
        service.extract_pdf(path)

    assert not list((tmp_path / "cache").glob("*.txt"))


def test_none_qwen_output_is_rejected_and_not_cached(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    session = FakeHTTPSession(FakeHTTPResponse(upload_policy()))
    service = make_transport_service(tmp_path, session, FakeResponsesClient(None))

    with pytest.raises(OCRExtractionError, match="law.pdf.*empty"):
        service.extract_pdf(path)

    assert not list((tmp_path / "cache").glob("*.txt"))


def test_upload_error_does_not_leak_credentials_or_signed_values(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    unsafe = requests.HTTPError(
        "Bearer test-api-key signed-secret oss://tmp/upload-secret/law.pdf"
    )
    session = FakeHTTPSession(FakeHTTPResponse(error=unsafe))
    service = make_transport_service(tmp_path, session, FakeResponsesClient())

    with pytest.raises(OCRExtractionError) as captured:
        service.extract_pdf(path)

    message = str(captured.value)
    assert "law.pdf" in message
    assert "test-api-key" not in message
    assert "signed-secret" not in message
    assert "oss://" not in message
    rendered_traceback = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "test-api-key" not in rendered_traceback
    assert "signed-secret" not in rendered_traceback
    assert "oss://" not in rendered_traceback


def test_qwen_error_does_not_leak_credentials_or_file_url(tmp_path):
    path = tmp_path / "law.pdf"
    write_pdf(path)
    session = FakeHTTPSession(FakeHTTPResponse(upload_policy()))
    responses = FakeResponsesClient(
        error=RuntimeError("test-api-key oss://tmp/upload-secret/law.pdf")
    )
    service = make_transport_service(tmp_path, session, responses)

    with pytest.raises(OCRExtractionError) as captured:
        service.extract_pdf(path)

    message = str(captured.value)
    assert "law.pdf" in message
    assert "test-api-key" not in message
    assert "oss://" not in message
    rendered_traceback = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "test-api-key" not in rendered_traceback
    assert "oss://" not in rendered_traceback
