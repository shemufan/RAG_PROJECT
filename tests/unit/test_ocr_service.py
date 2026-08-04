import base64
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app.services.ocr_service as ocr_module
from app.services.ocr_service import OCRExtractionError, QwenOCRService
from app.services.pdf_image_service import PdfImageRenderError


def write_pdf(path: Path, *, pages: int = 1, encrypted: bool = False) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    if encrypted:
        writer.encrypt("secret")
    with path.open("wb") as stream:
        writer.write(stream)


def write_mixed_pdf(path: Path) -> str:
    writer = PdfWriter()
    text = "Native legal text is available on this page. " * 4
    page = writer.add_blank_page(width=595, height=842)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_ref}
            )
        }
    )
    contents = DecodedStreamObject()
    contents.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(contents)
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as stream:
        writer.write(stream)
    return text


def write_sparse_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    contents = DecodedStreamObject()
    contents.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(contents)
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


class FakeImageService:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def render_pages(self, path, page_numbers=None, *, prune=False):
        self.calls.append((path, page_numbers))
        if page_numbers is None:
            return self.pages
        return [self.pages[number - 1] for number in page_numbers]


class FakeCompletions:
    def __init__(self, outputs, *, error_at=None, error=None):
        self.outputs = outputs
        self.error_at = error_at
        self.error = error or RuntimeError("simulated OCR failure")
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        call_number = len(self.calls)
        if self.error_at == call_number:
            raise self.error
        content = self.outputs[call_number - 1]
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def write_page_images(tmp_path: Path, count: int) -> list[Path]:
    pages = []
    for index in range(1, count + 1):
        page = tmp_path / f"page-{index:04d}.jpg"
        image = Image.new("RGB", (40, 40), "black")
        image.save(page, format="JPEG")
        image.close()
        pages.append(page)
    return pages


def make_transport_service(tmp_path, image_service, completions):
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return QwenOCRService(
        api_key="test-api-key",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.5-ocr",
        cache_dir=tmp_path / "cache",
        image_service=image_service,
        chat_client=client,
    )


def test_cache_miss_renders_pages_and_returns_ordered_qwen_output(tmp_path):
    pdf = tmp_path / "新标准.pdf"
    write_pdf(pdf, pages=2)
    pages = write_page_images(tmp_path, 2)
    renderer = FakeImageService(pages)
    completions = FakeCompletions(["第一页内容", "第二页内容"])
    service = make_transport_service(tmp_path, renderer, completions)

    text = service.extract_pdf(pdf)

    assert text == "【第 1 页】\n第一页内容\n\n【第 2 页】\n第二页内容"
    assert renderer.calls == [(pdf, [1, 2])]
    assert len(completions.calls) == 2


def test_qwen_request_uses_base64_image_and_legal_prompt(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf)
    page = write_page_images(tmp_path, 1)[0]
    completions = FakeCompletions(["第一条 测试法规"])
    service = make_transport_service(tmp_path, FakeImageService([page]), completions)

    service.extract_pdf(pdf)

    request = completions.calls[0]
    assert request["model"] == "qwen3.5-ocr"
    content = request["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    data_url = content[0]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == page.read_bytes()
    assert content[1]["type"] == "text"
    assert "不要总结" in content[1]["text"]
    assert "章节" in content[1]["text"]


def test_text_cache_hit_skips_rendering_and_api(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf)
    page = write_page_images(tmp_path, 1)[0]
    renderer = FakeImageService([page])
    completions = FakeCompletions(["第一条 测试法规"])
    service = make_transport_service(tmp_path, renderer, completions)

    first = service.extract_pdf(pdf)
    second = service.extract_pdf(pdf)

    assert second == first
    assert renderer.calls == [(pdf, [1])]
    assert len(completions.calls) == 1


def test_native_text_page_skips_image_rendering_and_qwen(tmp_path):
    pdf = tmp_path / "native.pdf"
    expected = write_mixed_pdf(pdf)
    renderer = FakeImageService(write_page_images(tmp_path, 2))
    completions = FakeCompletions(["扫描页内容"])
    service = make_transport_service(tmp_path, renderer, completions)

    text = service.extract_pdf(pdf)

    assert expected.strip() in text
    assert "【第 2 页】\n扫描页内容" in text
    assert renderer.calls == [(pdf, [2])]
    assert len(completions.calls) == 1


def test_sparse_valid_native_text_does_not_trigger_ocr(tmp_path):
    pdf = tmp_path / "sparse.pdf"
    write_sparse_text_pdf(pdf, "GB/T 35273-2020")
    renderer = FakeImageService(write_page_images(tmp_path, 1))
    completions = FakeCompletions([])
    service = make_transport_service(tmp_path, renderer, completions)

    text = service.extract_pdf(pdf)

    assert "GB/T 35273-2020" in text
    assert renderer.calls == []
    assert completions.calls == []


def test_blank_page_is_skipped_without_qwen_or_text_cache(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    native_text = write_mixed_pdf(pdf)
    pages = write_page_images(tmp_path, 2)
    blank = Image.new("RGB", (40, 40), "white")
    blank.save(pages[1], format="JPEG")
    blank.close()
    renderer = FakeImageService(pages)
    completions = FakeCompletions([])
    service = make_transport_service(tmp_path, renderer, completions)

    text = service.extract_pdf(pdf)

    assert native_text.strip() in text
    assert "【第 2 页】" not in text
    assert completions.calls == []
    assert not pages[1].exists()
    assert not list((tmp_path / "cache").glob("page-*.txt"))


def test_page_failure_reports_page_number_and_does_not_cache(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf, pages=2)
    pages = write_page_images(tmp_path, 2)
    completions = FakeCompletions(["第一页"], error_at=2)
    service = make_transport_service(tmp_path, FakeImageService(pages), completions)

    with pytest.raises(OCRExtractionError, match="law.pdf.*page 2"):
        service.extract_pdf(pdf)

    cache_files = list((tmp_path / "cache").glob("*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].name.startswith("page-")


def test_successful_pages_are_cached_before_a_later_page_fails(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf, pages=2)
    pages = write_page_images(tmp_path, 2)
    renderer = FakeImageService(pages)
    first_attempt = FakeCompletions(["第一页"], error_at=2)
    service = make_transport_service(tmp_path, renderer, first_attempt)

    with pytest.raises(OCRExtractionError, match="page 2"):
        service.extract_pdf(pdf)

    retry = FakeCompletions(["第二页"])
    retry_service = make_transport_service(tmp_path, renderer, retry)
    text = retry_service.extract_pdf(pdf)

    assert text == "【第 1 页】\n第一页\n\n【第 2 页】\n第二页"
    assert len(retry.calls) == 1


def test_render_failure_preserves_pdf_name_and_page_number(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf, pages=2)

    class FailingImageService:
        def render_pages(self, _, page_numbers=None, *, prune=False):
            raise PdfImageRenderError("law.pdf: failed to render page 2")

    service = make_transport_service(
        tmp_path,
        FailingImageService(),
        FakeCompletions([]),
    )

    with pytest.raises(OCRExtractionError, match="law.pdf.*page 2"):
        service.extract_pdf(pdf)


@pytest.mark.parametrize(
    "output",
    ["   ", None, "当前没有可供提取的法规文档，因此无法提取。"],
)
def test_invalid_qwen_output_is_rejected_and_not_cached(tmp_path, output):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf)
    page = write_page_images(tmp_path, 1)[0]
    service = make_transport_service(
        tmp_path,
        FakeImageService([page]),
        FakeCompletions([output]),
    )

    with pytest.raises(OCRExtractionError, match="law.pdf.*page 1.*empty or invalid"):
        service.extract_pdf(pdf)

    assert not list((tmp_path / "cache").glob("*.txt"))


def test_oversized_base64_page_is_rejected_before_api(tmp_path, monkeypatch):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf)
    page = write_page_images(tmp_path, 1)[0]
    monkeypatch.setattr(ocr_module, "MAX_BASE64_IMAGE_BYTES", 1)
    completions = FakeCompletions(["never"])
    service = make_transport_service(tmp_path, FakeImageService([page]), completions)

    with pytest.raises(OCRExtractionError, match="law.pdf.*page 1.*10 MB"):
        service.extract_pdf(pdf)

    assert completions.calls == []


def test_qwen_error_does_not_leak_credentials_or_image_data(tmp_path):
    pdf = tmp_path / "law.pdf"
    write_pdf(pdf)
    page = write_page_images(tmp_path, 1)[0]
    unsafe = RuntimeError("test-api-key data:image/jpeg;base64,secret-image")
    completions = FakeCompletions([], error_at=1, error=unsafe)
    service = make_transport_service(tmp_path, FakeImageService([page]), completions)

    with pytest.raises(OCRExtractionError) as captured:
        service.extract_pdf(pdf)

    message = str(captured.value)
    assert "law.pdf" in message
    assert "page 1" in message
    assert "test-api-key" not in message
    assert "base64" not in message
    rendered_traceback = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "test-api-key" not in rendered_traceback
    assert "secret-image" not in rendered_traceback
