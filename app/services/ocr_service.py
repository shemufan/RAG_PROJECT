"""PDF validation and cached Qwen OCR text extraction."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.schemas.knowledge_quality import ExtractedPage
from app.services.pdf_image_service import (
    PAGE_SEGMENT_VERSION,
    PdfImageRenderError,
    PdfImageService,
)

MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_BASE64_IMAGE_BYTES = 10 * 1024 * 1024
MIN_NATIVE_TEXT_CHARACTERS = 5
MIN_IMAGE_INK_RATIO = 0.0005
OCR_PROMPT_VERSION = "model-default-page-v7"
INVALID_OUTPUT_MARKERS = (
    "没有可供提取",
    "未提供法规文档",
    "未提供文档",
    "无法提取",
)
RETRYABLE_FORMAT_MARKERS = ("```html", "<html", "<body", "</html>")
STRICT_OCR_PROMPT = (
    "上一次输出包含了格式代码或推断内容，不符合逐字转录要求。请仅输出图像中清晰可见的"
    "原文纯文本，禁止输出 HTML 或 Markdown，禁止推断、解释、扩写或补充。图示只转录"
    "图中实际可见的文字标签；无法辨认的字符使用 ? 表示。"
)

_HTML_TAG = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
_BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "br",
    "caption",
    "div",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}


class _VisibleTextHTMLParser(HTMLParser):
    """Collect visible text while discarding only markup and embedded resources."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


class OCRExtractionError(RuntimeError):
    """Raised when a PDF cannot be safely converted to knowledge text."""


class QwenOCRService:
    """Validate PDFs, OCR locally rendered pages, and cache merged text."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        cache_dir: str | Path,
        timeout_seconds: float = 180,
        max_retries: int = 2,
        extractor=None,
        image_service=None,
        chat_client=None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._extractor = extractor
        self._image_service = image_service or PdfImageService()
        self._chat_client = chat_client

    def extract_pdf(self, path: str | Path) -> str:
        """Return cached or newly extracted text for one validated PDF."""
        pdf_path = Path(path)
        self._validate_pdf(pdf_path)
        cache_path = self._cache_path(pdf_path)
        if cache_path.is_file():
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached

        extractor = self._extractor or self._extract_with_qwen
        text = extractor(pdf_path).strip()
        if not text:
            raise OCRExtractionError(f"{pdf_path.name}: OCR returned empty text")
        self._write_cache(cache_path, text)
        return text

    def extract_pdf_pages(self, path: str | Path) -> list[ExtractedPage]:
        """Return ordered page-level text with extraction provenance."""
        pdf_path = Path(path)
        self._validate_pdf(pdf_path)
        if self._extractor is not None:
            text = self._extractor(pdf_path).strip()
            if not text:
                raise OCRExtractionError(f"{pdf_path.name}: OCR returned empty text")
            return [
                ExtractedPage(
                    page_number=1,
                    text=text,
                    extraction_method="custom",
                )
            ]
        return self._extract_with_qwen_pages(pdf_path)

    def _validate_pdf(self, path: Path) -> None:
        if not path.is_file():
            raise OCRExtractionError(f"{path.name}: PDF file does not exist")
        if path.stat().st_size > MAX_PDF_BYTES:
            raise OCRExtractionError(f"{path.name}: PDF exceeds size limit")
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise OCRExtractionError(f"{path.name}: encrypted PDF is not supported")
            page_count = len(reader.pages)
        except OCRExtractionError:
            raise
        except (PdfReadError, OSError, ValueError) as exc:
            raise OCRExtractionError(f"{path.name}: invalid PDF") from exc
        if not 1 <= page_count <= MAX_PDF_PAGES:
            raise OCRExtractionError(
                f"{path.name}: PDF page count must be between 1 and {MAX_PDF_PAGES}"
            )

    def _cache_path(self, path: Path) -> Path:
        return self.cache_dir / f"{self._cache_digest(path)}.txt"

    def _cache_digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        digest.update(self.model.encode("utf-8"))
        digest.update(OCR_PROMPT_VERSION.encode("utf-8"))
        digest.update(PAGE_SEGMENT_VERSION.encode("utf-8"))
        return digest.hexdigest()

    def _page_cache_path(self, path: Path, page_number: int) -> Path:
        digest = self._cache_digest(path)[:20]
        return self.cache_dir / f"page-{digest}-{page_number:04d}.txt"

    def _blank_cache_path(self, path: Path, page_number: int) -> Path:
        digest = self._cache_digest(path)[:20]
        return self.cache_dir / f"blank-{digest}-{page_number:04d}.flag"

    def _page_metadata_path(self, path: Path, page_number: int) -> Path:
        digest = self._cache_digest(path)[:20]
        return self.cache_dir / f"page-{digest}-{page_number:04d}.json"

    def _write_cache(self, cache_path: Path, text: str) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _extract_with_qwen(self, path: Path) -> str:
        pages = self._extract_with_qwen_pages(path)
        return "\n\n".join(
            f"【第 {page.page_number} 页】\n{page.text}"
            for page in pages
            if page.text
        )

    def _extract_with_qwen_pages(self, path: Path) -> list[ExtractedPage]:
        native_pages = self._extract_native_pages(path)
        extracted: dict[int, ExtractedPage] = {}
        pages_requiring_images = []
        for page_number, native_text in enumerate(native_pages, start=1):
            if self._native_text_is_usable(native_text):
                extracted[page_number] = ExtractedPage(
                    page_number=page_number,
                    text=native_text.strip(),
                    extraction_method="native",
                )
                continue
            page_cache, segment_count = self._read_page_cache_record(path, page_number)
            if page_cache:
                extracted[page_number] = ExtractedPage(
                    page_number=page_number,
                    text=page_cache,
                    extraction_method="ocr_cache",
                    segment_count=segment_count,
                )
            elif self._blank_cache_path(path, page_number).is_file():
                extracted[page_number] = ExtractedPage(
                    page_number=page_number,
                    text="",
                    extraction_method="blank_check",
                    is_blank=True,
                )
            else:
                pages_requiring_images.append(page_number)

        image_directory_exists = path.with_suffix("").is_dir()
        if pages_requiring_images or image_directory_exists:
            try:
                image_paths = self._image_service.render_pages(
                    path,
                    page_numbers=pages_requiring_images,
                    prune=True,
                )
            except PdfImageRenderError as exc:
                raise OCRExtractionError(str(exc)) from None
            except Exception:
                raise OCRExtractionError(
                    f"{path.name}: failed to render PDF pages"
                ) from None
            if len(image_paths) != len(pages_requiring_images):
                raise OCRExtractionError(
                    f"{path.name}: PDF rendering returned an unexpected page count"
                )
            for page_number, page_path in zip(
                pages_requiring_images,
                image_paths,
                strict=True,
            ):
                if self._image_is_blank(page_path):
                    page_path.unlink(missing_ok=True)
                    self._write_cache(
                        self._blank_cache_path(path, page_number),
                        "blank",
                    )
                    extracted[page_number] = ExtractedPage(
                        page_number=page_number,
                        text="",
                        extraction_method="blank_check",
                        is_blank=True,
                    )
                    continue
                segmenter = getattr(
                    self._image_service,
                    "segment_page_image",
                    lambda candidate: [candidate],
                )
                segments = segmenter(page_path)
                tiler = getattr(
                    self._image_service,
                    "tile_ocr_image",
                    lambda candidate: [candidate],
                )
                ocr_images = [
                    tile
                    for segment in segments
                    for tile in tiler(segment)
                ]
                segment_texts = []
                for segment_number, segment_path in enumerate(ocr_images, start=1):
                    segment_texts.append(
                        self._extract_page(
                            path.name,
                            page_number,
                            segment_path,
                            segment_number=(
                                segment_number if len(ocr_images) > 1 else None
                            ),
                        )
                    )
                page_text = "\n".join(segment_texts)
                self._write_cache(
                    self._page_cache_path(path, page_number),
                    page_text,
                )
                self._write_cache(
                    self._page_metadata_path(path, page_number),
                    json.dumps({"segment_count": len(ocr_images)}),
                )
                extracted[page_number] = ExtractedPage(
                    page_number=page_number,
                    text=page_text,
                    extraction_method="ocr",
                    segment_count=len(ocr_images),
                )

        return [extracted[number] for number in range(1, len(native_pages) + 1)]

    @staticmethod
    def _extract_native_pages(path: Path) -> list[str]:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages

    @staticmethod
    def _native_text_is_usable(text: str) -> bool:
        visible = "".join(character for character in text if not character.isspace())
        if len(visible) < MIN_NATIVE_TEXT_CHARACTERS:
            return False
        return visible.count("\ufffd") / len(visible) < 0.05

    def _read_page_cache(self, path: Path, page_number: int) -> str:
        return self._read_page_cache_record(path, page_number)[0]

    def _read_page_cache_record(self, path: Path, page_number: int) -> tuple[str, int]:
        cache_path = self._page_cache_path(path, page_number)
        try:
            text = self._normalize_ocr_output(
                cache_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError):
            return "", 1
        if self._output_is_invalid(text):
            return "", 1
        try:
            metadata = json.loads(
                self._page_metadata_path(path, page_number).read_text(encoding="utf-8")
            )
            segment_count = max(1, int(metadata.get("segment_count", 1)))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            segment_count = 1
        return text, segment_count

    @staticmethod
    def _image_is_blank(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                grayscale = image.convert("L")
                histogram = grayscale.histogram()
                pixel_count = grayscale.width * grayscale.height
        except (OSError, ValueError):
            return False
        ink_pixels = sum(histogram[:245])
        return ink_pixels / pixel_count < MIN_IMAGE_INK_RATIO

    def _extract_page(
        self,
        document_name: str,
        page_number: int,
        page_path: Path,
        segment_number: int | None = None,
    ) -> str:
        location = f"page {page_number}"
        if segment_number is not None:
            location += f" segment {segment_number}"
        encoded = base64.b64encode(page_path.read_bytes())
        if len(encoded) > MAX_BASE64_IMAGE_BYTES:
            raise OCRExtractionError(
                f"{document_name}: {location} exceeds the 10 MB Base64 limit"
            )
        data_url = f"data:image/jpeg;base64,{encoded.decode('ascii')}"
        for prompt in (None, STRICT_OCR_PROMPT):
            content_parts = []
            if prompt is not None:
                content_parts.append({"type": "text", "text": prompt})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                    "max_pixels": 32 * 32 * 8192,
                }
            )
            try:
                completions = self._get_chat_client().chat.completions
                response = completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ],
                )
                content = response.choices[0].message.content
            except Exception as exc:
                raise OCRExtractionError(
                    f"{document_name}: Qwen OCR request failed on {location} "
                    f"({type(exc).__name__})"
                ) from None

            raw_text = content.strip() if isinstance(content, str) else ""
            text = self._normalize_ocr_output(raw_text)
            if text and not self._output_is_invalid(text):
                return text
            if not any(
                marker in raw_text.lower() for marker in RETRYABLE_FORMAT_MARKERS
            ):
                break
        raise OCRExtractionError(
            f"{document_name}: {location} returned empty or invalid OCR text"
        )

    @staticmethod
    def _output_is_invalid(text: str) -> bool:
        lowered = text.lower()
        return any(marker in text for marker in INVALID_OUTPUT_MARKERS) or any(
            marker in lowered for marker in RETRYABLE_FORMAT_MARKERS
        )

    @staticmethod
    def _normalize_ocr_output(text: str) -> str:
        """Remove response wrappers without changing visible OCR wording."""
        stripped = text.strip()
        if stripped.lower().startswith("```html"):
            stripped = stripped[7:].lstrip("\r\n")
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
        if not _HTML_TAG.search(stripped):
            return stripped

        parser = _VisibleTextHTMLParser()
        try:
            parser.feed(stripped)
            parser.close()
        except ValueError:
            return stripped
        lines = []
        for line in "".join(parser.parts).splitlines():
            normalized = " ".join(line.split())
            if normalized:
                lines.append(normalized)
        return "\n".join(lines)

    def _get_chat_client(self):
        if self._chat_client is None:
            try:
                from openai import OpenAI

                self._chat_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries,
                )
            except Exception:
                raise OCRExtractionError(
                    "failed to initialize Qwen OCR client"
                ) from None
        return self._chat_client
