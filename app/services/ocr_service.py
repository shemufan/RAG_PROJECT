"""PDF validation and cached Qwen OCR text extraction."""

import base64
import hashlib
from pathlib import Path
from uuid import uuid4

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.services.pdf_image_service import PdfImageRenderError, PdfImageService

MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_BASE64_IMAGE_BYTES = 10 * 1024 * 1024
MIN_NATIVE_TEXT_CHARACTERS = 5
MIN_IMAGE_INK_RATIO = 0.0005
OCR_PROMPT_VERSION = "legal-hybrid-page-v3"
LEGAL_OCR_PROMPT = (
    "完整提取该法规页面中的所有文字，严格保持原始阅读顺序，并保留标题、章节、"
    "条款编号和自然段。不要总结、解释、改写或补充内容；无法辨认的字符使用 ? 表示。"
)
INVALID_OUTPUT_MARKERS = (
    "没有可供提取",
    "未提供法规文档",
    "未提供文档",
    "无法提取",
)


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
        return digest.hexdigest()

    def _page_cache_path(self, path: Path, page_number: int) -> Path:
        digest = self._cache_digest(path)[:20]
        return self.cache_dir / f"page-{digest}-{page_number:04d}.txt"

    def _blank_cache_path(self, path: Path, page_number: int) -> Path:
        digest = self._cache_digest(path)[:20]
        return self.cache_dir / f"blank-{digest}-{page_number:04d}.flag"

    def _write_cache(self, cache_path: Path, text: str) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _extract_with_qwen(self, path: Path) -> str:
        native_pages = self._extract_native_pages(path)
        extracted: dict[int, str] = {}
        pages_requiring_images = []
        for page_number, native_text in enumerate(native_pages, start=1):
            if self._native_text_is_usable(native_text):
                extracted[page_number] = native_text.strip()
                continue
            page_cache = self._read_page_cache(path, page_number)
            if page_cache:
                extracted[page_number] = page_cache
            elif self._blank_cache_path(path, page_number).is_file():
                extracted[page_number] = ""
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
                    extracted[page_number] = ""
                    continue
                page_text = self._extract_page(path.name, page_number, page_path)
                self._write_cache(
                    self._page_cache_path(path, page_number),
                    page_text,
                )
                extracted[page_number] = page_text

        ordered = []
        for page_number in range(1, len(native_pages) + 1):
            page_text = extracted[page_number]
            if page_text:
                ordered.append(f"【第 {page_number} 页】\n{page_text}")
        return "\n\n".join(ordered)

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
        cache_path = self._page_cache_path(path, page_number)
        try:
            text = cache_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""
        if any(marker in text for marker in INVALID_OUTPUT_MARKERS):
            return ""
        return text

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
    ) -> str:
        encoded = base64.b64encode(page_path.read_bytes())
        if len(encoded) > MAX_BASE64_IMAGE_BYTES:
            raise OCRExtractionError(
                f"{document_name}: page {page_number} exceeds the 10 MB Base64 limit"
            )
        data_url = f"data:image/jpeg;base64,{encoded.decode('ascii')}"
        try:
            completions = self._get_chat_client().chat.completions
            response = completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {"type": "text", "text": LEGAL_OCR_PROMPT},
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise OCRExtractionError(
                f"{document_name}: Qwen OCR request failed on page {page_number} "
                f"({type(exc).__name__})"
            ) from None

        text = content.strip() if isinstance(content, str) else ""
        if not text or any(marker in text for marker in INVALID_OUTPUT_MARKERS):
            raise OCRExtractionError(
                f"{document_name}: page {page_number} returned empty or invalid OCR text"
            )
        return text

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
