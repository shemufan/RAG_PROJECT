"""PDF validation and cached Qwen OCR text extraction."""

import hashlib
from pathlib import Path
from uuid import uuid4

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 50
OCR_PROMPT_VERSION = "legal-document-v1"
UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
LEGAL_OCR_PROMPT = (
    "完整提取该法规文档中的所有文字，严格保持原始阅读顺序，并保留标题、章节、"
    "条款编号和自然段。不要总结、解释、改写或补充内容；无法辨认的字符使用 ? 表示。"
)


class OCRExtractionError(RuntimeError):
    """Raised when a PDF cannot be safely converted to knowledge text."""


class QwenOCRService:
    """Validate PDFs and cache text produced by an injected or Qwen extractor."""

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
        http_session=None,
        responses_client=None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._extractor = extractor
        self._http_session = http_session
        self._responses_client = responses_client

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
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        digest.update(self.model.encode("utf-8"))
        digest.update(OCR_PROMPT_VERSION.encode("utf-8"))
        return self.cache_dir / f"{digest.hexdigest()}.txt"

    def _write_cache(self, cache_path: Path, text: str) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _extract_with_qwen(self, path: Path) -> str:
        file_url = self._upload_pdf(path)
        return self._call_qwen(path.name, file_url)

    def _upload_pdf(self, path: Path) -> str:
        session = self._http_session or requests.Session()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            policy_response = session.get(
                UPLOAD_POLICY_URL,
                headers=headers,
                params={"action": "getPolicy", "model": self.model},
                timeout=self.timeout_seconds,
            )
            policy_response.raise_for_status()
            policy = policy_response.json()["data"]
            key = f"{policy['upload_dir']}/{path.name}"
        except (requests.RequestException, KeyError, TypeError, ValueError):
            raise OCRExtractionError(
                f"{path.name}: failed to obtain Qwen upload policy"
            ) from None

        try:
            with path.open("rb") as pdf_stream:
                files = {
                    "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
                    "Signature": (None, policy["signature"]),
                    "policy": (None, policy["policy"]),
                    "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
                    "x-oss-forbid-overwrite": (
                        None,
                        policy["x_oss_forbid_overwrite"],
                    ),
                    "key": (None, key),
                    "success_action_status": (None, "200"),
                    "file": (path.name, pdf_stream, "application/pdf"),
                }
                upload_response = session.post(
                    policy["upload_host"],
                    files=files,
                    timeout=self.timeout_seconds,
                )
                upload_response.raise_for_status()
        except (requests.RequestException, KeyError, OSError, TypeError):
            raise OCRExtractionError(
                f"{path.name}: failed to upload PDF for OCR"
            ) from None
        return f"oss://{key}"

    def _call_qwen(self, document_name: str, file_url: str) -> str:
        responses = self._responses_client
        if responses is None:
            try:
                from openai import OpenAI

                responses = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries,
                ).responses
            except Exception:
                raise OCRExtractionError(
                    f"{document_name}: failed to initialize Qwen OCR client"
                ) from None
        try:
            response = responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_url": file_url},
                            {"type": "input_text", "text": LEGAL_OCR_PROMPT},
                        ],
                    }
                ],
                extra_body={"ocr_options": {"task": "document_parsing"}},
            )
        except Exception:
            raise OCRExtractionError(
                f"{document_name}: Qwen OCR request failed"
            ) from None
        output_text = getattr(response, "output_text", None)
        return output_text.strip() if isinstance(output_text, str) else ""
