"""Render PDF pages to reusable local JPEG images."""

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

import pypdfium2 as pdfium

MANIFEST_NAME = ".pdf-pages.json"
RENDER_VERSION = "pdfium-jpeg-v1"
PAGE_NAME_PATTERN = re.compile(r"page-\d{4}\.jpg")


class PdfImageRenderError(RuntimeError):
    """Raised when a PDF page cannot be rendered safely."""


class PdfImageService:
    """Render PDF pages beside their source and reuse valid results."""

    def __init__(self, *, dpi: int = 200, jpeg_quality: int = 90):
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality

    def render_pages(
        self,
        pdf_path: str | Path,
        page_numbers: list[int] | None = None,
        *,
        prune: bool = False,
    ) -> list[Path]:
        path = Path(pdf_path)
        output_dir = path.with_suffix("")
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise PdfImageRenderError(f"{path.name}: failed to open PDF") from exc

        try:
            page_count = len(document)
            requested = self._requested_pages(page_numbers, page_count)
            manifest_base = self._expected_manifest(path, page_count)
            output_dir.mkdir(parents=True, exist_ok=True)
            existing = self._read_manifest(output_dir / MANIFEST_NAME)
            if not self._manifest_matches(existing, manifest_base):
                self._remove_generated_files(output_dir)
                rendered_pages: set[int] = set()
                manifest_changed = True
            else:
                listed_pages = set(existing.get("rendered_pages", []))
                rendered_pages = {
                    number
                    for number in listed_pages
                    if self._page_path(output_dir, number).is_file()
                }
                manifest_changed = rendered_pages != listed_pages

            if prune:
                retained = set(requested)
                for child in output_dir.iterdir():
                    if child.is_file() and PAGE_NAME_PATTERN.fullmatch(child.name):
                        page_number = int(child.stem.removeprefix("page-"))
                        if page_number not in retained:
                            child.unlink()
                            manifest_changed = True
                if not rendered_pages.issubset(retained):
                    rendered_pages.intersection_update(retained)
                    manifest_changed = True

            missing = [number for number in requested if number not in rendered_pages]
            if missing:
                self._render_document(document, path.name, output_dir, missing)
                rendered_pages.update(missing)
                manifest_changed = True
            if manifest_changed:
                manifest = {**manifest_base, "rendered_pages": sorted(rendered_pages)}
                self._write_manifest(output_dir / MANIFEST_NAME, manifest)
            return [self._page_path(output_dir, number) for number in requested]
        finally:
            document.close()

    def _expected_manifest(self, path: Path, page_count: int) -> dict:
        return {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": page_count,
            "dpi": self.dpi,
            "jpeg_quality": self.jpeg_quality,
            "render_version": RENDER_VERSION,
        }

    @staticmethod
    def _requested_pages(
        page_numbers: list[int] | None,
        page_count: int,
    ) -> list[int]:
        requested = list(range(1, page_count + 1)) if page_numbers is None else page_numbers
        if len(set(requested)) != len(requested):
            raise ValueError("page numbers must be unique")
        if any(number < 1 or number > page_count for number in requested):
            raise ValueError("page number is outside the PDF page range")
        return requested

    @staticmethod
    def _page_path(output_dir: Path, page_number: int) -> Path:
        return output_dir / f"page-{page_number:04d}.jpg"

    @staticmethod
    def _read_manifest(manifest_path: Path) -> dict:
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _manifest_matches(actual: dict, expected: dict) -> bool:
        return all(actual.get(key) == value for key, value in expected.items())

    @staticmethod
    def _remove_generated_files(output_dir: Path) -> None:
        for child in output_dir.iterdir():
            if child.is_file() and (
                PAGE_NAME_PATTERN.fullmatch(child.name) or child.name == MANIFEST_NAME
            ):
                child.unlink()

    def _render_document(
        self,
        document,
        document_name: str,
        output_dir: Path,
        page_numbers: list[int],
    ) -> None:
        scale = self.dpi / 72
        for page_number in page_numbers:
            output_path = self._page_path(output_dir, page_number)
            temporary = output_path.with_name(
                f".{output_path.name}.{uuid4().hex}.tmp"
            )
            try:
                page = document[page_number - 1]
                try:
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil().convert("RGB")
                    try:
                        image.save(
                            temporary,
                            format="JPEG",
                            quality=self.jpeg_quality,
                            optimize=True,
                        )
                    finally:
                        image.close()
                        bitmap.close()
                finally:
                    page.close()
                temporary.replace(output_path)
            except Exception as exc:
                raise PdfImageRenderError(
                    f"{document_name}: failed to render page {page_number}"
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_manifest(path: Path, manifest: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
