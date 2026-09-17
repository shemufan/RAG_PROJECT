"""Render PDF pages to reusable local JPEG images."""

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

import pypdfium2 as pdfium
from PIL import Image

MANIFEST_NAME = ".pdf-pages.json"
RENDER_VERSION = "pdfium-jpeg-v1"
PAGE_SEGMENT_VERSION = "projection-gutter-tile-v3"
PAGE_NAME_PATTERN = re.compile(r"page-\d{4}\.jpg")
LONG_PAGE_ASPECT_RATIO = 2.2
MIN_GUTTER_RATIO = 0.02
MAX_GUTTER_INK_RATIO = 0.01


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

    def segment_page_image(self, page_path: str | Path) -> list[Path]:
        """Split an unusually tall scan at reliable horizontal white gutters."""
        path = Path(page_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        try:
            width, height = image.size
            if height / max(width, 1) < LONG_PAGE_ASPECT_RATIO:
                return [path]
            grayscale = image.convert("L")
            row_means = list(
                grayscale.resize(
                    (1, height),
                    resample=Image.Resampling.BOX,
                ).get_flattened_data()
            )
            blank_rows = [
                (255 - value) / 255 <= MAX_GUTTER_INK_RATIO for value in row_means
            ]
            minimum_gutter = max(4, round(width * MIN_GUTTER_RATIO))
            candidates = self._gutter_centers(blank_rows, minimum_gutter)
            cuts = self._select_page_gutters(candidates, width, height)
            if not cuts:
                return [path]

            boundaries = [0, *cuts, height]
            segments = []
            for index, (top, bottom) in enumerate(
                zip(boundaries, boundaries[1:]),
                start=1,
            ):
                if bottom - top < minimum_gutter * 2:
                    return [path]
                output = path.with_name(f"{path.stem}-segment-{index:04d}.jpg")
                image.crop((0, top, width, bottom)).save(
                    output,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                )
                segments.append(output)
            return segments
        finally:
            image.close()

    def tile_ocr_image(self, page_path: str | Path) -> list[Path]:
        """Split one physical page at a central whitespace band for safer OCR."""
        path = Path(page_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        try:
            width, height = image.size
            if height / max(width, 1) < 1.1:
                return [path]
            grayscale = image.convert("L")
            row_means = list(
                grayscale.resize(
                    (1, height),
                    resample=Image.Resampling.BOX,
                ).get_flattened_data()
            )
            blank_rows = [
                (255 - value) / 255 <= MAX_GUTTER_INK_RATIO for value in row_means
            ]
            minimum_gutter = max(4, round(width * MIN_GUTTER_RATIO))
            candidates = self._gutter_centers(blank_rows, minimum_gutter)
            candidates = [
                value for value in candidates if height * 0.3 <= value <= height * 0.7
            ]
            if not candidates:
                return [path]
            boundary = min(candidates, key=lambda value: abs(value - height / 2))
            outputs = []
            for index, (top, bottom) in enumerate(
                ((0, boundary), (boundary, height)),
                start=1,
            ):
                output = path.with_name(f"{path.stem}-tile-{index:04d}.jpg")
                image.crop((0, top, width, bottom)).save(
                    output,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                )
                outputs.append(output)
            return outputs
        finally:
            image.close()

    @staticmethod
    def _gutter_centers(blank_rows: list[bool], minimum_length: int) -> list[int]:
        centers = []
        start = None
        for index, is_blank in enumerate([*blank_rows, False]):
            if is_blank and start is None:
                start = index
            elif not is_blank and start is not None:
                if (
                    start > 0
                    and index < len(blank_rows)
                    and index - start >= minimum_length
                ):
                    centers.append((start + index) // 2)
                start = None
        return centers

    @staticmethod
    def _select_page_gutters(candidates: list[int], width: int, height: int) -> list[int]:
        """Choose page-sized boundaries instead of paragraph whitespace."""
        selected = []
        previous = 0
        target_height = width * 1.414
        minimum_height = width * 0.75
        maximum_height = width * 2.2
        while height - previous > maximum_height:
            choices = [
                value
                for value in candidates
                if minimum_height <= value - previous <= maximum_height
                and height - value >= minimum_height
            ]
            if not choices:
                return []
            target = previous + target_height
            boundary = min(choices, key=lambda value: abs(value - target))
            selected.append(boundary)
            previous = boundary
        return selected

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
