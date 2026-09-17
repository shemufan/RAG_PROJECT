import os
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfWriter

from app.services.pdf_image_service import PdfImageService


def write_pdf(path: Path, *, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as stream:
        writer.write(stream)


def test_render_pages_creates_ordered_images_in_pdf_named_directory(tmp_path):
    pdf_path = tmp_path / "个人信息安全规范.pdf"
    write_pdf(pdf_path, pages=2)

    pages = PdfImageService().render_pages(pdf_path)

    assert [page.name for page in pages] == ["page-0001.jpg", "page-0002.jpg"]
    assert all(page.parent == tmp_path / "个人信息安全规范" for page in pages)
    assert all(page.read_bytes().startswith(b"\xff\xd8") for page in pages)
    assert (tmp_path / "个人信息安全规范" / ".pdf-pages.json").is_file()


def test_render_pages_reuses_images_when_pdf_and_settings_match(tmp_path):
    pdf_path = tmp_path / "law.pdf"
    write_pdf(pdf_path, pages=1)
    service = PdfImageService()
    page = service.render_pages(pdf_path)[0]
    old_timestamp = 946684800
    os.utime(page, (old_timestamp, old_timestamp))

    reused = service.render_pages(pdf_path)

    assert reused == [page]
    assert page.stat().st_mtime == old_timestamp


def test_render_pages_replaces_only_generated_files_when_pdf_changes(tmp_path):
    pdf_path = tmp_path / "law.pdf"
    write_pdf(pdf_path, pages=2)
    service = PdfImageService()
    output_dir = pdf_path.with_suffix("")
    service.render_pages(pdf_path)
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    write_pdf(pdf_path, pages=1)
    pages = service.render_pages(pdf_path)

    assert [page.name for page in pages] == ["page-0001.jpg"]
    assert not (output_dir / "page-0002.jpg").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_render_pages_generates_only_requested_pages_and_reuses_them(tmp_path):
    pdf_path = tmp_path / "mixed.pdf"
    write_pdf(pdf_path, pages=3)
    service = PdfImageService()

    second_page = service.render_pages(pdf_path, page_numbers=[2])
    first_page = service.render_pages(pdf_path, page_numbers=[1])

    output_dir = pdf_path.with_suffix("")
    assert second_page == [output_dir / "page-0002.jpg"]
    assert first_page == [output_dir / "page-0001.jpg"]
    assert (output_dir / "page-0002.jpg").is_file()
    assert not (output_dir / "page-0003.jpg").exists()


def test_render_pages_prunes_images_not_requested_by_current_extraction(tmp_path):
    pdf_path = tmp_path / "law.pdf"
    write_pdf(pdf_path, pages=3)
    service = PdfImageService()
    output_dir = pdf_path.with_suffix("")
    service.render_pages(pdf_path)

    pages = service.render_pages(pdf_path, page_numbers=[2], prune=True)

    assert pages == [output_dir / "page-0002.jpg"]
    assert not (output_dir / "page-0001.jpg").exists()
    assert not (output_dir / "page-0003.jpg").exists()


def test_segment_page_image_splits_abnormal_long_image_at_white_gutters(tmp_path):
    page = tmp_path / "page-0001.jpg"
    image = Image.new("RGB", (120, 720), "white")
    draw = ImageDraw.Draw(image)
    for top in (20, 260, 500):
        draw.rectangle((10, top, 110, top + 160), fill="black")
    image.save(page, format="JPEG", quality=100)
    image.close()

    segments = PdfImageService().segment_page_image(page)

    assert [item.name for item in segments] == [
        "page-0001-segment-0001.jpg",
        "page-0001-segment-0002.jpg",
        "page-0001-segment-0003.jpg",
    ]
    assert all(item.is_file() for item in segments)


def test_segment_page_image_keeps_normal_page_unchanged(tmp_path):
    page = tmp_path / "page-0001.jpg"
    Image.new("RGB", (200, 280), "black").save(page, format="JPEG")

    assert PdfImageService().segment_page_image(page) == [page]


def test_segment_page_image_does_not_guess_without_reliable_gutters(tmp_path):
    page = tmp_path / "page-0001.jpg"
    Image.new("RGB", (100, 500), "black").save(page, format="JPEG")

    assert PdfImageService().segment_page_image(page) == [page]


def test_segment_page_image_ignores_small_internal_whitespace_runs(tmp_path):
    page = tmp_path / "page-0001.jpg"
    image = Image.new("RGB", (100, 600), "white")
    draw = ImageDraw.Draw(image)
    for page_top in (0, 150, 300, 450):
        for offset in (10, 55, 100):
            draw.rectangle(
                (8, page_top + offset, 92, page_top + offset + 25),
                fill="black",
            )
    image.save(page, format="JPEG", quality=100)
    image.close()

    segments = PdfImageService().segment_page_image(page)

    assert len(segments) == 4


def test_segment_page_image_detects_narrow_gutters_in_very_long_scan(tmp_path):
    page = tmp_path / "page-0001.jpg"
    image = Image.new("RGB", (100, 600), "white")
    draw = ImageDraw.Draw(image)
    for page_top in (0, 150, 300, 450):
        draw.rectangle((5, page_top + 2, 95, page_top + 145), fill="black")
    image.save(page, format="JPEG", quality=100)
    image.close()

    assert len(PdfImageService().segment_page_image(page)) == 4


def test_tile_ocr_image_splits_page_at_central_white_band(tmp_path):
    page = tmp_path / "page-0001-segment-0001.jpg"
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 190, 120), fill="black")
    draw.rectangle((10, 180, 190, 290), fill="black")
    image.save(page, format="JPEG", quality=100)
    image.close()

    tiles = PdfImageService().tile_ocr_image(page)

    assert [tile.name for tile in tiles] == [
        "page-0001-segment-0001-tile-0001.jpg",
        "page-0001-segment-0001-tile-0002.jpg",
    ]


def test_tile_ocr_image_keeps_page_when_center_has_no_safe_gap(tmp_path):
    page = tmp_path / "page-0001.jpg"
    Image.new("RGB", (200, 300), "black").save(page, format="JPEG")

    assert PdfImageService().tile_ocr_image(page) == [page]
