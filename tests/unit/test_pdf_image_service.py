import os
from pathlib import Path

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
