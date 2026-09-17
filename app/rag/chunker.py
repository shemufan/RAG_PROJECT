"""Structure-aware splitting for laws, standards, and classification rules."""

import re
import uuid

from langchain_core.documents import Document

from app.schemas.knowledge_quality import ExtractedPage

_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百0-9]+章")
_ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百0-9]+条")
_RULE_PATTERN = re.compile(r"^#{2,}\s*规则\s*([0-9一二三四五六七八九十]+)[：:]?")
_STANDARD_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+\S")
_APPENDIX_PATTERN = re.compile(r"^附录\s+([A-ZＡ-Ｚ一二三四五六七八九十]+)")
_APPENDIX_ARTICLE_PATTERN = re.compile(r"^([A-ZＡ-Ｚ]\.(?:\d+\.?)+)\s+\S")
_MAX_CHUNK_CHARACTERS = 1600
_CHUNK_OVERLAP = 150


def split_knowledge_text(
    text: str,
    document_name: str,
    *,
    source_type: str,
    version: str,
    source_format: str = "txt",
    source_sha256: str | None = None,
) -> list[Document]:
    """Split one text source at legal, standard, or Markdown boundaries."""
    lines = [(line, None) for line in text.splitlines()]
    return _split_lines(
        lines,
        document_name,
        source_type=source_type,
        version=version,
        source_format=source_format,
        source_sha256=source_sha256,
    )


def split_knowledge_pages(
    pages: list[ExtractedPage],
    document_name: str,
    *,
    source_type: str,
    version: str,
    source_format: str = "pdf",
    source_sha256: str | None = None,
) -> list[Document]:
    """Split page-aware text while preserving each chunk's source page range."""
    lines = [
        (line, page.page_number)
        for page in pages
        for line in page.text.splitlines()
    ]
    return _split_lines(
        lines,
        document_name,
        source_type=source_type,
        version=version,
        source_format=source_format,
        source_sha256=source_sha256,
    )


def _split_lines(
    lines: list[tuple[str, int | None]],
    document_name: str,
    *,
    source_type: str,
    version: str,
    source_format: str,
    source_sha256: str | None,
) -> list[Document]:
    chunks: list[Document] = []
    buffer: list[tuple[str, int | None]] = []
    chapter = "未标注章节"
    article = "全文"

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(line for line, _ in buffer).strip()
        if not content:
            buffer = []
            return
        page_numbers = [page for _, page in buffer if page is not None]
        parts = _split_long_content(content)
        for part_number, part in enumerate(parts, start=1):
            identity = "|".join(
                [
                    source_sha256 or document_name,
                    version,
                    chapter,
                    article,
                    str(part_number),
                    part,
                ]
            )
            metadata = {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                "document_name": document_name,
                "source_type": source_type,
                "chapter": chapter,
                "article": article,
                "version": version,
                "source_format": source_format,
            }
            if source_sha256 is not None:
                metadata["source_sha256"] = source_sha256
            if page_numbers:
                metadata["page_start"] = min(page_numbers)
                metadata["page_end"] = max(page_numbers)
            if len(parts) > 1:
                metadata["part"] = part_number
            for level in ("L4", "L3", "L2", "L1"):
                if level in part:
                    metadata["sensitivity_level"] = level
                    break
            chunks.append(Document(page_content=part, metadata=metadata))
        buffer = []

    for raw_line, page_number in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _CHAPTER_PATTERN.match(line):
            flush()
            chapter = line
            article = "章说明"
            buffer.append((line, page_number))
            continue
        if _APPENDIX_PATTERN.match(line):
            flush()
            chapter = line
            article = "附录说明"
            buffer.append((line, page_number))
            continue
        if _ARTICLE_PATTERN.match(line):
            flush()
            article = line[:80]
        elif _APPENDIX_ARTICLE_PATTERN.match(line):
            flush()
            article = line[:80]
        else:
            standard_match = _STANDARD_PATTERN.match(line)
            if standard_match:
                flush()
                number = standard_match.group(1)
                if "." not in number:
                    chapter = line[:80]
                article = line[:80]
            else:
                rule_match = _RULE_PATTERN.match(line)
                if rule_match:
                    if any(not item[0].startswith("# ") for item in buffer):
                        flush()
                    buffer = []
                    article = f"规则 {rule_match.group(1)}：{line.split('：', 1)[-1]}"
        buffer.append((line, page_number))

    flush()
    return chunks


def _split_long_content(content: str) -> list[str]:
    if len(content) <= _MAX_CHUNK_CHARACTERS:
        return [content]
    parts = []
    start = 0
    while start < len(content):
        end = min(start + _MAX_CHUNK_CHARACTERS, len(content))
        parts.append(content[start:end])
        if end == len(content):
            break
        start = end - _CHUNK_OVERLAP
    return parts
