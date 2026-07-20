"""UTF-8 aware article/chapter chunking for Chinese compliance documents."""

import re
import uuid

from langchain_core.documents import Document

CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章")
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+条")
MARKDOWN_RULE_RE = re.compile(r"^#{2,}\s*规则\s*\d+")


def split_by_article(
    text: str,
    document_name: str,
    source_type: str = "enterprise_rule",
    domain: str = "general",
):
    chunks: list[Document] = []
    chapter: str | None = None
    article: str | None = None
    buffer: list[str] = []

    def flush():
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if not content:
            return
        metadata = {
            "chunk_id": str(uuid.uuid4()),
            "document_name": document_name,
            "source_type": source_type,
            "domain": domain,
            "chapter": chapter or "未标注章节",
            "article": article or "未标注条款",
            "hierarchy_level": " / ".join(value for value in [chapter, article] if value),
            "version": "week3",
        }
        for level in ("L4", "L3", "L2", "L1"):
            if level in content:
                metadata["sensitivity_level"] = level
                break
        chunks.append(Document(page_content=content, metadata=metadata))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if CHAPTER_RE.match(line):
            flush()
            buffer = []
            chapter = line
            article = None
            continue
        if ARTICLE_RE.match(line):
            flush()
            buffer = []
            article = line[:80]
        elif MARKDOWN_RULE_RE.match(line):
            if article is not None:
                flush()
            buffer = []
            article = line.lstrip("# ")[:80]
        buffer.append(line)
    flush()
    return chunks
